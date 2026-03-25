#!/usr/bin/env python3
"""Chidori — Cached inference test with StaticCache

Uses HuggingFace StaticCache to pre-allocate KV cache at max_seq_len.
All matmul dimensions become fixed per token, enabling:
  1. Heuristic cache hits (same dimensions = same cache key)
  2. Server-side replay (fingerprint matches every token)

Expected: faster than dynamic KV cache inference over network.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import time
import sys
import math

torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Device: {dev} ({vram_gb:.1f} GB)")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Explain how GPU-over-IP works in simple terms:"
MAX_NEW_TOKENS = 200

# ============================================================
# LOAD PHASE
# ============================================================
print(f"\n[LOAD] Loading {MODEL}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="eager"
)
load_cpu_dt = time.time() - t0
print(f"[LOAD] CPU load: {load_cpu_dt:.1f}s")

print("[LOAD] Moving to GPU...")
t0 = time.time()
model = model.cuda()
model.eval()
torch.cuda.synchronize()
load_gpu_dt = time.time() - t0

mem_mb = torch.cuda.memory_allocated() / 1e6
print(f"[LOAD] GPU transfer: {load_gpu_dt:.1f}s ({mem_mb:.0f} MB)")

# ============================================================
# SETUP STATIC CACHE
# ============================================================
inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
prompt_len = inputs["input_ids"].shape[1]
max_cache_len = prompt_len + MAX_NEW_TOKENS + 16  # small padding

print(f"\n[CACHE] Setting up StaticCache (max_len={max_cache_len})")
t0 = time.time()
static_cache = StaticCache(
    config=model.config,
    max_cache_len=max_cache_len,
)
torch.cuda.synchronize()
cache_dt = time.time() - t0
print(f"[CACHE] Created in {cache_dt:.1f}s (lazy init on first forward)")

# ============================================================
# BASELINE: Dynamic KV cache (standard generate, 20 tokens)
# ============================================================
print(f"\n[BASELINE] Generating 20 tokens with dynamic KV cache...")
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    baseline_out = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )
torch.cuda.synchronize()
baseline_dt = time.time() - t0
baseline_toks = baseline_out.shape[1] - prompt_len
baseline_tps = baseline_toks / baseline_dt
print(f"[BASELINE] {baseline_toks} tokens in {baseline_dt:.1f}s = {baseline_tps:.1f} tok/s")

# ============================================================
# CACHED: Static KV cache inference
# ============================================================
print(f"\n[CACHED] Generating {MAX_NEW_TOKENS} tokens with StaticCache...")

# Reset cache state
static_cache.reset()

torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    cached_out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        past_key_values=static_cache,
    )
torch.cuda.synchronize()
cached_dt = time.time() - t0

cached_toks = cached_out.shape[1] - prompt_len
cached_tps = cached_toks / cached_dt

print(f"[CACHED] {cached_toks} tokens in {cached_dt:.1f}s = {cached_tps:.1f} tok/s")

# ============================================================
# VERIFY OUTPUT
# ============================================================
text = tokenizer.decode(cached_out[0].cpu().tolist(), skip_special_tokens=True)
print(text[:500])

# ============================================================
# SERVER-SIDE GRAPH DECODE (if available)
# ============================================================
server_tps = 0
server_toks = 0
try:
    import ctypes
    lib = ctypes.CDLL("build/libcuda.so.1")
    if hasattr(lib, 'chidori_graph_decode'):
        print(f"\n[SERVER] Setting up CUDA Graph + server-side decode...")

        static_cache2 = StaticCache(config=model.config, max_cache_len=max_cache_len)
        with torch.no_grad():
            pf = model(inputs["input_ids"], past_key_values=static_cache2, use_cache=True)
            nt = pf.logits[:, -1:, :].argmax(-1)
        cp = torch.tensor([prompt_len], device="cuda")
        with torch.no_grad():
            model(nt, past_key_values=static_cache2, use_cache=True, cache_position=cp)
        torch.cuda.synchronize()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        si = nt.clone()
        sp = torch.tensor([prompt_len + 1], device="cuda", dtype=torch.long)
        with torch.cuda.stream(s):
            with torch.no_grad():
                so = model(si, past_key_values=static_cache2, use_cache=True, cache_position=sp)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        sptr = ctypes.c_void_p(s.cuda_stream)
        lib.cuStreamBeginCapture_v2(sptr, ctypes.c_int(0))
        with torch.cuda.stream(s):
            with torch.no_grad():
                so = model(si, past_key_values=static_cache2, use_cache=True, cache_position=sp)
        gptr = ctypes.c_void_p(0)
        lib.cuStreamEndCapture(sptr, ctypes.byref(gptr))
        eptr = ctypes.c_void_p(0)
        lib.cuGraphInstantiate(ctypes.byref(eptr), gptr, ctypes.c_ulonglong(0))

        out_tokens = (ctypes.c_int32 * MAX_NEW_TOKENS)()
        out_count = ctypes.c_int32(0)
        torch.cuda.synchronize()
        t0 = time.time()
        lib.chidori_graph_decode(
            ctypes.c_uint64(eptr.value), ctypes.c_uint64(sptr.value),
            ctypes.c_uint64(si.data_ptr()), ctypes.c_uint64(sp.data_ptr()),
            ctypes.c_uint64(so.logits[:, -1:, :].contiguous().data_ptr()),
            ctypes.c_int32(model.config.vocab_size),
            ctypes.c_int32(MAX_NEW_TOKENS - 2), ctypes.c_int32(prompt_len + 2),
            out_tokens, ctypes.byref(out_count),
        )
        server_dt = time.time() - t0
        server_toks = out_count.value
        server_tps = server_toks / server_dt if server_dt > 0 else 0
        server_text = tokenizer.decode([out_tokens[i] for i in range(server_toks)], skip_special_tokens=True)
        print(f"[SERVER] {server_toks} tokens in {server_dt:.1f}s = {server_tps:.1f} tok/s")
        print(f"[SERVER] {server_text[:200]}")
except Exception as e:
    print(f"[SERVER] Graph decode not available: {e}")

# ============================================================
# SUMMARY
# ============================================================
speedup = cached_tps / baseline_tps if baseline_tps > 0 else 0

print(f"\n{'='*60}")
print(f"  Inference Results — {dev}")
print(f"{'='*60}")
print(f"  Model:        {MODEL}")
print(f"  GPU load:     {load_gpu_dt:.1f}s ({mem_mb:.0f} MB)")
print(f"  Baseline:     {baseline_tps:.1f} tok/s (dynamic KV, {baseline_toks} tokens)")
print(f"  StaticCache:  {cached_tps:.1f} tok/s ({cached_toks} tokens)")
if server_tps > 0:
    print(f"  ServerDecode: {server_tps:.1f} tok/s ({server_toks} tokens)")
print(f"{'='*60}")

if cached_toks >= 100 and math.isfinite(cached_tps) and cached_tps > 0:
    print("PASS: test_inference_cached")
    sys.exit(0)
else:
    print("FAIL: test_inference_cached")
    sys.exit(1)
