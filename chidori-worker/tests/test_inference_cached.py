#!/usr/bin/env python3
"""Chidori — Cached inference test with StaticCache

Uses HuggingFace StaticCache to pre-allocate KV cache at max_seq_len.
All matmul dimensions become fixed per token, enabling:
  1. Heuristic cache hits (same dimensions = same cache key)
  2. Server-side replay (fingerprint matches every token)

Expected: faster than dynamic KV cache inference over network.
"""
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
print(f"\n--- Output ---")
print(text[:500])
print(f"--- End ---")

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
print(f"  Speedup:      {speedup:.1f}x")
print(f"{'='*60}")

if cached_toks >= 100 and math.isfinite(cached_tps) and cached_tps > 0:
    print("PASS: test_inference_cached")
    sys.exit(0)
else:
    print("FAIL: test_inference_cached")
    sys.exit(1)
