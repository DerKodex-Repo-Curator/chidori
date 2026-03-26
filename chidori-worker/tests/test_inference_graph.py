#!/usr/bin/env python3
"""Chidori — CUDA Graph inference test

Captures the decode step into a CUDA graph, then replays it per token.
Each token = 1 graph launch RPC instead of ~100 individual CUDA calls.

Requires StaticCache (fixed tensor shapes for graph capture).
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
MAX_NEW_TOKENS = 100

# ============================================================
# LOAD
# ============================================================
print(f"\n[LOAD] Loading {MODEL}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="eager"
)
load_cpu_dt = time.time() - t0

print("[LOAD] Moving to GPU...")
t0 = time.time()
model = model.cuda()
model.eval()
torch.cuda.synchronize()
load_gpu_dt = time.time() - t0
mem_mb = torch.cuda.memory_allocated() / 1e6
print(f"[LOAD] GPU: {load_gpu_dt:.1f}s ({mem_mb:.0f} MB)")

# ============================================================
# BASELINE: no graph, no static cache
# ============================================================
inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
prompt_len = inputs["input_ids"].shape[1]

print(f"\n[BASELINE] Generating {MAX_NEW_TOKENS} tokens (no graph)...")
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    baseline_out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
torch.cuda.synchronize()
baseline_dt = time.time() - t0
baseline_toks = baseline_out.shape[1] - prompt_len
baseline_tps = baseline_toks / baseline_dt
print(f"[BASELINE] {baseline_toks} tokens in {baseline_dt:.1f}s = {baseline_tps:.1f} tok/s")

# ============================================================
# CUDA GRAPH: capture decode step, replay per token
# ============================================================
print(f"\n[GRAPH] Setting up StaticCache + CUDA Graph capture...")

# We need to do manual token-by-token generation with graph capture
static_cache = StaticCache(
    config=model.config,
    max_cache_len=prompt_len + MAX_NEW_TOKENS + 16,
)

# Prefill: run the prompt through the model to fill KV cache
with torch.no_grad():
    prefill_ids = inputs["input_ids"]
    prefill_out = model(
        prefill_ids,
        past_key_values=static_cache,
        use_cache=True,
    )
    next_token = prefill_out.logits[:, -1:, :].argmax(dim=-1)

print(f"[GRAPH] Prefill done. Attempting graph capture...")
generated = [next_token.item()]

# Warmup one decode step (required before capture)
with torch.no_grad():
    cache_position = torch.tensor([prompt_len], device="cuda")
    warmup_out = model(
        next_token,
        past_key_values=static_cache,
        use_cache=True,
        cache_position=cache_position,
    )
    warmup_next = warmup_out.logits[:, -1:, :].argmax(dim=-1)

generated.append(warmup_next.item())
torch.cuda.synchronize()

# Try CUDA Graph capture using direct driver API (bypasses PyTorch's memory pool setup)
try:
    import ctypes
    lib = ctypes.CDLL("build/libcuda.so.1")

    # Create a CUDA stream for capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    # Static input/output buffers for graph replay
    static_input = next_token.clone()
    static_position = torch.tensor([prompt_len + 1], device="cuda")

    with torch.cuda.stream(s):
        # Warmup in capture stream
        with torch.no_grad():
            out = model(
                static_input,
                past_key_values=static_cache,
                use_cache=True,
                cache_position=static_position,
            )
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Capture using driver API directly (not torch.cuda.graph which needs memory pools)
    stream_ptr = ctypes.c_void_p(s.cuda_stream)
    r = lib.cuStreamBeginCapture_v2(stream_ptr, ctypes.c_int(0))
    if r != 0:
        raise RuntimeError(f"cuStreamBeginCapture failed: {r}")

    with torch.cuda.stream(s):
        with torch.no_grad():
            static_output = model(
                static_input,
                past_key_values=static_cache,
                use_cache=True,
                cache_position=static_position,
            )

    graph_ptr = ctypes.c_void_p(0)
    r = lib.cuStreamEndCapture(stream_ptr, ctypes.byref(graph_ptr))
    if r != 0 or not graph_ptr.value:
        raise RuntimeError(f"cuStreamEndCapture failed: r={r} graph={graph_ptr.value}")

    exec_ptr = ctypes.c_void_p(0)
    r = lib.cuGraphInstantiate(ctypes.byref(exec_ptr), graph_ptr, ctypes.c_ulonglong(0))
    if r != 0 or not exec_ptr.value:
        raise RuntimeError(f"cuGraphInstantiate failed: r={r}")

    print(f"[GRAPH] Capture successful!")

    # Generate tokens using graph replay via driver API
    # All ops must be on the SAME stream as the graph to ensure ordering.
    # The graph was captured on stream s, so we run everything there.
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(MAX_NEW_TOKENS - 2):  # -2 for prefill + warmup tokens
        # Update input for this step (on capture stream for ordering)
        with torch.cuda.stream(s):
            static_input.copy_(warmup_next)
            static_position.fill_(prompt_len + 2 + i)

        # Replay the graph — ONE RPC instead of ~100
        lib.cuGraphLaunch(exec_ptr, stream_ptr)

        # Wait for graph to finish before reading output
        torch.cuda.current_stream().wait_stream(s)

        # Get next token from static output
        warmup_next = static_output.logits[:, -1:, :].argmax(dim=-1)
        generated.append(warmup_next.item())

    torch.cuda.synchronize()
    graph_dt = time.time() - t0
    graph_toks = len(generated)
    graph_tps = graph_toks / graph_dt

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"[GRAPH] {graph_toks} tokens in {graph_dt:.1f}s = {graph_tps:.1f} tok/s")
    print(f"\n--- Output ---")
    print(text[:500])
    print(f"--- End ---")

    speedup = graph_tps / baseline_tps if baseline_tps > 0 else 0

    # ============================================================
    # SERVER-SIDE DECODE: Run graph+argmax loop on server, return all tokens
    # One RPC for N tokens instead of N RPCs
    # ============================================================
    server_tps = 0
    server_toks = 0
    try:
        print(f"\n[SERVER] Testing server-side autoregressive decode...")
        # Reset static cache for fresh generation
        static_cache2 = StaticCache(
            config=model.config,
            max_cache_len=prompt_len + MAX_NEW_TOKENS + 16,
        )
        with torch.no_grad():
            pfout2 = model(
                inputs["input_ids"],
                past_key_values=static_cache2,
                use_cache=True,
            )
            nt2 = pfout2.logits[:, -1:, :].argmax(dim=-1)

        # Warmup decode step
        cp2 = torch.tensor([prompt_len], device="cuda")
        with torch.no_grad():
            wo2 = model(nt2, past_key_values=static_cache2, use_cache=True, cache_position=cp2)
        wn2 = wo2.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()

        # Re-capture graph with fresh cache
        s2 = torch.cuda.Stream()
        s2.wait_stream(torch.cuda.current_stream())
        si2 = nt2.clone()
        sp2 = torch.tensor([prompt_len + 1], device="cuda", dtype=torch.long)
        with torch.cuda.stream(s2):
            with torch.no_grad():
                so2 = model(si2, past_key_values=static_cache2, use_cache=True, cache_position=sp2)
        torch.cuda.current_stream().wait_stream(s2)
        torch.cuda.synchronize()

        sptr2 = ctypes.c_void_p(s2.cuda_stream)
        r = lib.cuStreamBeginCapture_v2(sptr2, ctypes.c_int(0))
        with torch.cuda.stream(s2):
            with torch.no_grad():
                so2 = model(si2, past_key_values=static_cache2, use_cache=True, cache_position=sp2)
        gptr2 = ctypes.c_void_p(0)
        lib.cuStreamEndCapture(sptr2, ctypes.byref(gptr2))
        eptr2 = ctypes.c_void_p(0)
        lib.cuGraphInstantiate(ctypes.byref(eptr2), gptr2, ctypes.c_ulonglong(0))

        # Get GPU addresses for server-side decode
        input_data_ptr = si2.data_ptr()
        position_data_ptr = sp2.data_ptr()
        logits_data_ptr = so2.logits.data_ptr()
        vocab = model.config.vocab_size

        # Build GRAPH_DECODE payload
        import struct
        decode_payload = struct.pack('<QQQQQiii',
            eptr2.value,           # graph_exec handle
            sptr2.value,           # stream handle
            input_data_ptr,        # input_ptr (GPU)
            position_data_ptr,     # position_ptr (GPU)
            logits_data_ptr,       # logits_ptr (GPU)
            vocab,                 # vocab_size
            MAX_NEW_TOKENS - 2,    # num_tokens
            prompt_len + 2,        # start_position
        )

        # Call chidori_graph_decode — exposed public function in libchidori.so
        out_tokens = (ctypes.c_int32 * (MAX_NEW_TOKENS))()
        out_count = ctypes.c_int32(0)

        torch.cuda.synchronize()
        t0 = time.time()
        ret = lib.chidori_graph_decode(
            ctypes.c_uint64(eptr2.value),
            ctypes.c_uint64(sptr2.value),
            ctypes.c_uint64(input_data_ptr),
            ctypes.c_uint64(position_data_ptr),
            ctypes.c_uint64(logits_data_ptr),
            ctypes.c_int32(vocab),
            ctypes.c_int32(MAX_NEW_TOKENS - 2),
            ctypes.c_int32(prompt_len + 2),
            out_tokens,
            ctypes.byref(out_count),
        )
        server_dt = time.time() - t0
        server_toks = out_count.value
        server_tps = server_toks / server_dt if server_dt > 0 else 0

        server_text = tokenizer.decode([out_tokens[i] for i in range(server_toks)], skip_special_tokens=True)
        print(f"[SERVER] {server_toks} tokens in {server_dt:.1f}s = {server_tps:.1f} tok/s (ret={ret})")
        print(f"[SERVER] Text: {server_text[:200]}")

    except Exception as e:
        print(f"[SERVER] Server decode setup: {e}")

    print(f"\n{'='*60}")
    print(f"  CUDA Graph Inference — {dev}")
    print(f"{'='*60}")
    print(f"  Baseline:   {baseline_tps:.1f} tok/s (no graph)")
    print(f"  Graph:      {graph_tps:.1f} tok/s")
    if server_tps > 0:
        print(f"  Server:     {server_tps:.1f} tok/s (server-side decode)")
    print(f"  Speedup:    {speedup:.1f}x")
    print(f"{'='*60}")

    if graph_toks >= 50 and math.isfinite(graph_tps) and graph_tps > baseline_tps:
        print("PASS: test_inference_graph")
        sys.exit(0)
    else:
        print("FAIL: test_inference_graph (no speedup)")
        sys.exit(1)

except Exception as e:
    print(f"[GRAPH] Capture FAILED: {e}")
    print(f"[GRAPH] This is expected if CUDA Graph API is not fully supported yet.")
    print(f"\nBaseline result: {baseline_tps:.1f} tok/s")
    print("FAIL: test_inference_graph (capture failed)")
    sys.exit(1)
