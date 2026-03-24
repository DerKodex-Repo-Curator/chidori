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

# Try CUDA Graph capture
try:
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

    # Now capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        with torch.no_grad():
            static_output = model(
                static_input,
                past_key_values=static_cache,
                use_cache=True,
                cache_position=static_position,
            )

    print(f"[GRAPH] Capture successful!")

    # Generate tokens using graph replay
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(MAX_NEW_TOKENS - 2):  # -2 for prefill + warmup tokens
        # Update input for this step
        static_input.copy_(warmup_next)
        static_position.fill_(prompt_len + 2 + i)

        # Replay the graph — ONE RPC instead of ~100
        graph.replay()

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
    print(f"\n{'='*60}")
    print(f"  CUDA Graph Inference — {dev}")
    print(f"{'='*60}")
    print(f"  Baseline:   {baseline_tps:.1f} tok/s (no graph)")
    print(f"  Graph:      {graph_tps:.1f} tok/s")
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
