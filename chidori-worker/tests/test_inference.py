#!/usr/bin/env python3
"""Chidori — 1B parameter inference test"""
import torch
import time

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Explain how GPU-over-IP works in simple terms:"

# ============================================================
# LOAD PHASE — download model + upload to GPU
# ============================================================
print(f"\n[LOAD] Loading {MODEL}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="eager")
load_cpu_dt = time.time() - t0
print(f"[LOAD] CPU load: {load_cpu_dt:.1f}s")

print("[LOAD] Moving to GPU...")
t0 = time.time()
model = model.cuda()
torch.cuda.synchronize()
load_gpu_dt = time.time() - t0
print(f"[LOAD] GPU transfer: {load_gpu_dt:.1f}s")

mem_mb = torch.cuda.memory_allocated() / 1e6
print(f"[LOAD] GPU memory used: {mem_mb:.0f} MB")

# ============================================================
# INFERENCE PHASE — generate tokens
# ============================================================
inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
print(f"\n[INFERENCE] Prompt: {PROMPT}")

# Quick embedding sanity check
print("\n[EMBED] Testing embedding forward pass...")
with torch.no_grad():
    hidden = model.model.embed_tokens(inputs["input_ids"])
    print(f"[EMBED] mean={hidden.float().mean().item():.6f} std={hidden.float().std().item():.6f}")
    if hidden.float().std().item() < 0.001:
        print("[EMBED] FAIL — embedding output is near-zero, aborting")
        exit(1)
    print("[EMBED] OK — embedding produces non-trivial output")

print(f"\n[INFERENCE] Generating 200 tokens...")

t0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
    )
torch.cuda.synchronize()
gen_dt = time.time() - t0

new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
tokens_per_sec = new_tokens / gen_dt

# Debug: inspect raw output tokens
out_cpu = outputs[0].cpu()
print(f"[DEBUG] Output shape: {outputs.shape}, dtype: {outputs.dtype}")
print(f"[DEBUG] First 20 tokens: {out_cpu[:20].tolist()}")
print(f"[DEBUG] Last 20 tokens: {out_cpu[-20:].tolist()}")
print(f"[DEBUG] Min: {out_cpu.min().item()}, Max: {out_cpu.max().item()}")

text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

print(f"[INFERENCE] Done: {gen_dt:.2f}s — {new_tokens} tokens — {tokens_per_sec:.1f} tok/s")
print(f"\n--- Output ---")
print(text[:500])
print(f"--- End ---")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"MODEL:     {MODEL}")
print(f"CPU LOAD:  {load_cpu_dt:.1f}s")
print(f"GPU LOAD:  {load_gpu_dt:.1f}s ({mem_mb:.0f} MB)")
print(f"INFERENCE: {gen_dt:.2f}s ({new_tokens} tokens, {tokens_per_sec:.1f} tok/s)")
print(f"{'='*60}")
