#!/usr/bin/env python3
"""Chidori — Profile inference RPC pattern"""
import torch
import time
import os

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="eager")

print("[LOAD] Moving to GPU...")
model = model.cuda()
torch.cuda.synchronize()
print("[LOAD] Done")

# Single forward pass profiling
inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")

# Warmup
with torch.no_grad():
    out = model(inputs["input_ids"])
    logits = out.logits
    torch.cuda.synchronize()

# Timed single forward pass
t0 = time.time()
with torch.no_grad():
    out = model(inputs["input_ids"])
    logits = out.logits
    # This is what generate does per token - DtoH to get logits
    next_token = logits[0, -1].argmax().item()
t1 = time.time()
print(f"\n[PROFILE] Single forward + argmax: {(t1-t0)*1000:.1f}ms")
print(f"[PROFILE] Logits shape: {logits.shape}, next_token: {next_token}")

# Timed generation of just 5 tokens
t0 = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
t1 = time.time()
new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
print(f"\n[PROFILE] Generate 5 tokens: {(t1-t0)*1000:.1f}ms ({new_tokens} tokens, {(t1-t0)/new_tokens:.2f}s/tok)")

print("\nDONE")
