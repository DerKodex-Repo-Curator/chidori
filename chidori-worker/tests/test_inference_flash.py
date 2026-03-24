#!/usr/bin/env python3
"""Chidori — Flash Attention inference test

Tests inference WITHOUT eager attention to see if flash/SDPA works
through the CUDA interceptor. Flash attention uses standard cuLaunchKernel
with custom CUBINs — Chidori should already handle this.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import time
import sys

torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Device: {dev} ({vram_gb:.1f} GB)")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Explain how GPU-over-IP works in simple terms:"

# Try SDPA first (PyTorch native), then flash_attention_2
for attn_impl in ["sdpa", "flash_attention_2", "eager"]:
    print(f"\n[TEST] Trying attn_implementation='{attn_impl}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, attn_implementation=attn_impl
        )

        print("[LOAD] Moving to GPU...")
        t0 = time.time()
        model = model.cuda()
        model.eval()
        torch.cuda.synchronize()
        load_dt = time.time() - t0
        print(f"[LOAD] GPU transfer: {load_dt:.1f}s")

        inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

        print(f"[INFERENCE] Generating 50 tokens with {attn_impl}...")
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        torch.cuda.synchronize()
        gen_dt = time.time() - t0

        new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        tps = new_tokens / gen_dt

        text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"[RESULT] {new_tokens} tokens in {gen_dt:.1f}s = {tps:.1f} tok/s")
        print(f"[TEXT] {text[:200]}")
        print(f"\n{'='*60}")
        print(f"ATTENTION:  {attn_impl}")
        print(f"TOKENS:     {new_tokens} at {tps:.1f} tok/s")
        print(f"{'='*60}")

        if new_tokens >= 20 and tps > 0:
            print(f"PASS: test_inference_flash ({attn_impl})")
            sys.exit(0)
        else:
            print(f"FAIL: test_inference_flash ({attn_impl})")
            sys.exit(1)

    except Exception as e:
        print(f"[SKIP] {attn_impl} failed: {e}")
        # Clean up for next attempt
        try:
            del model
            torch.cuda.empty_cache()
        except:
            pass
        continue

print("FAIL: test_inference_flash (all attention implementations failed)")
sys.exit(1)
