#!/usr/bin/env python3
"""Partial GPU offloading — HuggingFace Accelerate device_map.

Split TinyLlama across local GPU (device 0) and remote GPU (device 1)
using Accelerate's automatic device placement.
"""
import sys
import torch

torch._C._cuda_init()
assert torch.cuda.device_count() >= 2, "Need 2 devices (local + remote)"

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The meaning of life is"

print(f"Loading {MODEL} with device_map='auto' across 2 GPUs...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager",
)

devices_used = set(str(v) for v in model.hf_device_map.values())
print(f"Devices used: {devices_used}")
assert len(devices_used) >= 2, f"Model only on {devices_used}, expected 2 devices"

inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
print(f"Generated {new_tokens} tokens")
print(f"Output: {text[:200]}")
assert new_tokens >= 20, f"Only generated {new_tokens} tokens"

print("PASS: test_partial_offload_device_map")
sys.exit(0)
