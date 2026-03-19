#!/usr/bin/env python3
"""Multi-server device_map test: use HuggingFace Accelerate to split a model.

Loads a small model with device_map splitting layers across cuda:0 and cuda:1.
Runs inference and verifies output.
"""
import sys
import torch

torch._C._cuda_init()
assert torch.cuda.device_count() >= 2, "Need 2+ devices"

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The meaning of life is"

print(f"Loading {MODEL} with device_map across 2 GPUs...")

# Split model: half layers on cuda:0, half on cuda:1
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",  # Accelerate distributes across available devices
    torch_dtype=torch.float16,
    attn_implementation="eager",
)

# Show where layers ended up
device_map = model.hf_device_map
devices_used = set(str(v) for v in device_map.values())
print(f"Devices used: {devices_used}")
assert len(devices_used) >= 2, f"Model only on {devices_used}, expected 2 devices"

# Run inference
inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Output: {text[:200]}")

new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
print(f"Generated {new_tokens} tokens")
assert new_tokens >= 20, f"Only generated {new_tokens} tokens"

print("PASS: test_multi_server_device_map")
sys.exit(0)
