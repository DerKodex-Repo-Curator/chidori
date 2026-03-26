#!/usr/bin/env python3
"""Chidori — Split GPU inference test

Splits TinyLlama across local GPU (4090) + remote GPU (A100).
Uses accelerate dispatch_model for correct cross-device forwarding.
Measures tok/s for split inference.

Requires CHIDORI_LOCAL_DEVICES=0 to enable partial offload.
"""
import os, torch, time, sys
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model

torch._C._cuda_init()
n_devs = torch.cuda.device_count()
if n_devs < 2:
    print(f"SKIP: need 2 GPUs, found {n_devs}")
    sys.exit(0)

dev0 = torch.cuda.get_device_name(0)
dev1 = torch.cuda.get_device_name(1)
print(f"Device 0 (local): {dev0}")
print(f"Device 1 (remote): {dev1}")
print("=" * 60)

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Explain how GPU-over-IP works:"
MAX_TOKENS = 20

# Load to CPU first (avoids device_map from_pretrained crash)
print(f"\n[LOAD] Loading {MODEL} to CPU...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="eager"
)
cpu_dt = time.time() - t0
print(f"[LOAD] CPU: {cpu_dt:.1f}s")

# Split: first half on device 0, second half on device 1
layers = model.model.layers
N = len(layers)
HALF = N // 2

device_map = {"model.embed_tokens": 0}
for i in range(HALF):
    device_map[f"model.layers.{i}"] = 0
for i in range(HALF, N):
    device_map[f"model.layers.{i}"] = 1
device_map["model.norm"] = 1
device_map["lm_head"] = 1

print(f"[SPLIT] Dispatching: layers 0-{HALF-1} → dev0, {HALF}-{N-1} → dev1")
t0 = time.time()
model = dispatch_model(model, device_map=device_map)
model.eval()
split_dt = time.time() - t0
print(f"[SPLIT] Dispatch: {split_dt:.1f}s")

# Generate
tok = AutoTokenizer.from_pretrained(MODEL)
inp = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
plen = inp.shape[1]

print(f"\n[INFER] Generating {MAX_TOKENS} tokens (split across 2 GPUs)...")
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
t0 = time.time()
with torch.no_grad():
    out = model.generate(inp, max_new_tokens=MAX_TOKENS, do_sample=False)
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
dt = time.time() - t0
n = out.shape[1] - plen
tps = n / dt

text = tok.decode(out[0], skip_special_tokens=True)
print(f"[INFER] {n} tokens in {dt:.1f}s = {tps:.2f} tok/s")
print(f"\n--- Output ---")
print(text[:300])
print(f"--- End ---")

print(f"\n{'='*60}")
print(f"  Split Inference — {dev0} + {dev1}")
print(f"{'='*60}")
print(f"  Layers:   0-{HALF-1} on {dev0[:20]}, {HALF}-{N-1} on {dev1[:20]}")
print(f"  Speed:    {tps:.2f} tok/s")
print(f"{'='*60}")

if n >= 10 and tps > 0:
    print("PASS: test_split_inference")
    sys.exit(0)
else:
    print("FAIL: test_split_inference")
    sys.exit(1)
