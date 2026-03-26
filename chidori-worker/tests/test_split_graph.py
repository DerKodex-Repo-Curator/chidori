#!/usr/bin/env python3
"""Chidori — Optimized split GPU inference

Split model across local 4090 + remote A100 via dispatch_model.
Baseline: 0.02 tok/s (individual RPCs for remote layers).
This test measures the baseline, then we can optimize from there.

Requires CHIDORI_LOCAL_DEVICES=0.
"""
import os, sys, torch, time
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model
torch._C._cuda_init()

if torch.cuda.device_count() < 2:
    print("SKIP: need 2 GPUs"); sys.exit(0)

dev0 = torch.cuda.get_device_name(0)
dev1 = torch.cuda.get_device_name(1)
print(f"Dev0 (local): {dev0}")
print(f"Dev1 (remote): {dev1}")

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load to CPU, then dispatch
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="eager")
N = len(model.model.layers)
HALF = N // 2

device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0}
for i in range(HALF): device_map[f"model.layers.{i}"] = 0
for i in range(HALF, N): device_map[f"model.layers.{i}"] = 1
device_map["model.norm"] = 1
device_map["lm_head"] = 1

model = dispatch_model(model, device_map=device_map)
model.eval()
print(f"Split: 0-{HALF-1} local, {HALF}-{N-1} remote")

tok = AutoTokenizer.from_pretrained(MODEL)
inp = tok("Explain GPU-over-IP:", return_tensors="pt").input_ids.to("cuda:0")
plen = inp.shape[1]

# Measure 10 tokens
print(f"Generating 10 tokens (split, {plen} prompt)...", flush=True)
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
t0 = time.time()
with torch.no_grad():
    out = model.generate(inp, max_new_tokens=10, do_sample=False)
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
dt = time.time() - t0
n = out.shape[1] - plen
tps = n / dt if dt > 0 else 0

text = tok.decode(out[0], skip_special_tokens=True)
print(f"\nRESULT: {n} tok in {dt:.1f}s = {tps:.3f} tok/s")
print(f"TEXT: {text[:200]}")
print(f"Per-token: {dt/n:.1f}s" if n > 0 else "")

if n >= 5:
    print("PASS: test_split_graph")
else:
    print("FAIL: test_split_graph")
