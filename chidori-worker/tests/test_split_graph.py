#!/usr/bin/env python3
"""Chidori — Split GPU inference with StaticCache

Uses dispatch_model for correct cross-device forwarding with KV cache.
StaticCache eliminates per-token memory allocation RPCs.
"""
import os, sys, torch, time
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
from accelerate import dispatch_model
torch._C._cuda_init()

if torch.cuda.device_count() < 2:
    print("SKIP: need 2 GPUs"); sys.exit(0)

print(f"Dev0: {torch.cuda.get_device_name(0)}")
print(f"Dev1: {torch.cuda.get_device_name(1)}")

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_TOKENS = 30

# Load to CPU, then dispatch
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="eager")
N = len(model.model.layers); HALF = N // 2

dm = {"model.embed_tokens": 0, "model.rotary_emb": 0}
for i in range(HALF): dm[f"model.layers.{i}"] = 0
for i in range(HALF, N): dm[f"model.layers.{i}"] = 1
dm["model.norm"] = 1; dm["lm_head"] = 1

model = dispatch_model(model, device_map=dm)
model.eval()
print(f"Split: 0-{HALF-1} local, {HALF}-{N-1} remote", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
inp = tok("Explain GPU-over-IP:", return_tensors="pt").input_ids.to("cuda:0")
plen = inp.shape[1]

# --- Test 1: Baseline with StaticCache ---
sc = StaticCache(config=model.config, max_cache_len=plen + MAX_TOKENS + 10)
print(f"\n[CACHED] Generating {MAX_TOKENS} tokens with StaticCache...", flush=True)
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
t0 = time.time()
with torch.no_grad():
    out = model.generate(inp, max_new_tokens=MAX_TOKENS, do_sample=False,
                         past_key_values=sc, cache_implementation="static")
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
dt = time.time() - t0
n = out.shape[1] - plen
tps = n / dt if dt > 0 else 0
text = tok.decode(out[0], skip_special_tokens=True)
print(f"[CACHED] {n} tok in {dt:.1f}s = {tps:.2f} tok/s", flush=True)
print(f"TEXT: {text[:200]}", flush=True)

print(f"\n{'='*60}")
print(f"  Split Inference — StaticCache")
print(f"{'='*60}")
print(f"  Speed: {tps:.2f} tok/s")
print(f"{'='*60}")

if n >= 10 and tps > 0:
    print("PASS: test_split_graph")
else:
    print("FAIL: test_split_graph")
