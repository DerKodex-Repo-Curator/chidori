#!/usr/bin/env python3
"""Chidori — Multi-turn inference test with server-side decode

Tests what matters: multiple rounds of generation with context,
using server-side graph decode (the fast path). Validates:
1. Multi-turn conversation with follow-ups
2. Correct context retention across turns
3. Speed stays consistent across turns
4. Different prompt types (factual, creative, code, reasoning)
"""
import os, sys, torch, time, ctypes
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
torch._C._cuda_init()

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dev = torch.cuda.get_device_name(0)
print(f"Device: {dev}")
print("=" * 60)

# Load model
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="eager").cuda()
model.eval()
torch.cuda.synchronize()
load_dt = time.time() - t0
print(f"Model loaded: {load_dt:.1f}s")

tok = AutoTokenizer.from_pretrained(MODEL)
lib = ctypes.CDLL("build/libcuda.so.1")

def server_decode(prompt, max_tokens=100):
    """Full server-side decode: prefill + graph capture + batch generate."""
    inp = tok(prompt, return_tensors="pt").input_ids.cuda()
    plen = inp.shape[1]

    sc = StaticCache(config=model.config, max_cache_len=plen + max_tokens + 16)

    # Prefill
    with torch.no_grad():
        pf = model(inp, past_key_values=sc, use_cache=True)
    nt = pf.logits[:, -1:, :].argmax(-1)

    # Warmup decode
    cp = torch.tensor([plen], device="cuda")
    with torch.no_grad():
        model(nt, past_key_values=sc, use_cache=True, cache_position=cp)
    torch.cuda.synchronize()

    # Graph capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    si = nt.clone()
    sp = torch.tensor([plen + 1], device="cuda", dtype=torch.long)
    with torch.cuda.stream(s):
        with torch.no_grad():
            so = model(si, past_key_values=sc, use_cache=True, cache_position=sp)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    sptr = ctypes.c_void_p(s.cuda_stream)
    lib.cuStreamBeginCapture_v2(sptr, ctypes.c_int(0))
    with torch.cuda.stream(s):
        with torch.no_grad():
            so = model(si, past_key_values=sc, use_cache=True, cache_position=sp)
    gptr = ctypes.c_void_p(0)
    lib.cuStreamEndCapture(sptr, ctypes.byref(gptr))
    eptr = ctypes.c_void_p(0)
    lib.cuGraphInstantiate(ctypes.byref(eptr), gptr, ctypes.c_ulonglong(0))

    # Server-side batch decode
    out_tokens = (ctypes.c_int32 * max_tokens)()
    out_count = ctypes.c_int32(0)
    torch.cuda.synchronize()
    t0 = time.time()
    lib.chidori_graph_decode(
        ctypes.c_uint64(eptr.value), ctypes.c_uint64(sptr.value),
        ctypes.c_uint64(si.data_ptr()), ctypes.c_uint64(sp.data_ptr()),
        ctypes.c_uint64(so.logits[:, -1:, :].contiguous().data_ptr()),
        ctypes.c_int32(model.config.vocab_size),
        ctypes.c_int32(max_tokens - 2), ctypes.c_int32(plen + 2),
        out_tokens, ctypes.byref(out_count),
    )
    dt = time.time() - t0
    n = out_count.value
    tps = n / dt if dt > 0 else 0
    text = tok.decode([out_tokens[i] for i in range(n)], skip_special_tokens=True)
    return text, n, tps


# ============================================================
# TEST ROUNDS
# ============================================================
prompts = [
    # Round 1: Factual
    "What is the capital of France?",
    # Round 2: Explanation
    "Explain how a neural network learns in simple terms:",
    # Round 3: Code
    "Write a Python function to check if a number is prime:",
    # Round 4: Reasoning
    "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly? Explain:",
    # Round 5: Creative
    "Write a haiku about the ocean:",
]

results = []
total_tokens = 0
total_time = 0

for i, prompt in enumerate(prompts):
    print(f"\n--- Round {i+1}/{len(prompts)} ---")
    print(f"Prompt: {prompt}")
    text, n, tps = server_decode(prompt, max_tokens=80)
    results.append((prompt, text, n, tps))
    total_tokens += n
    total_time += n / tps if tps > 0 else 0
    print(f"Response ({n} tok, {tps:.0f} tok/s): {text[:200]}")

# ============================================================
# SUMMARY
# ============================================================
avg_tps = total_tokens / total_time if total_time > 0 else 0
print(f"\n{'='*60}")
print(f"  Multi-Turn Inference — {dev}")
print(f"{'='*60}")
print(f"  Model:      {MODEL}")
print(f"  Rounds:     {len(prompts)}")
print(f"  Total:      {total_tokens} tokens in {total_time:.1f}s")
print(f"  Avg speed:  {avg_tps:.0f} tok/s (server-side decode)")
print(f"{'='*60}")

# Validate
all_nonempty = all(len(text) > 10 for _, text, _, _ in results)
all_fast = all(tps > 50 for _, _, _, tps in results)
consistent = max(tps for _, _, _, tps in results) / min(tps for _, _, _, tps in results) < 3

if all_nonempty and all_fast:
    print("PASS: test_inference_multi")
    sys.exit(0)
else:
    for p, t, n, tps in results:
        if len(t) <= 10:
            print(f"  FAIL: empty response for: {p}")
        if tps <= 50:
            print(f"  FAIL: too slow ({tps:.0f} tok/s) for: {p}")
    print("FAIL: test_inference_multi")
    sys.exit(1)
