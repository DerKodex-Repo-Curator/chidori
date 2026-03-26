#!/usr/bin/env python3
"""Llama 3.1 8B FP16 — real inference on remote GPU via server-side decode."""
import os, sys, torch, time, ctypes
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
torch._C._cuda_init()

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="eager").cuda()
model.eval(); torch.cuda.synchronize()
print(f"Loaded: {time.time()-t0:.0f}s, {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
lib = ctypes.CDLL("build/libcuda.so.1")

def ask(question, max_tok=120):
    msgs = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").input_ids.cuda()
    plen = inp.shape[1]
    sc = StaticCache(config=model.config, max_cache_len=plen + max_tok + 16)
    with torch.no_grad():
        pf = model(inp, past_key_values=sc, use_cache=True)
    nt = pf.logits[:, -1:, :].argmax(-1)
    cp = torch.tensor([plen], device="cuda")
    with torch.no_grad():
        model(nt, past_key_values=sc, use_cache=True, cache_position=cp)
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    si = nt.clone(); sp = torch.tensor([plen + 1], device="cuda", dtype=torch.long)
    with torch.cuda.stream(s):
        with torch.no_grad():
            so = model(si, past_key_values=sc, use_cache=True, cache_position=sp)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    sptr = ctypes.c_void_p(s.cuda_stream)
    lib.cuStreamBeginCapture_v2(sptr, ctypes.c_int(0))
    with torch.cuda.stream(s):
        with torch.no_grad():
            so = model(si, past_key_values=sc, use_cache=True, cache_position=sp)
    gptr = ctypes.c_void_p(0); lib.cuStreamEndCapture(sptr, ctypes.byref(gptr))
    eptr = ctypes.c_void_p(0); lib.cuGraphInstantiate(ctypes.byref(eptr), gptr, ctypes.c_ulonglong(0))
    out_tokens = (ctypes.c_int32 * max_tok)()
    out_count = ctypes.c_int32(0)
    torch.cuda.synchronize()
    t0 = time.time()
    lib.chidori_graph_decode(
        ctypes.c_uint64(eptr.value), ctypes.c_uint64(sptr.value),
        ctypes.c_uint64(si.data_ptr()), ctypes.c_uint64(sp.data_ptr()),
        ctypes.c_uint64(so.logits.data_ptr()),
        ctypes.c_int32(model.config.vocab_size), ctypes.c_int32(max_tok - 2),
        ctypes.c_int32(plen + 2), out_tokens, ctypes.byref(out_count),
    )
    dt = time.time() - t0
    n = out_count.value
    tps = n / dt if dt > 0 else 0
    return tok.decode([out_tokens[j] for j in range(n)], skip_special_tokens=True), n, tps

questions = [
    "Explain quantum computing to a 10 year old",
    "Write a Python two_sum function with O(n) time complexity",
    "What are the pros and cons of microservices vs monolith architecture?",
]

for i, q in enumerate(questions):
    resp, n, tps = ask(q)
    sep = "=" * 60
    print(f"\n{sep}", flush=True)
    print(f"Q{i+1}: {q}", flush=True)
    print(f"({n} tok, {tps:.0f} tok/s)", flush=True)
    print(f"A: {resp[:500]}", flush=True)

print("\nDONE", flush=True)
