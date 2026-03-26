#!/usr/bin/env python3
"""Chidori — Optimized split GPU inference

Direct layer calls + CUDA Graph on remote half.
Per token: local layers (4090) → transfer activations → remote graph (A100, 1 RPC).
"""
import os, sys, torch, time, ctypes
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

from transformers import AutoModelForCausalLM, AutoTokenizer
torch._C._cuda_init()

if torch.cuda.device_count() < 2:
    print("SKIP: need 2 GPUs"); sys.exit(0)

print(f"Dev0: {torch.cuda.get_device_name(0)}")
print(f"Dev1: {torch.cuda.get_device_name(1)}")

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_TOKENS = 30

# --- Load and split ---
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="eager")
layers = model.model.layers
N = len(layers); HALF = N // 2
HIDDEN = model.config.hidden_size
VOCAB = model.config.vocab_size

model.model.embed_tokens = model.model.embed_tokens.to("cuda:0")
rotary_local = model.model.rotary_emb.to("cuda:0")
for i in range(HALF): layers[i] = layers[i].to("cuda:0")

import copy
rotary_remote = copy.deepcopy(model.model.rotary_emb).to("cuda:1")
for i in range(HALF, N): layers[i] = layers[i].to("cuda:1")
model.model.norm = model.model.norm.to("cuda:1")
model.lm_head = model.lm_head.to("cuda:1")
model.eval()
print(f"Split: 0-{HALF-1} local, {HALF}-{N-1} remote", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
lib = ctypes.CDLL("build/libcuda.so.1")

# --- Prefill (full sequence, no graph) ---
inp = tok("Explain GPU-over-IP:", return_tensors="pt").input_ids.to("cuda:0")
plen = inp.shape[1]
pos_local = torch.arange(plen, device="cuda:0").unsqueeze(0)

with torch.no_grad():
    h = model.model.embed_tokens(inp)
    cos, sin = rotary_local(h, pos_local)
    for i in range(HALF):
        h = layers[i](h, position_embeddings=(cos, sin))
        if isinstance(h, tuple): h = h[0]
    h = h.to("cuda:1")
    pos_remote = pos_local.to("cuda:1")
    cos_r, sin_r = rotary_remote(h, pos_remote)
    for i in range(HALF, N):
        h = layers[i](h, position_embeddings=(cos_r, sin_r))
        if isinstance(h, tuple): h = h[0]
    h = model.model.norm(h)
    logits = model.lm_head(h)
    first_token = logits[0, -1, :].argmax().item()

print(f"Prefill: {first_token} ({tok.decode([first_token])})", flush=True)

# --- Warmup single-token decode ---
dec_inp = torch.tensor([[first_token]], device="cuda:0", dtype=torch.long)
dec_pos = torch.tensor([[plen]], device="cuda:0", dtype=torch.long)

with torch.no_grad():
    h = model.model.embed_tokens(dec_inp)
    cos, sin = rotary_local(h, dec_pos)
    for i in range(HALF):
        h = layers[i](h, position_embeddings=(cos, sin))
        if isinstance(h, tuple): h = h[0]
    # Static buffers on A100 for graph capture
    static_h = h.to("cuda:1").contiguous()
    static_pos = dec_pos.to("cuda:1")
    cos_r, sin_r = rotary_remote(static_h, static_pos)
    for i in range(HALF, N):
        static_h = layers[i](static_h, position_embeddings=(cos_r, sin_r))
        if isinstance(static_h, tuple): static_h = static_h[0]
    static_h_out = model.model.norm(static_h)
    static_logits = model.lm_head(static_h_out)
    second_token = static_logits[0, -1, :].argmax().item()

print(f"Warmup: {second_token} ({tok.decode([second_token])})", flush=True)
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)

# --- Capture remote graph on A100 ---
# Re-create static input buffer for the remote half
static_activation = torch.zeros(1, 1, HIDDEN, dtype=torch.float16, device="cuda:1")
static_pos_remote = torch.tensor([[plen + 1]], device="cuda:1", dtype=torch.long)

# Warmup on capture stream
s1 = torch.cuda.Stream(device="cuda:1")
s1.wait_stream(torch.cuda.current_stream(1))

def remote_forward(h_in, pos_in):
    cos_r, sin_r = rotary_remote(h_in, pos_in)
    h = h_in
    for i in range(HALF, N):
        h = layers[i](h, position_embeddings=(cos_r, sin_r))
        if isinstance(h, tuple): h = h[0]
    h = model.model.norm(h)
    return model.lm_head(h)

with torch.cuda.stream(s1):
    with torch.no_grad():
        static_logits_out = remote_forward(static_activation, static_pos_remote)
torch.cuda.current_stream(1).wait_stream(s1)
torch.cuda.synchronize(1)
print("Remote warmup on capture stream OK", flush=True)

# Capture
sptr = ctypes.c_void_p(s1.cuda_stream)
r = lib.cuStreamBeginCapture_v2(sptr, ctypes.c_int(0))
print(f"BeginCapture: r={r}", flush=True)
if r != 0:
    print("FAIL: capture failed"); sys.exit(1)

with torch.cuda.stream(s1):
    with torch.no_grad():
        static_logits_out = remote_forward(static_activation, static_pos_remote)

gptr = ctypes.c_void_p(0)
lib.cuStreamEndCapture(sptr, ctypes.byref(gptr))
eptr = ctypes.c_void_p(0)
lib.cuGraphInstantiate(ctypes.byref(eptr), gptr, ctypes.c_ulonglong(0))

if not eptr.value:
    print("FAIL: graph instantiate failed"); sys.exit(1)
print(f"Remote graph captured! exec={eptr.value:#x}", flush=True)

# --- Generate with: local layers + remote graph via GRAPH_DECODE_SPLIT ---
generated = [first_token, second_token]
current_token = second_token
position = plen + 1

torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
print(f"\nGenerating {MAX_TOKENS} tokens with split graph...", flush=True)
t0 = time.time()

for step in range(MAX_TOKENS - 2):
    # 1. Local forward on 4090
    dec_inp.fill_(current_token)
    dec_pos.fill_(position)
    with torch.no_grad():
        h = model.model.embed_tokens(dec_inp)
        cos, sin = rotary_local(h, dec_pos)
        for i in range(HALF):
            h = layers[i](h, position_embeddings=(cos, sin))
            if isinstance(h, tuple): h = h[0]

    # 2. DtoH activations from 4090
    h_host = h.cpu().numpy().tobytes()

    # 3. Send activations + launch remote graph via GRAPH_DECODE_SPLIT
    out_token = ctypes.c_int32(0)
    act_buf = ctypes.create_string_buffer(h_host)
    ret = lib.chidori_graph_decode_split(
        ctypes.c_uint64(eptr.value),
        ctypes.c_uint64(sptr.value),
        ctypes.c_uint64(static_activation.data_ptr()),
        ctypes.c_uint64(static_pos_remote.data_ptr()),
        ctypes.c_uint64(static_logits_out[:, -1:, :].contiguous().data_ptr()),
        ctypes.c_int32(VOCAB),
        ctypes.c_int32(position),
        act_buf,
        ctypes.c_int32(len(h_host)),
        ctypes.byref(out_token),
    )

    current_token = out_token.value
    generated.append(current_token)
    position += 1
    if current_token == 2: break

torch.cuda.synchronize(0)
dt = time.time() - t0
tps = len(generated) / dt

text = tok.decode(generated, skip_special_tokens=True)
print(f"\nSPLIT GRAPH: {len(generated)} tok in {dt:.1f}s = {tps:.1f} tok/s")
print(f"TEXT: Explain GPU-over-IP:{text[:300]}")
print(f"\n{'='*60}")
print(f"  Baseline split: 0.72 tok/s (dispatch_model)")
print(f"  Graph split:    {tps:.1f} tok/s (local fwd + 1 RPC/token)")
print(f"{'='*60}")

if len(generated) >= 10 and tps > 0.5:
    print("PASS: test_split_graph")
else:
    print("FAIL: test_split_graph")
