#!/usr/bin/env python3
"""Chidori — 1B parameter inference test"""
import torch
import time

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Explain how GPU-over-IP works in simple terms:"

# ============================================================
# LOAD PHASE — download model + upload to GPU
# ============================================================
print(f"\n[LOAD] Loading {MODEL}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="eager")
load_cpu_dt = time.time() - t0
print(f"[LOAD] CPU load: {load_cpu_dt:.1f}s")

print("[LOAD] Moving to GPU...")
t0 = time.time()
model = model.cuda()
torch.cuda.synchronize()
load_gpu_dt = time.time() - t0
print(f"[LOAD] GPU transfer: {load_gpu_dt:.1f}s")

mem_mb = torch.cuda.memory_allocated() / 1e6
print(f"[LOAD] GPU memory used: {mem_mb:.0f} MB")

# ============================================================
# INFERENCE PHASE — generate tokens
# ============================================================
# Quick sanity check: roundtrip int64 tensor through GPU
print("\n[SANITY] Testing int64 GPU roundtrip...")
test_data = torch.tensor([1, 2, 3, 42, 100, 32000], dtype=torch.int64)
test_gpu = test_data.cuda()
torch.cuda.synchronize()
test_back = test_gpu.cpu()
if torch.equal(test_data, test_back):
    print("[SANITY] OK — int64 roundtrip matches")
else:
    print(f"[SANITY] FAIL — sent: {test_data.tolist()}, got: {test_back.tolist()}")

# Test basic GPU compute operations
print("\n[SANITY] Testing matmul...")
a = torch.randn(4, 4, dtype=torch.float16).cuda()
b = torch.randn(4, 4, dtype=torch.float16).cuda()
c = torch.mm(a, b).cpu()
print(f"[SANITY] matmul nonzero: {(c != 0).sum().item()}/16")

print("\n[SANITY] Testing tensor indexing (embedding-like)...")
w = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float16).cuda()
idx = torch.tensor([2, 0, 1], dtype=torch.int64).cuda()
result = w[idx].cpu()
print(f"[SANITY] w[idx] expected: [[7,8,9],[1,2,3],[4,5,6]] got: {result.tolist()}")

print("\n[SANITY] Testing index_select (same as indexing but explicit)...")
idx1d = torch.tensor([2, 0, 1], dtype=torch.int64).cuda()
is_out = torch.index_select(w, 0, idx1d).cpu()
print(f"[SANITY] index_select: {is_out.tolist()}")

print("\n[SANITY] Testing torch.embedding directly...")
emb_w = torch.randn(5, 4, dtype=torch.float16).cuda()
emb_idx = torch.tensor([[0, 2, 4]], dtype=torch.int64).cuda()
torch.cuda.synchronize()
emb_out = torch.nn.functional.embedding(emb_idx, emb_w)
torch.cuda.synchronize()
err = torch.cuda.get_last_error() if hasattr(torch.cuda, 'get_last_error') else None
print(f"[SANITY] cudaGetLastError after embedding: {err}")
emb_out_cpu = emb_out.cpu()
print(f"[SANITY] embedding nonzero: {(emb_out_cpu != 0).sum().item()}/{emb_out_cpu.numel()}")
print(f"[SANITY] embedding output: {emb_out_cpu[0].tolist()}")

# Compare: manually index the same data
manual = emb_w[emb_idx[0]].cpu()
print(f"[SANITY] manual w[idx] nonzero: {(manual != 0).sum().item()}/{manual.numel()}")
print(f"[SANITY] manual w[idx] output: {manual.tolist()}")

# Test torch.gather as another path
print("\n[SANITY] Testing gather...")
gather_idx = emb_idx[0].unsqueeze(1).expand(3, 4)
gather_out = torch.gather(emb_w, 0, gather_idx).cpu()
print(f"[SANITY] gather nonzero: {(gather_out != 0).sum().item()}/{gather_out.numel()}")
print(f"[SANITY] gather output: {gather_out.tolist()}")

inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
print(f"\n[INFERENCE] Prompt: {PROMPT}")

# Verify input tokens survived HtoD transfer
input_cpu = inputs["input_ids"][0].cpu()
print(f"[DEBUG] Input tokens on GPU->CPU: {input_cpu.tolist()}")

# Check model weights survived GPU transfer
print("\n[WEIGHTS] Checking model weights on GPU...")
emb_w = model.model.embed_tokens.weight
emb_cpu = emb_w.cpu()
print(f"[WEIGHTS] embed_tokens shape={emb_w.shape} dtype={emb_w.dtype}")
print(f"[WEIGHTS] embed_tokens[:3,:5]: {emb_cpu[:3,:5].tolist()}")
print(f"[WEIGHTS] embed_tokens nonzero: {(emb_cpu != 0).sum().item()} / {emb_cpu.numel()}")
print(f"[WEIGHTS] embed_tokens mean={emb_cpu.float().mean().item():.6f} std={emb_cpu.float().std().item():.6f}")

# Check a deeper layer
ln_w = model.model.layers[0].input_layernorm.weight
ln_cpu = ln_w.cpu()
print(f"[WEIGHTS] layer0.input_layernorm shape={ln_w.shape}")
print(f"[WEIGHTS] layer0.input_layernorm[:10]: {ln_cpu[:10].tolist()}")

# Trace through model layers to find where output goes to zero
print("\n[TRACE] Layer-by-layer forward pass...")
with torch.no_grad():
    # 1. Embedding
    hidden = model.model.embed_tokens(inputs["input_ids"])
    h_cpu = hidden[0, 0, :10].cpu()
    print(f"[TRACE] embed_tokens: mean={hidden.float().mean().item():.6f} std={hidden.float().std().item():.6f} first10={h_cpu.tolist()}")

    # 2. First transformer layer
    from transformers.cache_utils import DynamicCache
    position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    layer0_out = model.model.layers[0](hidden, position_ids=position_ids)
    h0 = layer0_out[0]
    h0_cpu = h0[0, 0, :10].cpu()
    print(f"[TRACE] layer[0]: mean={h0.float().mean().item():.6f} std={h0.float().std().item():.6f} first10={h0_cpu.tolist()}")

    # 3. Full model forward (through all layers + norm)
    base_out = model.model(inputs["input_ids"])
    hs = base_out.last_hidden_state
    hs_cpu = hs[0, -1, :10].cpu()
    print(f"[TRACE] model.model (all layers+norm): mean={hs.float().mean().item():.6f} std={hs.float().std().item():.6f} last_pos_first10={hs_cpu.tolist()}")

    # 4. lm_head
    logits = model.lm_head(hs)
    logits_cpu = logits[0, -1, :].cpu()
    print(f"[TRACE] lm_head logits: mean={logits_cpu.float().mean().item():.6f} std={logits_cpu.float().std().item():.6f}")
    print(f"[TRACE] logits[0,-1,:5]: {logits_cpu[:5].tolist()}")
    print(f"[TRACE] NaN={torch.isnan(logits_cpu).sum().item()} Inf={torch.isinf(logits_cpu).sum().item()}")
    next_tok = logits_cpu.argmax().item()
    print(f"[TRACE] Argmax token: {next_tok} -> '{tokenizer.decode([next_tok])}'")

print(f"[INFERENCE] Generating 200 tokens...")

t0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
    )
torch.cuda.synchronize()
gen_dt = time.time() - t0

new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
tokens_per_sec = new_tokens / gen_dt

# Debug: inspect raw output tokens
out_cpu = outputs[0].cpu()
print(f"[DEBUG] Output shape: {outputs.shape}, dtype: {outputs.dtype}")
print(f"[DEBUG] First 20 tokens: {out_cpu[:20].tolist()}")
print(f"[DEBUG] Last 20 tokens: {out_cpu[-20:].tolist()}")
print(f"[DEBUG] Min: {out_cpu.min().item()}, Max: {out_cpu.max().item()}")

text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

print(f"[INFERENCE] Done: {gen_dt:.2f}s — {new_tokens} tokens — {tokens_per_sec:.1f} tok/s")
print(f"\n--- Output ---")
print(text[:500])
print(f"--- End ---")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"MODEL:     {MODEL}")
print(f"CPU LOAD:  {load_cpu_dt:.1f}s")
print(f"GPU LOAD:  {load_gpu_dt:.1f}s ({mem_mb:.0f} MB)")
print(f"INFERENCE: {gen_dt:.2f}s ({new_tokens} tokens, {tokens_per_sec:.1f} tok/s)")
print(f"{'='*60}")
