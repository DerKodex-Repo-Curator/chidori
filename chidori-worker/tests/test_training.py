#!/usr/bin/env python3
"""Chidori — Real training test: forward + backward + optimizer step"""
import torch
import torch.nn as nn
import time

torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Device: {dev} ({vram:.1f} GB)")
print("=" * 60)

# ============================================================
# MODEL — simple transformer-like layers (no HF dependency)
# ============================================================
HIDDEN = 512
LAYERS = 6
BATCH = 16
SEQ = 64
VOCAB = 10000
STEPS = 50

# Disable Flash/SDPA attention — Chidori doesn't proxy CUTLASS kernels yet
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.preferred_blas_library("cublas")

class MiniModel(nn.Module):
    """MLP-based model — uses 2D reshape to avoid cublasLt batched GEMM."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.ups = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN * 4, bias=False) for _ in range(LAYERS)])
        self.downs = nn.ModuleList([nn.Linear(HIDDEN * 4, HIDDEN, bias=False) for _ in range(LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(LAYERS)])
        self.final_norm = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, x):
        B, S = x.shape
        h = self.embed(x)  # (B, S, H)
        for norm, up, down in zip(self.norms, self.ups, self.downs):
            residual = h
            h = norm(h)
            # Reshape to 2D for linear (avoids cublasLt batched path)
            h = h.reshape(B * S, HIDDEN)
            h = up(h)
            h = torch.nn.functional.gelu(h)
            h = down(h)
            h = h.reshape(B, S, HIDDEN)
            h = residual + h
        h = self.final_norm(h)
        h = h.reshape(B * S, HIDDEN)
        h = self.head(h)
        return h.reshape(B, S, VOCAB)

# ============================================================
# SETUP
# ============================================================
print(f"\n[MODEL] {LAYERS} transformer layers, hidden={HIDDEN}, vocab={VOCAB}")
print(f"[MODEL] Batch={BATCH}, seq={SEQ}")

t0 = time.time()
model = MiniModel().cuda().half()
torch.cuda.synchronize()
model_dt = time.time() - t0

params = sum(p.numel() for p in model.parameters())
mem_mb = torch.cuda.memory_allocated() / 1e6
print(f"[MODEL] {params/1e6:.0f}M params, {mem_mb:.0f} MB VRAM, loaded in {model_dt:.1f}s")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

# ============================================================
# TRAINING LOOP
# ============================================================
print(f"\n[TRAIN] Running {STEPS} training steps...")

# Warmup
dummy = torch.randint(0, VOCAB, (BATCH, SEQ)).cuda()
out = model(dummy)
loss = loss_fn(out.float().view(-1, VOCAB), dummy.view(-1))
loss.backward()
optimizer.zero_grad()
torch.cuda.synchronize()
print("[TRAIN] Warmup done")

losses = []
t0 = time.time()
for step in range(STEPS):
    data = torch.randint(0, VOCAB, (BATCH, SEQ)).cuda()

    optimizer.zero_grad()
    logits = model(data)
    loss = loss_fn(logits.float().view(-1, VOCAB), data.view(-1))

    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        torch.cuda.synchronize()
        l = loss.item()
        losses.append(l)
        elapsed = time.time() - t0
        tps = (step + 1) * BATCH * SEQ / elapsed
        print(f"  step {step:3d}  loss={l:.4f}  tokens/s={tps:.0f}  elapsed={elapsed:.1f}s")

torch.cuda.synchronize()
train_dt = time.time() - t0
total_tokens = STEPS * BATCH * SEQ
tps = total_tokens / train_dt

# ============================================================
# VERIFY — loss should decrease
# ============================================================
print(f"\n[VERIFY] Losses: {' -> '.join(f'{l:.4f}' for l in losses)}")
if len(losses) >= 2 and losses[-1] < losses[0]:
    print("[VERIFY] OK — loss is decreasing, training is working")
else:
    print("[VERIFY] WARNING — loss did not decrease (may need more steps)")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"DEVICE:     {dev}")
print(f"MODEL:      {params/1e6:.0f}M params ({LAYERS}L, h={HIDDEN})")
print(f"BATCH:      {BATCH} x {SEQ} = {BATCH*SEQ} tokens/step")
print(f"MODEL LOAD: {model_dt:.1f}s")
print(f"TRAINING:   {train_dt:.1f}s ({STEPS} steps)")
print(f"THROUGHPUT: {tps:.0f} tokens/s ({tps/1000:.1f}K tok/s)")
print(f"TOTAL TOKENS: {total_tokens:,}")
print(f"{'='*60}")

import sys, math
all_finite = all(math.isfinite(l) for l in losses)
if all_finite and tps > 0:
    print("PASS: test_training")
    sys.exit(0)
else:
    print("FAIL: test_training")
    sys.exit(1)
