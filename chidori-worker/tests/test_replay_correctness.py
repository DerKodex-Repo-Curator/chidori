#!/usr/bin/env python3
"""Chidori — Replay correctness test.

Validates that two runs with identical seeds produce identical results.
Run twice (once without CHIDORI_REPLAY=1, once with) and compare output.

Usage:
    # Run 1 — baseline (or with Chidori interception)
    python test_replay_correctness.py

    # Run 2 — with replay
    CHIDORI_REPLAY=1 python test_replay_correctness.py

    # Compare: losses and weight hash must be IDENTICAL between runs.
"""
import hashlib
import io
import math
import os
import sys
import time

import torch
import torch.nn as nn

# ==============================================================
# DETERMINISM SETTINGS — must be set BEFORE any CUDA operations
# ==============================================================
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Disable Flash/SDPA attention — Chidori doesn't proxy CUTLASS kernels yet
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.preferred_blas_library("cublas")

# ==============================================================
# CUDA INIT
# ==============================================================
torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
replay_mode = os.environ.get("CHIDORI_REPLAY", "0")
print(f"Device: {dev} ({vram:.1f} GB)")
print(f"CHIDORI_REPLAY={replay_mode}")
print("=" * 60)

# ==============================================================
# MODEL — identical to test_training.py
# ==============================================================
HIDDEN = 512
LAYERS = 6
BATCH = 16
SEQ = 64
VOCAB = 10000
STEPS = 10

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

# ==============================================================
# SETUP — seed again right before model creation for safety
# ==============================================================
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

print(f"\n[MODEL] {LAYERS} transformer layers, hidden={HIDDEN}, vocab={VOCAB}")
print(f"[MODEL] Batch={BATCH}, seq={SEQ}, steps={STEPS}")

t0 = time.time()
model = MiniModel().cuda()  # FP32
torch.cuda.synchronize()
model_dt = time.time() - t0

params = sum(p.numel() for p in model.parameters())
mem_mb = torch.cuda.memory_allocated() / 1e6
print(f"[MODEL] {params/1e6:.0f}M params, {mem_mb:.0f} MB VRAM, loaded in {model_dt:.1f}s")

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# ==============================================================
# TRAINING LOOP — deterministic data via generator with fixed seed
# ==============================================================
print(f"\n[TRAIN] Running {STEPS} training steps (seed={SEED})...")

# Re-seed before training to ensure identical data sequence
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
g = torch.Generator(device='cpu')
g.manual_seed(SEED)

losses = []
t0 = time.time()
for step in range(STEPS):
    # Generate training data with explicit generator for reproducibility
    data = torch.randint(0, VOCAB, (BATCH, SEQ), generator=g).cuda()

    optimizer.zero_grad()
    logits = model(data)
    loss = loss_fn(logits.view(-1, VOCAB), data.view(-1))

    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    l = loss.item()

    losses.append(l)
    elapsed = time.time() - t0
    print(f"  step {step:3d}  loss={l:.6f}  elapsed={elapsed:.1f}s")

torch.cuda.synchronize()
train_dt = time.time() - t0

# ==============================================================
# COMPUTE WEIGHT HASH — deterministic fingerprint of final state
# ==============================================================
def compute_weight_hash(model):
    """SHA-256 hash of all model parameters, concatenated in state_dict order."""
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state.keys()):
        tensor = state[key].cpu()
        # Use a BytesIO buffer to get deterministic binary representation
        buf = io.BytesIO()
        torch.save(tensor, buf)
        h.update(buf.getvalue())
    return h.hexdigest()

weight_hash = compute_weight_hash(model)

# ==============================================================
# RESULTS
# ==============================================================
print(f"\n{'='*60}")
print(f"REPLAY CORRECTNESS TEST RESULTS")
print(f"{'='*60}")
print(f"DEVICE:       {dev}")
print(f"REPLAY MODE:  {replay_mode}")
print(f"SEED:         {SEED}")
print(f"STEPS:        {STEPS}")
print(f"TRAIN TIME:   {train_dt:.1f}s")
print(f"{'='*60}")
print(f"\nLOSSES:")
for i, l in enumerate(losses):
    print(f"  step {i:3d}: {l:.6f}")

print(f"\nWEIGHT HASH: {weight_hash}")
print(f"{'='*60}")

# ==============================================================
# VALIDATION
# ==============================================================
all_finite = all(math.isfinite(l) for l in losses)
if not all_finite:
    print("\nFAIL: non-finite loss detected")
    sys.exit(1)

if len(losses) >= 2 and losses[-1] >= losses[0]:
    print("\nWARNING: loss did not decrease (may need investigation)")

print("\nPASS: test_replay_correctness")
print("  To validate replay: run twice and compare LOSSES + WEIGHT HASH")
sys.exit(0)
