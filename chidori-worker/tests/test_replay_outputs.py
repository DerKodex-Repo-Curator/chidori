#!/usr/bin/env python3
"""Chidori — Replay determinism test.

Validates that forward-pass outputs are bit-identical with and without
CHIDORI_REPLAY=1.  Run twice (once normal, once with replay) and diff
the printed numbers.  Any difference means replay is corrupting data.

Usage:
    # Run 1 — baseline (no replay)
    python tests/test_replay_outputs.py > /tmp/replay_baseline.txt

    # Run 2 — with replay enabled
    CHIDORI_REPLAY=1 python tests/test_replay_outputs.py > /tmp/replay_test.txt

    # Compare
    diff /tmp/replay_baseline.txt /tmp/replay_test.txt
    # Must produce zero differences.
"""
import sys
import os
import math
import hashlib
import struct
import torch
import torch.nn as nn

# ==============================================================
# DETERMINISM — lock everything down
# ==============================================================
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Disable non-deterministic CUDA algorithms globally (PyTorch >= 1.11)
if hasattr(torch, "use_deterministic_algorithms"):
    torch.use_deterministic_algorithms(True, warn_only=True)

# Disable Flash/SDPA attention — Chidori doesn't proxy CUTLASS kernels
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Force cuBLAS (not cublasLt) for matrix multiplies
torch.backends.cuda.preferred_blas_library("cublas")

# Reduce cuBLAS workspace to force deterministic algo selection
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ==============================================================
# CUDA INIT
# ==============================================================
torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Device: {dev} ({vram:.1f} GB)", file=sys.stderr)
print(f"CHIDORI_REPLAY={os.environ.get('CHIDORI_REPLAY', '<not set>')}", file=sys.stderr)

# ==============================================================
# MODEL — identical to test_training.py MiniModel
# ==============================================================
HIDDEN = 512
LAYERS = 6
BATCH  = 16
SEQ    = 64
VOCAB  = 10000

WARMUP_STEPS = 5
MEASURE_STEPS = 10  # steps 6-15

class MiniModel(nn.Module):
    """MLP-based model — uses 2D reshape to avoid cublasLt batched GEMM."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.ups   = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN * 4, bias=False) for _ in range(LAYERS)])
        self.downs = nn.ModuleList([nn.Linear(HIDDEN * 4, HIDDEN, bias=False) for _ in range(LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(LAYERS)])
        self.final_norm = nn.LayerNorm(HIDDEN)
        self.head  = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, x):
        B, S = x.shape
        h = self.embed(x)  # (B, S, H)
        for norm, up, down in zip(self.norms, self.ups, self.downs):
            residual = h
            h = norm(h)
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
# SETUP — deterministic init
# ==============================================================
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
model = MiniModel().cuda()
torch.cuda.synchronize()

params = sum(p.numel() for p in model.parameters())
print(f"Model: {params/1e6:.1f}M params, {LAYERS}L h={HIDDEN} v={VOCAB}", file=sys.stderr)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Fixed RNG for generating input data — reset seed so inputs are identical
# across runs regardless of model init consuming RNG state
g = torch.Generator()
g.manual_seed(SEED + 1000)

def make_input():
    """Generate a fixed input batch using the dedicated generator."""
    return torch.randint(0, VOCAB, (BATCH, SEQ), generator=g).cuda()

# ==============================================================
# WARMUP — 5 steps (lets replay activate if CHIDORI_REPLAY=1)
# ==============================================================
print(f"Running {WARMUP_STEPS} warmup steps...", file=sys.stderr)
for step in range(WARMUP_STEPS):
    data = make_input()
    optimizer.zero_grad()
    logits = model(data)
    loss = loss_fn(logits.view(-1, VOCAB), data.view(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

print("Warmup complete.", file=sys.stderr)

# ==============================================================
# FIXED PROBE INPUT — same tensor used every measurement step
# ==============================================================
probe_gen = torch.Generator()
probe_gen.manual_seed(SEED + 9999)
probe_input = torch.randint(0, VOCAB, (BATCH, SEQ), generator=probe_gen).cuda()

# ==============================================================
# MEASUREMENT — steps 6-15, print exact outputs
# ==============================================================
print(f"Running {MEASURE_STEPS} measurement steps (6-15)...", file=sys.stderr)
print("=" * 72)
print("REPLAY DETERMINISM TEST OUTPUT")
print("=" * 72)

all_losses = []

for step_idx in range(MEASURE_STEPS):
    global_step = WARMUP_STEPS + step_idx + 1  # 6..15

    # --- Forward pass on the fixed probe input ---
    with torch.no_grad():
        probe_logits = model(probe_input)  # (B, S, VOCAB)
    torch.cuda.synchronize()

    # Extract first sample, first 10 token positions
    # probe_logits[0, :10, :] is (10, VOCAB) — take argmax logit value per position
    first_sample = probe_logits[0, :10, :]  # (10, VOCAB)
    # Print the max logit and its index for each of the 10 positions
    max_vals, max_idxs = first_sample.max(dim=-1)  # each (10,)

    # Also print the raw logit values for vocab indices 0..9 at position 0
    pos0_logits = probe_logits[0, 0, :10].cpu()  # first 10 vocab logits at position 0

    print(f"\n--- Step {global_step} ---")
    print(f"LOGITS_POS0_RAW: {' '.join(f'{v:.8f}' for v in pos0_logits.tolist())}")
    print(f"MAX_LOGIT_VALS:  {' '.join(f'{v:.8f}' for v in max_vals.cpu().tolist())}")
    print(f"MAX_LOGIT_IDXS:  {' '.join(f'{int(v)}' for v in max_idxs.cpu().tolist())}")

    # --- Training step (with gradient) using fresh input from the RNG stream ---
    train_data = make_input()
    optimizer.zero_grad()
    logits = model(train_data)
    loss = loss_fn(logits.view(-1, VOCAB), train_data.view(-1))

    loss_val = loss.item()
    all_losses.append(loss_val)
    print(f"LOSS:            {loss_val:.8f}")

    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

# ==============================================================
# PARAMETER CHECKSUM — hash every model weight
# ==============================================================
print("\n" + "=" * 72)
print("PARAMETER CHECKSUM")
print("=" * 72)

hasher = hashlib.sha256()
param_sum = 0.0
param_count = 0

for name, param in sorted(model.named_parameters()):
    p = param.detach().cpu()
    # Add raw bytes to hash
    hasher.update(p.numpy().tobytes())
    # Accumulate statistics
    psum = p.double().sum().item()
    param_sum += psum
    param_count += p.numel()
    print(f"PARAM {name:40s}  sum={psum:.8f}  mean={p.double().mean().item():.8f}  std={p.double().std().item():.8f}")

print(f"\nTOTAL_PARAM_SUM:   {param_sum:.8f}")
print(f"TOTAL_PARAM_COUNT: {param_count}")
print(f"SHA256:            {hasher.hexdigest()}")

# ==============================================================
# LOSS TRAJECTORY
# ==============================================================
print("\n" + "=" * 72)
print("LOSS TRAJECTORY (steps 6-15)")
print("=" * 72)
for i, l in enumerate(all_losses):
    print(f"STEP_{WARMUP_STEPS + i + 1:02d}_LOSS: {l:.8f}")

# ==============================================================
# SUMMARY
# ==============================================================
print("\n" + "=" * 72)
all_finite = all(math.isfinite(l) for l in all_losses)
if all_finite:
    print("PASS: test_replay_outputs (all values finite, ready for diff)")
    sys.exit(0)
else:
    print("FAIL: test_replay_outputs (non-finite loss detected)")
    sys.exit(1)
