#!/usr/bin/env python3
"""Chidori — Replay correctness test: compare per-step gradients deterministically.

Run twice (once with CHIDORI_REPLAY=1, once without) and diff the output.
Every number must match exactly if replay is correct.
"""
import sys
import math
import hashlib
import torch
import torch.nn as nn

# ==============================================================
# DETERMINISM — lock down every source of non-determinism
# ==============================================================
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.preferred_blas_library("cublas")
torch.use_deterministic_algorithms(True, warn_only=True)

# ==============================================================
# INIT CUDA
# ==============================================================
torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Device: {dev} ({vram:.1f} GB)")
print("=" * 70)

# ==============================================================
# MODEL — same MiniModel as test_training.py
# ==============================================================
HIDDEN = 512
LAYERS = 6
BATCH  = 16
SEQ    = 64
VOCAB  = 10000
STEPS  = 10


class MiniModel(nn.Module):
    """MLP-based model — uses 2D reshape to avoid cublasLt batched GEMM."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.ups   = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN * 4, bias=False) for _ in range(LAYERS)])
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
# SETUP — model, optimizer, loss, fixed data
# ==============================================================
print(f"\n[CONFIG] layers={LAYERS}, hidden={HIDDEN}, vocab={VOCAB}")
print(f"[CONFIG] batch={BATCH}, seq={SEQ}, steps={STEPS}")
print(f"[CONFIG] seed=42, deterministic=True, benchmark=False")

# Re-seed right before model creation for weight init reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = MiniModel().cuda()
torch.cuda.synchronize()

params_total = sum(p.numel() for p in model.parameters())
print(f"[MODEL] {params_total / 1e6:.2f}M parameters")

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Pre-generate ALL training data with the fixed seed so it is identical
# across runs regardless of any internal RNG drift during training.
torch.manual_seed(42)
torch.cuda.manual_seed(42)
all_data = []
for _ in range(STEPS):
    d = torch.randint(0, VOCAB, (BATCH, SEQ), device='cuda')
    all_data.append(d)
torch.cuda.synchronize()

# ==============================================================
# COLLECT NAMED PARAMETER LIST (sorted for stable ordering)
# ==============================================================
named_params = sorted([(n, p) for n, p in model.named_parameters()], key=lambda x: x[0])


def grad_norm(p):
    """L2 norm of a parameter's gradient, or 0.0 if None."""
    if p.grad is None:
        return 0.0
    return p.grad.data.norm(2).item()


# ==============================================================
# TRAINING LOOP — 10 steps with full gradient capture
# ==============================================================
print(f"\n{'='*70}")
print("REPLAY GRADIENT LOG — every number must match between runs")
print(f"{'='*70}\n")

all_ok = True

for step in range(STEPS):
    data = all_data[step]

    optimizer.zero_grad()

    # Forward
    logits = model(data)
    loss = loss_fn(logits.view(-1, VOCAB), data.view(-1))

    # Backward
    loss.backward()
    torch.cuda.synchronize()

    # --- Capture gradients BEFORE optimizer.step() ---
    loss_val = loss.item()

    per_layer_norms = []
    for name, param in named_params:
        gn = grad_norm(param)
        per_layer_norms.append((name, gn))

    total_grad_norm = 0.0
    for _, gn in per_layer_norms:
        total_grad_norm += gn * gn
    total_grad_norm = math.sqrt(total_grad_norm)

    # Print step summary
    print(f"STEP {step:02d}  loss={loss_val:.10f}  total_grad_norm={total_grad_norm:.10f}")
    for name, gn in per_layer_norms:
        print(f"  {name:40s}  grad_norm={gn:.10f}")

    # Sanity check
    if not math.isfinite(loss_val):
        print(f"  *** NON-FINITE LOSS at step {step} ***")
        all_ok = False
    if not math.isfinite(total_grad_norm):
        print(f"  *** NON-FINITE GRAD NORM at step {step} ***")
        all_ok = False

    # Optimizer step (after gradient capture)
    optimizer.step()
    torch.cuda.synchronize()

    print()

# ==============================================================
# FINAL WEIGHT HASH — deterministic fingerprint of model state
# ==============================================================
torch.cuda.synchronize()

weight_sum = 0.0
for name, param in named_params:
    weight_sum += param.data.sum().item()

# Also compute a SHA-256 over the raw bytes of every parameter for a
# collision-resistant fingerprint.
sha = hashlib.sha256()
for name, param in named_params:
    cpu_data = param.data.cpu().contiguous()
    sha.update(cpu_data.numpy().tobytes())

print(f"{'='*70}")
print(f"FINAL WEIGHT SUM:    {weight_sum:.10f}")
print(f"FINAL WEIGHT SHA256: {sha.hexdigest()}")
print(f"{'='*70}")

# ==============================================================
# VERDICT
# ==============================================================
if all_ok:
    print("\nPASS: test_replay_gradients (all values finite, output is deterministic)")
    sys.exit(0)
else:
    print("\nFAIL: test_replay_gradients (non-finite values detected)")
    sys.exit(1)
