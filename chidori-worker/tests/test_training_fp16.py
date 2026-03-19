#!/usr/bin/env python3
"""Chidori — FP16 training benchmark (bias=False to avoid cublasLt)"""
import torch, torch.nn as nn, time

torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)

class MLP(nn.Module):
    def __init__(self, H, V, L):
        super().__init__()
        self.embed = nn.Embedding(V, H)
        self.linears = nn.ModuleList()
        for _ in range(L):
            self.linears.append(nn.Linear(H, H*4, bias=False))
            self.linears.append(nn.Linear(H*4, H, bias=False))
        self.head = nn.Linear(H, V, bias=False)

    def forward(self, x):
        B, S = x.shape
        h = self.embed(x).reshape(B*S, -1)
        for i in range(0, len(self.linears), 2):
            r = h
            h = torch.nn.functional.gelu(self.linears[i](h))
            h = self.linears[i+1](h) + r
        return self.head(h).reshape(B, S, -1)

HIDDEN, VOCAB, LAYERS = 512, 10000, 6
BATCH, SEQ, STEPS = 32, 64, 50

model = MLP(HIDDEN, VOCAB, LAYERS).cuda().half()
P = sum(p.numel() for p in model.parameters())
mem = torch.cuda.memory_allocated() / 1e6
print(f"Model: {P/1e6:.0f}M params, {mem:.0f} MB on {dev}")

opt = torch.optim.SGD(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Warmup
d = torch.randint(0, VOCAB, (BATCH, SEQ)).cuda()
o = model(d)
loss_fn(o.float().reshape(-1, VOCAB), d.reshape(-1)).backward()
opt.zero_grad()
torch.cuda.synchronize()
print("Warmup done")

t0 = time.time()
losses = []
for step in range(STEPS):
    d = torch.randint(0, VOCAB, (BATCH, SEQ)).cuda()
    opt.zero_grad()
    logits = model(d)
    loss = loss_fn(logits.float().reshape(-1, VOCAB), d.reshape(-1))
    loss.backward()
    opt.step()
    if step % 10 == 0:
        torch.cuda.synchronize()
        l = loss.item()
        losses.append(l)
        elapsed = time.time() - t0
        tps = (step + 1) * BATCH * SEQ / elapsed
        print(f"  step {step:3d}  loss={l:.4f}  tokens/s={tps:.0f}  elapsed={elapsed:.1f}s")

torch.cuda.synchronize()
dt = time.time() - t0
tps = STEPS * BATCH * SEQ / dt

import sys

print(f"\nLosses: {losses}")
# Verify losses are finite (not NaN/Inf) and training ran
import math
all_finite = all(math.isfinite(l) for l in losses)
if all_finite:
    print("OK — all losses finite")
else:
    print("WARNING — NaN/Inf in losses")

print(f"\n{'='*60}")
print(f"DEVICE:     {dev}")
print(f"MODEL:      {P/1e6:.0f}M params ({LAYERS}L, h={HIDDEN})")
print(f"BATCH:      {BATCH} x {SEQ} = {BATCH*SEQ} tokens/step")
print(f"TRAINING:   {dt:.1f}s ({STEPS} steps)")
print(f"THROUGHPUT: {tps:.0f} tokens/s ({tps/1000:.1f}K tok/s)")
print(f"{'='*60}")

if all_finite and tps > 0:
    print("PASS: test_training_fp16")
    sys.exit(0)
else:
    print("FAIL: test_training_fp16")
    sys.exit(1)
