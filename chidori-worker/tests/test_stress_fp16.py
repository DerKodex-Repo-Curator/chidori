#!/usr/bin/env python3
"""Chidori stress test — FP16 training, sustained GPU load for 60+ seconds.

Large model (150M params FP16), big batches, 1000 steps.
FP16 is faster per step so needs more steps to sustain load.
"""
import torch, torch.nn as nn, time, sys, math

torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Device: {dev} ({vram_gb:.1f} GB)")
print("=" * 70)

HIDDEN = 1024
LAYERS = 12
BATCH = 64
SEQ = 128
VOCAB = 32000
STEPS = 1000

class HeavyModel(nn.Module):
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
        h = self.embed(x)
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
        return self.head(h).reshape(B, S, VOCAB)

print(f"[CONFIG] {LAYERS}L, h={HIDDEN}, vocab={VOCAB}, batch={BATCH}x{SEQ}, steps={STEPS}")
print(f"[LOAD] Building model (FP16)...")
t0 = time.time()
model = HeavyModel().cuda().half()
torch.cuda.synchronize()
params = sum(p.numel() for p in model.parameters())
mem_gb = torch.cuda.memory_allocated() / 1e9
print(f"[LOAD] {params/1e6:.0f}M params, {mem_gb:.2f} GB VRAM, loaded in {time.time()-t0:.1f}s")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

# Warmup
print("[TRAIN] Warmup...")
d = torch.randint(0, VOCAB, (BATCH, SEQ)).cuda()
out = model(d)
loss_fn(out.float().reshape(-1, VOCAB), d.reshape(-1)).backward()
optimizer.step()
optimizer.zero_grad()
torch.cuda.synchronize()

print(f"[TRAIN] Running {STEPS} steps — GPU should be pinned for 60+ seconds")
print("-" * 70)

losses = []
t_start = time.time()
for step in range(STEPS):
    data = torch.randint(0, VOCAB, (BATCH, SEQ)).cuda()
    optimizer.zero_grad()
    logits = model(data)
    loss = loss_fn(logits.float().reshape(-1, VOCAB), data.reshape(-1))
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        torch.cuda.synchronize()
        l = loss.item()
        losses.append(l)
        elapsed = time.time() - t_start
        tps = (step + 1) * BATCH * SEQ / elapsed if elapsed > 0 else 0
        mem_used = torch.cuda.memory_allocated() / 1e9
        print(f"  step {step:4d}  loss={l:.4f}  tok/s={tps:,.0f}  mem={mem_used:.1f}GB  elapsed={elapsed:.1f}s")

torch.cuda.synchronize()
total_time = time.time() - t_start
total_tokens = STEPS * BATCH * SEQ
tps = total_tokens / total_time

print("-" * 70)
print(f"[VERIFY] Losses: {' → '.join(f'{l:.4f}' for l in losses)}")

ok = True
if not all(math.isfinite(l) for l in losses):
    print("[VERIFY] FAIL — NaN/Inf in losses")
    ok = False
elif len(losses) >= 2 and losses[-1] >= losses[0]:
    print("[VERIFY] WARNING — loss did not decrease (random data, may be OK)")
else:
    print("[VERIFY] OK — loss decreasing, all finite")

if total_time < 30:
    print(f"[VERIFY] WARNING — only ran {total_time:.0f}s, expected 60+")

print(f"\n{'=' * 70}")
print(f"DEVICE:      {dev}")
print(f"MODEL:       {params/1e6:.0f}M params ({LAYERS}L, h={HIDDEN}, FP16)")
print(f"BATCH:       {BATCH} x {SEQ} = {BATCH*SEQ:,} tokens/step")
print(f"STEPS:       {STEPS}")
print(f"GPU TIME:    {total_time:.1f}s")
print(f"THROUGHPUT:  {tps:,.0f} tokens/s ({tps/1000:.1f}K tok/s)")
print(f"TOTAL:       {total_tokens:,} tokens")
print(f"PEAK VRAM:   {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"{'=' * 70}")

if ok and tps > 0:
    print("PASS: test_stress_fp16")
else:
    print("FAIL: test_stress_fp16")
    sys.exit(1)
