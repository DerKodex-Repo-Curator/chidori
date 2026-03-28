#!/usr/bin/env python3
"""Chidori stress test — partial offload training, both GPUs sweating.

Splits a ~800M param model across local GPU (device 0) + remote GPU (device 1).
Manual placement (no accelerate) so we control exactly what runs where.
Both GPUs should show sustained utilization for 60+ seconds.

Requires CHIDORI_LOCAL_DEVICES=0.
"""
import os, torch, torch.nn as nn, time, sys, math
os.environ["TORCHDYNAMO_DISABLE"] = "1"

torch._C._cuda_init()
n_devs = torch.cuda.device_count()
if n_devs < 2:
    print(f"SKIP: need 2 GPUs for offload test, found {n_devs}")
    print("Set CHIDORI_LOCAL_DEVICES=0 to enable partial offload")
    sys.exit(0)

dev0 = torch.cuda.get_device_name(0)
dev1 = torch.cuda.get_device_name(1)
mem0 = torch.cuda.get_device_properties(0).total_memory / 1e9
mem1 = torch.cuda.get_device_properties(1).total_memory / 1e9
print(f"Device 0 (local):  {dev0} ({mem0:.0f} GB)")
print(f"Device 1 (remote): {dev1} ({mem1:.0f} GB)")
print("=" * 70)

HIDDEN = 1024
LAYERS = 12
BATCH = 32
SEQ = 128
VOCAB = 32000
STEPS = 500

class TransformerBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, hidden * 4, bias=False)
        self.down = nn.Linear(hidden * 4, hidden, bias=False)

    def forward(self, h):
        residual = h
        B_S = h.shape[0] * h.shape[1] if h.dim() == 3 else h.shape[0]
        h = self.norm(h)
        orig_shape = h.shape
        h = h.reshape(-1, HIDDEN)
        h = self.up(h)
        h = torch.nn.functional.gelu(h)
        h = self.down(h)
        h = h.reshape(orig_shape)
        return residual + h

class SplitModel(nn.Module):
    """Model with explicit device placement — no accelerate needed."""
    def __init__(self, dev_local, dev_remote, split_at):
        super().__init__()
        self.dev_local = dev_local
        self.dev_remote = dev_remote
        self.split_at = split_at

        # Local half
        self.embed = nn.Embedding(VOCAB, HIDDEN).to(dev_local)
        self.local_layers = nn.ModuleList(
            [TransformerBlock(HIDDEN).to(dev_local) for _ in range(split_at)]
        )

        # Remote half
        self.remote_layers = nn.ModuleList(
            [TransformerBlock(HIDDEN).to(dev_remote) for _ in range(LAYERS - split_at)]
        )
        self.final_norm = nn.LayerNorm(HIDDEN).to(dev_remote)
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False).to(dev_remote)

    def forward(self, x):
        # x starts on dev_local
        h = self.embed(x)  # on dev_local

        # Local layers
        for layer in self.local_layers:
            h = layer(h)

        # Cross-device transfer
        h = h.to(self.dev_remote)

        # Remote layers
        for layer in self.remote_layers:
            h = layer(h)

        h = self.final_norm(h)
        B, S, H = h.shape
        h = h.reshape(B * S, H)
        return self.head(h).reshape(B, S, VOCAB)

split_at = LAYERS // 2
print(f"[CONFIG] {LAYERS}L, h={HIDDEN}, vocab={VOCAB}, batch={BATCH}x{SEQ}, steps={STEPS}")
print(f"[SPLIT] Layers 0-{split_at-1} → device 0 (local)")
print(f"[SPLIT] Layers {split_at}-{LAYERS-1} + head → device 1 (remote)")

print(f"[LOAD] Building model...")
t0 = time.time()
model = SplitModel(
    dev_local=torch.device("cuda:0"),
    dev_remote=torch.device("cuda:1"),
    split_at=split_at,
)
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
load_dt = time.time() - t0

params = sum(p.numel() for p in model.parameters())
vram0 = torch.cuda.memory_allocated(0) / 1e9
vram1 = torch.cuda.memory_allocated(1) / 1e9
print(f"[LOAD] {params/1e6:.0f}M params, loaded in {load_dt:.1f}s")
print(f"[LOAD] Dev0 VRAM: {vram0:.2f} GB")
print(f"[LOAD] Dev1 VRAM: {vram1:.2f} GB")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

# Warmup
print("[TRAIN] Warmup...")
d = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda:0")
out = model(d)
target = d.to(out.device)
loss_fn(out.view(-1, VOCAB), target.view(-1)).backward()
optimizer.step()
optimizer.zero_grad()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)

print(f"[TRAIN] Running {STEPS} steps — BOTH GPUs should show sustained usage")
print(f"[TRAIN] Check: nvidia-smi on both machines NOW")
print("-" * 70)

losses = []
t_start = time.time()
for step in range(STEPS):
    data = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda:0")
    optimizer.zero_grad()
    logits = model(data)
    target = data.to(logits.device)
    loss = loss_fn(logits.view(-1, VOCAB), target.view(-1))
    loss.backward()
    optimizer.step()

    if step % 30 == 0:
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        l = loss.item()
        losses.append(l)
        elapsed = time.time() - t_start
        tps = (step + 1) * BATCH * SEQ / elapsed if elapsed > 0 else 0
        v0 = torch.cuda.memory_allocated(0) / 1e9
        v1 = torch.cuda.memory_allocated(1) / 1e9
        print(f"  step {step:4d}  loss={l:.4f}  tok/s={tps:,.0f}  dev0={v0:.1f}GB  dev1={v1:.1f}GB  elapsed={elapsed:.1f}s")

torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
total_time = time.time() - t_start
total_tokens = STEPS * BATCH * SEQ
tps = total_tokens / total_time

print("-" * 70)
print(f"[VERIFY] Losses: {' → '.join(f'{l:.4f}' for l in losses)}")

ok = True
if not all(math.isfinite(l) for l in losses):
    print("[VERIFY] FAIL — NaN/Inf in losses")
    ok = False
else:
    print("[VERIFY] OK — all losses finite")

if total_time < 30:
    print(f"[VERIFY] WARNING — only ran {total_time:.0f}s, expected 60+")

peak0 = torch.cuda.max_memory_allocated(0) / 1e9
peak1 = torch.cuda.max_memory_allocated(1) / 1e9
print(f"\n{'=' * 70}")
print(f"LOCAL GPU:   {dev0}")
print(f"REMOTE GPU:  {dev1}")
print(f"MODEL:       {params/1e6:.0f}M params ({LAYERS}L, h={HIDDEN}, FP32)")
print(f"SPLIT:       layers 0-{split_at-1} local, {split_at}-{LAYERS-1}+head remote")
print(f"BATCH:       {BATCH} x {SEQ} = {BATCH*SEQ:,} tokens/step")
print(f"STEPS:       {STEPS}")
print(f"GPU TIME:    {total_time:.1f}s")
print(f"THROUGHPUT:  {tps:,.0f} tokens/s ({tps/1000:.1f}K tok/s)")
print(f"TOTAL:       {total_tokens:,} tokens")
print(f"PEAK VRAM 0: {peak0:.2f} GB")
print(f"PEAK VRAM 1: {peak1:.2f} GB")
print(f"{'=' * 70}")

if ok and tps > 0 and peak0 > 0.5 and peak1 > 0.5:
    print("PASS: test_stress_offload")
else:
    if peak0 < 0.5:
        print(f"FAIL: device 0 barely used (peak {peak0:.2f} GB)")
    if peak1 < 0.5:
        print(f"FAIL: device 1 barely used (peak {peak1:.2f} GB)")
    print("FAIL: test_stress_offload")
    sys.exit(1)
