# Chidori Test Suite

## Prerequisites

- Linux x86_64
- GCC, Make
- CUDA Toolkit 12.x+ (for server build)
- Python 3.10+ with:
  - `torch` (PyTorch with CUDA)
  - `transformers`, `sentencepiece`, `protobuf` (for inference test)

## Clone & Build

```bash
git clone https://github.com/fpolica91/chidori.git
cd chidori/chidori-worker
make all CUDA_PATH=/usr/local/cuda
cd build && ln -sf libchidori.so libcuda.so.1 && cd ..
```

## Run Tests

### Loopback (server + client on same machine with GPU)

```bash
./tests/run_all.sh
```

This starts a local Chidori server, runs all tests, and prints results.

### Remote server

On the GPU server:
```bash
./build/chidori-worker --standalone 43211
```

On the client (no GPU needed):
```bash
./tests/run_all.sh <server_ip> 43211
```

### Example output

```
=== Chidori Test Suite ===
Server:  127.0.0.1:43211
Python:  /home/user/.venv/bin/python
Build:   /home/user/chidori/chidori-worker/build

  protocol (wire format)                  PASS  (0s)
  test_pytorch.py (basic ops)             PASS  (3s)
  test_train.py (matmul 4096)             PASS  (20s)
  test_training_fp16.py (FP16)            PASS  (5s)
  test_training.py (FP32 train)           PASS  (15s)
  test_inference.py (TinyLlama)           PASS  (10s)

=== Results ===
6/6 passed, 0 failed
ALL TESTS PASSED
```

## Test Descriptions

| Test | What it validates | Exit 0 when |
|------|-------------------|-------------|
| `protocol` | Wire protocol serialization (no GPU) | All protocol round-trips succeed |
| `test_pytorch.py` | Tensor creation, math, matmul, CPU↔GPU transfer | Values match expected results |
| `test_train.py` | 2000x FP32 4096×4096 matmul benchmark | Sum is non-zero and reasonable |
| `test_training_fp16.py` | FP16 MLP training loop (50 steps) | All losses are finite |
| `test_training.py` | FP32 transformer training (50 steps) | All losses are finite |
| `test_inference.py` | TinyLlama 1.1B text generation (200 tokens) | Generates 100+ coherent tokens |

## Adding a New Test

1. Create `tests/test_yourtest.py`
2. Use `sys.exit(0)` on success, `sys.exit(1)` on failure
3. Print `PASS: test_yourtest` or `FAIL: test_yourtest` as the last line
4. Add it to `tests/run_all.sh` in the GPU tests section
