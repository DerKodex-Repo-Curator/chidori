# 13. Test Suite

## Test Categories

### 1. Protocol Tests (No GPU Required)

**`test_protocol.c`** -- Wire protocol validation via Unix socketpair.

- 8 tests total:
  - `cuInit` -- initialization handshake
  - `cuDeviceGetCount` -- device count retrieval
  - `cuDeviceGetName` -- device name string
  - `cuMemAlloc` -- memory allocation
  - `cuMemGetInfo` -- memory info query
  - Header sizes -- request/response header structure validation
  - `cuMemFree` -- memory deallocation
  - Unknown function -- error handling for unrecognized function IDs
- Uses a mock server that returns hardcoded values.
- Runs entirely in-process via socketpair -- no network or GPU needed.

**`test_e2e_cuda.c`** -- Real CUDA calls through the wire protocol.

- 12 tests using real CUDA API calls (`cuInit`, `cuDriverGetVersion`, `cuDeviceGetCount`, etc.).
- Requires a physical GPU.
- Tests HtoD/DtoH data round-trips and DtoD zero-copy verification.

---

### 2. Test Scripts (Shell)

| Script | Purpose |
|--------|---------|
| `start-server.sh` | Builds and launches `chidori-worker` on the GPU machine |
| `run-test.sh` | Client-side runner for C-based e2e test |
| `run_all.sh` | Full suite (protocol + Python tests), loopback or remote |
| `run_training_test.sh` | Inline Python MLP training test (3-layer, FP32, MSE loss, 20 steps) |
| `run_suite.sh` | Production suite with SSH management and GPU monitoring |
| `run_crossarch.sh` | Cross-architecture tests via SSH tunnel to Vast.ai |
| `run_partial_offload.sh` | Partial offload mode testing |
| `run_replay_validation.sh` | A/B comparison (baseline vs replay, diff outputs) |
| `request.sh` | Deployment + test automation |
| `request_example.sh` | Example deployment + test automation |

---

### 3. Python Test Files

#### Smoke Tests

**`test_pytorch.py`** -- Basic PyTorch validation:
- Basic tensor operations
- Small matrix multiplication
- CPU-to-GPU and GPU-to-CPU transfers

#### Compute Benchmarks

**`test_train.py`** -- Pure matmul benchmark:
- 4096x4096 FP32 matrix multiplication
- 2000 iterations
- Measures throughput and latency

**`test_compute.py`** -- Sustained element-wise compute stress:
- Long-running element-wise operations
- Validates stability under sustained load

**`test_pytorch_heavy.py`** -- Comprehensive stress test:
- Large data transfers
- Matmul scaling (multiple sizes)
- Batched operations
- Memory pressure testing

#### Training Tests

**`test_training.py`** -- 6-layer transformer MLP:
- FP32 precision
- 50 training steps
- CrossEntropyLoss
- Validates gradient flow and loss convergence

**`test_training_fp16.py`** -- 6-layer MLP with mixed precision:
- FP16 (half precision)
- 50 training steps
- `bias=False` configuration
- Validates numerical stability

**`test_train.py`** -- Pure matmul benchmark (listed above under compute benchmarks)

#### Inference Tests

**`test_inference.py`** -- TinyLlama-1.1B token generation:
- Loads model over the network
- Generates tokens
- Validates output coherence

**`test_inference_flash.py`** -- Attention mechanism variants:
- SDPA (Scaled Dot Product Attention)
- `flash_attention_2`
- `eager` attention
- Compares outputs across variants

**`test_inference_cached.py`** -- KV cache strategies:
- Dynamic cache
- StaticCache
- Graph decode path

**`test_inference_graph.py`** -- CUDA graph capture and replay:
- Captures inference computation as a CUDA graph
- Replays the graph for multiple forward passes
- Validates output consistency

**`test_inference_multi.py`** -- Multi-prompt server-side graph decode:
- Server-side graph decode
- 5 concurrent prompts
- Validates isolation between prompts

**`test_llama8b.py`** -- Llama-3.1-8B inference:
- Larger model validation
- Graph decode path
- Memory and correctness validation

#### Partial Offload Tests

**`test_partial_offload_basic.py`** -- Device enumeration:
- Verifies 2 devices are visible
- Both devices are valid CUDA devices

**`test_partial_offload_compute.py`** -- Dual-device computation:
- Independent matmul on both local and remote devices
- Validates results match

**`test_partial_offload_transfer.py`** -- Cross-device data integrity:
- Transfer tensors between local and remote GPUs
- Verify data matches after transfer

**`test_partial_offload_split_model.py`** -- Split model with backward:
- 2-layer model split across devices
- Full backward pass
- Gradient correctness validation

**`test_partial_offload_device_map.py`** -- Automatic device mapping:
- TinyLlama-1.1B with `accelerate` `device_map="auto"`
- Automatic layer distribution across devices

**`test_partial_offload_full.py`** -- Comprehensive partial offload:
- 16+ tests covering all partial offload operations

**`test_partial_offload_advanced.py`** -- Advanced partial offload:
- 13 tests including large matmul, FP16, transformer, gradient accumulation, split model training

#### Replay Tests

**`test_replay_correctness.py`** -- Loss trajectory verification:
- Runs training twice (baseline and replay)
- Verifies identical loss trajectory

**`test_replay_gradients.py`** -- Per-layer gradient verification:
- Compares gradient norms at 10 decimal places
- Validates bit-exact reproducibility

**`test_replay_outputs.py`** -- Exact output verification:
- Compares logit values
- Compares weight hashes
- SHA-256 checksums for binary-exact validation

#### Split GPU Tests

**`test_split_inference.py`** -- TinyLlama split inference:
- Uses `accelerate.dispatch_model` for layer splitting
- Validates output matches single-GPU inference

**`test_split_graph.py`** -- Split inference with caching:
- Split inference with StaticCache for KV cache management
- Validates CUDA graph compatibility with split models

#### Suite Tests

**`test_suite_over_wire.py`** -- Full training suite over network:
- Matmul, FP32/FP16/mixed precision training
- Transformer training
- Split model training
- All over the wire protocol

**`test_gpu_stress.py`** -- Maximum stress test:
- 8192x8192 FP32 and FP16 matmul
- 4-layer training under load

---

## Common Environment Variables

| Variable | Purpose |
|----------|---------|
| `CHIDORI_DIRECT_CONNECT=1` | Bypass control plane, use direct TCP connection to server |
| `CHIDORI_CFG_OVERRIDE` | JSON config override: `{"id":"...", "servers":["host:port"]}` |
| `LD_PRELOAD=build/libcuda.so.1` | Load the interception library |
| `LD_LIBRARY_PATH=build` | Find proxy libraries (cuBLAS, NVML) in build directory |
| `CHIDORI_LOCAL_DEVICES=0` | Enable local GPU for partial offloading (or disable with empty value) |
| `CHIDORI_REPLAY=1` | Enable replay mode for deterministic re-execution |

---

## Determinism Requirements for Replay Tests

Replay tests require bit-exact reproducibility. The following settings are mandatory:

```python
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Disable Flash Attention and SDPA (non-deterministic)
# Force cuBLAS for all attention computations

CUBLAS_WORKSPACE_CONFIG = ":4096:8"

torch.use_deterministic_algorithms(True, warn_only=True)
```

Additional requirements:
- **Separate random generators** for training data and probe input (prevents seed interference).
- **Fixed input data** -- training data is generated deterministically from the seed.
- **No data augmentation** or other stochastic preprocessing.
- **Fixed batch ordering** -- no shuffling between runs.
