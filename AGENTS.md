# AGENTS.md

Guide for AI agents working in the Chidori codebase.

## Project Overview

Chidori is a **GPU-over-IP system** that enables transparent remote GPU access for CUDA workloads over TCP/IP. A client-side interception library (`libchidori.so`, loaded via `LD_PRELOAD`) intercepts CUDA API calls, serializes them over TCP, and a server-side worker dispatches them to real GPUs. Applications require zero code changes.

**Key insight**: CUDA applications never touch GPU memory directly — everything goes through API calls that return opaque handles. A `CUdeviceptr` is just a 64-bit number the CPU code never dereferences.

## Project Structure

```
chidori/
├── chidori-worker/          # CUDA interception layer (C)
│   ├── src/
│   │   ├── server/          # chidori-worker binary (runs on GPU host)
│   │   │   ├── chidori_worker.c   # Server main loop, socket mgmt
│   │   │   ├── cuda_dispatch.c    # ~3200-line switch dispatching RPCs to CUDA
│   │   │   └── handle_map.h       # Wire handle ↔ real CUDA handle mapping
│   │   ├── client/          # libchidori.so (LD_PRELOAD on client)
│   │   │   └── chidori_client.c   # ~5000 lines of CUDA API interception
│   │   ├── protocol/        # Wire protocol I/O
│   │   │   └── chidori_net.c      # TCP read/write primitives
│   │   └── common/
│   ├── include/
│   │   └── chidori_protocol.h     # Wire protocol definitions, func IDs
│   ├── codegen/             # Auto-generates CUDA stubs from headers
│   │   └── generate_stubs.py
│   └── tests/               # Protocol + e2e + PyTorch tests
│
├── control-plane/           # Session orchestration (Go)
│   ├── cmd/
│   │   ├── agent/           # Runs on GPU hosts, manages worker processes
│   │   ├── controller/      # Central scheduler with REST API
│   │   └── juicify/         # Client CLI launcher
│   └── pkg/                 # Shared Go packages
│
└── Makefile                 # Top-level build orchestration
```

## Build Commands

**Linux only.** Requires CUDA SDK for server build.

```bash
# Build everything
make all

# Build only the CUDA worker (server + client library)
make chidori-worker

# Build only the Go control plane
make control-plane

# Run protocol unit tests (no GPU required)
cd chidori-worker && make test

# Run end-to-end tests (requires GPU)
cd chidori-worker && make test-e2e

# Clean all build artifacts
make clean
```

### Build Outputs

Located in `chidori-worker/build/`:
- `chidori-worker` — GPU host server binary
- `libchidori.so` — Client interception library

### CUDA Path

Default: `/usr/local/cuda`. Override with:
```bash
make all CUDA_PATH=/opt/cuda
```

## Running Tests

### Protocol Tests (No GPU)

```bash
cd chidori-worker && make test
```

Uses socketpair, no real GPU needed.

### Full Test Suite (Requires GPU)

The full test suite runs on a remote GPU machine (Lenovo). Tests are executed via SSH:

```bash
rm -f .claude/test_output.txt

ssh fabricio@lenovo.local 'cd ~/chidori/chidori-worker && \
  kill $(pgrep chidori-worker) 2>/dev/null; sleep 1; \
  nohup build/chidori-worker --standalone 43211 > /tmp/server.log 2>&1 & sleep 1; \
  ENV="CHIDORI_DIRECT_CONNECT=1 CHIDORI_CFG_OVERRIDE={\"id\":\"test\",\"servers\":[\"127.0.0.1:43211\"]} LD_PRELOAD=build/libcuda.so.1 LD_LIBRARY_PATH=build" && \
  echo "=== run_training_test.sh ===" && ./tests/run_training_test.sh && \
  echo "=== test_train.py ===" && eval $ENV ~/chidori/.venv/bin/python tests/test_train.py && \
  echo "=== test_training_fp16.py ===" && eval $ENV ~/chidori/.venv/bin/python tests/test_training_fp16.py && \
  echo "=== test_training.py ===" && eval $ENV ~/chidori/.venv/bin/python tests/test_training.py && \
  echo "=== test_inference.py ===" && eval $ENV ~/chidori/.venv/bin/python tests/test_inference.py && \
  echo "=== ALL TESTS PASSED ==="' 2>&1 | tee .claude/test_output.txt
```

### Test Rules

1. **Always delete output file first**: `rm -f .claude/test_output.txt`
2. **Run the FULL test suite** — never a subset
3. **Wait for completion** — the SSH command is non-blocking
4. **Read the output file in full** after every run
5. **Fix ALL failures before re-running** — never fix one test and re-run
6. **Never assume success** — read the output file and confirm zero failures
7. **After 3 identical failures**, stop and report the full output

### Before Pushing Code

```bash
git push origin main
# Then on Lenovo:
git pull && make clean && make all CUDA_PATH=/opt/cuda && cd build && ln -sf libchidori.so libcuda.so.1
```

## Wire Protocol

### Request Header (16 bytes)

```
Offset  Size  Field
0       4     msg_len        Total message length including header
4       2     func_id        CUDA function identifier
6       2     flags          CHIDORI_FLAG_* bitmask
8       4     seq_num        Sequence number for matching
12      4     payload_size   Size of payload following header
```

### Response Header (12 bytes)

```
Offset  Size  Field
0       4     msg_len        Total message length including header
4       4     seq_num        Matching request sequence number
8       4     result         CUresult / cudaError_t return value
```

### Message Flags

| Flag | Value | Meaning |
|------|-------|---------|
| `CHIDORI_FLAG_SYNC` | 0x0001 | Client blocks for response |
| `CHIDORI_FLAG_ASYNC` | 0x0002 | No immediate response needed |
| `CHIDORI_FLAG_BATCH_MORE` | 0x0004 | More messages follow |
| `CHIDORI_FLAG_ERROR` | 0x0008 | Response indicates error |

### Function ID Ranges

| Range | Category |
|-------|----------|
| 0x0001-0x00FF | Driver API |
| 0x0100-0x01FF | Runtime API |
| 0x0200-0x02FF | cuBLAS |
| 0x0300-0x03FF | cuDNN |

All structs use `__attribute__((packed))`. Native byte order (x86_64 only), no endian conversion.

## Server Modes

1. **Standalone** (`--standalone <port>`) — Direct TCP listener, one client at a time, for testing
2. **Agent-mediated** (`--id <session_id> --ipc_write <fd> --ipc_read <fd>`) — Production mode, receives sockets from Go agent via `SCM_RIGHTS`

## Key Performance Patterns

- **Batch-read dispatch**: 256KB recv buffer, processes all complete requests in tight inner loop
- **TCP_CORK batching**: Client-side; packs multiple async writes into single TCP segment
- **DtoD zero-network**: Device-to-device copies send only 24 bytes over wire (data stays on GPU)
- **Async RPC path**: Compute-phase calls use `CHIDORI_FLAG_ASYNC` — no response sent

### Async vs Sync Operations

**Async (no round-trip)**: `cuMemcpyHtoD`, `cuMemcpyDtoD`, `cuMemFree`, `cuLaunchKernel`, all GEMM operations, `cublasSetStream`, etc.

**Sync (blocks for response)**: `cuInit`, `cuDeviceGet*`, `cuCtxCreate`, `cuMemAlloc`, `cuMemcpyDtoH`, `cuStreamSynchronize`, `cudaDeviceSynchronize`, etc.

## Adding a New CUDA Function

1. Add function ID constant to `include/chidori_protocol.h`
2. Add client-side serialization in `src/client/chidori_client.c`
3. Add server-side dispatch case in `src/server/cuda_dispatch.c`
4. If the function returns handles, add handle mapping using `handle_map.h`

## Handle Mapping

Server-side maps wire handles (uint64_t) to real CUDA handles. Located in `src/server/handle_map.h`:

- `hmap_init()` — Initialize map
- `hmap_put()` — O(1) append-only insert
- `hmap_get()` — O(n) linear scan lookup
- Max 8192 entries (`HANDLE_MAP_MAX`)

Separate maps exist for each CUDA object type: modules, functions, streams, events, cublas handles, cudnn handles, descriptors, etc.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `CHIDORI_CFG_OVERRIDE` | JSON config with session ID and server addresses. Required. Format: `{"id":"session-uuid","servers":["host:port"]}` |
| `CHIDORI_DIRECT_CONNECT` | If `"1"`, bypass HTTP/agent protocol and connect directly to worker TCP port |
| `CHIDORI_TRACE` | If `"1"`, logs every RPC call with timestamp, func_id, payload size |
| `CHIDORI_VERBOSE` | If `"1"`, logs per-RPC details |
| `CHIDORI_SERVER_VERBOSE` | If `"1"`, server logs every dispatched request |

## Client-Side Caching

The client library caches frequently-accessed data to avoid sync RPCs:

- **Device attribute cache** (`g_attr_cache`) — PyTorch queries 100+ attributes during init
- **Context cache** (`g_cached_ctx`) — Avoids `cuCtxGetCurrent` RPCs
- **Device count cache** (`g_device_count_cache`)
- **Current device cache** (`g_current_device`) — `cudaGetDevice()` returns without RPC
- **Version caches** — Driver/runtime/cuBLAS/cuDNN versions cached after first query

## Code Style (C)

- GCC with `-Wall -Wextra -O2 -fPIC`
- 4-space indentation (observe existing files)
- `__attribute__((packed))` for all wire protocol structs
- Handle maps use static inline functions
- Dispatch is a giant switch statement (~3200 lines in `cuda_dispatch.c`)

## Code Style (Go)

- Go 1.20
- Package structure follows standard Go conventions
- Uses Gorilla Mux for HTTP routing
- GORM for database access
- NVML for GPU metrics

## Known Issues / Blockers

1. **cublasLt Heuristic**: `cublasLtMatmulAlgoGetHeuristic` returns 0 algorithms. PyTorch 2.6 FP32 batched GEMM with bias has no fallback.
2. **Fat Binary Size**: `cuModuleLoadData` uses hardcoded 64KB limit. Large kernels get truncated.
3. **Flash/SDPA Attention**: Uses CUTLASS kernels with custom dispatch. Workaround: `enable_flash_sdp(False)`

## CUDA API Coverage

- ~30 Driver API functions
- ~26 Runtime API functions
- ~27 cuBLAS functions
- ~47 cuDNN functions
- ~150 symbol stubs for library compatibility
- NCCL: Not yet implemented

## Key Files Reference

| File | Purpose |
|------|---------|
| `chidori-worker/include/chidori_protocol.h` | Wire protocol definitions |
| `chidori-worker/src/protocol/chidori_net.c` | TCP I/O primitives |
| `chidori-worker/src/server/chidori_worker.c` | Server main(), dispatch loop |
| `chidori-worker/src/server/cuda_dispatch.c` | RPC → CUDA dispatch (3200+ lines) |
| `chidori-worker/src/server/handle_map.h` | Handle mapping utilities |
| `chidori-worker/src/client/chidori_client.c` | Client interception (5000+ lines) |
| `chidori-worker/ARCHITECTURE.md` | Detailed architecture trace |
| `CLAUDE.md` | Claude-specific instructions and test procedures |
