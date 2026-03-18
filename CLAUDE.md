# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chidori is a **GPU-over-IP system** — it enables transparent remote GPU access for CUDA workloads over TCP/IP. A client-side interception library (`libchidori.so`, loaded via `LD_PRELOAD`) intercepts CUDA API calls, serializes them over TCP, and a server-side worker dispatches them to real GPUs. Applications require zero code changes.

## Build Commands

```bash
# Build everything (requires Linux with CUDA SDK)
make all

# Build only the CUDA worker (server + client library)
make chidori-worker

# Build the Go control plane
make control-plane

# Run protocol unit tests (no GPU required)
cd chidori-worker && make test

# Run end-to-end tests (requires GPU)
cd chidori-worker && make test-e2e

# Clean all build artifacts
make clean
```

The chidori-worker Makefile compiles C with GCC and links against `libcuda`, `libcudart`, `libcublas`, and `libcudnn`. Build outputs go to `chidori-worker/build/`:
- `chidori-worker` — GPU host server binary
- `libchidori.so` — Client interception library

## Running Tests

**Protocol test** (no GPU, uses socketpair):
```bash
cd chidori-worker && make test
```

**Remote GPU tests** (two machines or terminals):
```bash
# Server side (machine with GPU)
cd chidori-worker/tests && ./start-server.sh <port>

# Client side (no GPU needed)
cd chidori-worker/tests && ./run-test.sh <server-ip> <port>
```

**Python/PyTorch tests** (`test_inference.py`, `test_training.py`, `test_pytorch.py`, etc.) run on the client side with `LD_PRELOAD=build/libchidori.so` pointing at a running server.

## Architecture

### Two main components:

**chidori-worker/** (C) — The core interception and dispatch layer:
- `src/client/chidori_client.c` — `LD_PRELOAD` library that intercepts ~280 CUDA symbols, serializes calls into wire protocol, sends over TCP
- `src/server/chidori_worker.c` — Server main loop, socket management, dispatch entry point
- `src/server/cuda_dispatch.c` — Giant switch dispatching function IDs to real CUDA calls (~3200 lines)
- `src/server/handle_map.h` — Maps wire handles ↔ real CUDA handles (append-only arrays, 8192 entry limit)
- `src/protocol/chidori_net.c` — TCP I/O primitives (blocking read/write)
- `include/chidori_protocol.h` — Wire protocol definitions (request/response headers, function ID ranges, flags)
- `codegen/` — Tool to auto-generate CUDA stubs from headers

**control-plane/** (Go) — Enterprise orchestration:
- `cmd/agent/` — Runs on GPU hosts, manages worker processes, reports GPU metrics via NVML
- `cmd/controller/` — Central scheduler with REST API, database backend (PostgreSQL/SQLite)
- `cmd/juicify/` — Client CLI launcher

### Wire Protocol

- **Request header** (16 bytes): `msg_len(4) | func_id(2) | flags(2) | seq_num(4) | payload_size(4)`
- **Response header** (12 bytes): `msg_len(4) | seq_num(4) | result(4)` + optional payload
- Native byte order (x86_64 only), no endian conversion
- Function ID ranges: `0x0001-0x00FF` (Driver API), `0x0100-0x01FF` (Runtime API), `0x0200-0x02FF` (cuBLAS), `0x0300-0x03FF` (cuDNN)

### Key Performance Patterns

- **Batch-read dispatch**: 256KB recv buffer, processes all complete requests in tight inner loop
- **TCP_CORK batching**: Packs multiple async responses into single TCP segment
- **DtoD zero-network**: Device-to-device copies send only 24 bytes over wire (data stays on GPU)
- **Async RPC path**: Compute-phase calls use `CHIDORI_FLAG_ASYNC` — no response sent, allowing 100s of queued operations

### Server Modes

1. **Standalone** (`--standalone <port>`) — Direct TCP listener, one client at a time, for testing
2. **Agent-mediated** (`--id <session_id> --ipc_write <fd> --ipc_read <fd>`) — Production mode, receives sockets from Go agent via `SCM_RIGHTS`

## CUDA API Coverage

Currently implements: ~30 Driver API functions, ~26 Runtime API functions, ~27 cuBLAS functions, ~47 cuDNN functions, plus ~150 symbol stubs for library compatibility. NCCL is not yet implemented.

## Adding a New CUDA Function

1. Add function ID constant to `include/chidori_protocol.h`
2. Add client-side serialization in `src/client/chidori_client.c`
3. Add server-side dispatch case in `src/server/cuda_dispatch.c`
4. If the function returns handles, add handle mapping in the dispatch case using `handle_map.h`



## Rules

### Testing — Mandatory After Every Change

1. Before running tests, ALWAYS delete the output file first:
   `rm -f .claude/test_output.txt`

2. Run the FULL test suite on the Lenovo loopback — never a subset:
   ```bash
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
   Before pushing code, ALWAYS: `git push origin main` then on Lenovo `git pull && make clean && make all CUDA_PATH=/opt/cuda && cd build && ln -sf libchidori.so libcuda.so.1`

3. The test command is NON-BLOCKING — wait for it to fully complete
   before reading the output. Do not proceed while tests are running.

4. Read `.claude/test_output.txt` in full after every run.

5. If ANY test fails:
   - Identify ALL failures in the file before touching any code
   - Fix ALL identified failures
   - Delete `.claude/test_output.txt`
   - Re-run the full suite from step 1
   - Repeat until the file shows zero failures

6. NEVER report success until you have read the output file and 
   confirmed zero failures. Do not assume — read the file.

7. NEVER fix one test and re-run — always fix all visible failures
   before the next run.

8. If the same failure repeats 3 times unchanged, STOP and report 
   the full contents of `.claude/test_output.txt` to me — do not loop forever.