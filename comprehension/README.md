# Chidori Codebase Comprehension Documentation

This directory contains a complete, line-by-line analysis of the Chidori GPU-over-IP system. Every source file has been read and documented.

## Reading Order

Start here, then read in this order:

1. **[01-system-overview.md](01-system-overview.md)** -- High-level architecture, how all pieces fit together
2. **[02-wire-protocol.md](02-wire-protocol.md)** -- The binary protocol that everything else depends on
3. **[03-client-interception.md](03-client-interception.md)** -- How LD_PRELOAD works, connection setup, symbol interception
4. **[04-proxy-libraries.md](04-proxy-libraries.md)** -- cuBLAS/NVML proxy libs and versioned symbol interception
5. **[05-handle-mapping.md](05-handle-mapping.md)** -- Wire handle to real CUDA handle translation
6. **[06-server-dispatch.md](06-server-dispatch.md)** -- The batch-read dispatch loop and every dispatch case
7. **[07-replay-system.md](07-replay-system.md)** -- Call sequence recording and replay for training optimization
8. **[08-split-gpu-and-graph-decode.md](08-split-gpu-and-graph-decode.md)** -- Pipeline-parallel inference and server-side decode
9. **[09-fat-binary-handling.md](09-fat-binary-handling.md)** -- Cross-architecture GPU code loading (the most complex subsystem)
10. **[10-control-plane.md](10-control-plane.md)** -- Go orchestration: agent, controller, juicify CLI
11. **[11-partial-offloading.md](11-partial-offloading.md)** -- Using local + remote GPU simultaneously
12. **[12-build-system.md](12-build-system.md)** -- Makefiles, build targets, codegen
13. **[13-test-suite.md](13-test-suite.md)** -- All test scripts and Python test files catalogued

## Quick Reference

| File | Lines | What it does |
|------|-------|-------------|
| `src/client/chidori_client.c` | ~9510 | LD_PRELOAD library intercepting ~280 CUDA symbols |
| `src/server/cuda_dispatch.c` | ~4618 | Giant switch dispatching 100+ function IDs to real CUDA |
| `src/server/chidori_worker.c` | ~391 | Server main loop and TCP connection management |
| `src/server/replay.c` | ~400 | Call sequence recording and replay |
| `src/server/handle_map.h` | ~55 | Append-only key-value map for handle translation |
| `src/server/replay.h` | ~120 | Replay system data structures and API |
| `src/protocol/chidori_net.c` | ~90 | Blocking TCP I/O primitives |
| `include/chidori_protocol.h` | ~350 | Wire protocol definitions, all function IDs |
| `src/client/chidori_nvml.c` | ~124 | NVML interception shim |
| `src/client/chidori_client.map` | ~76 | GNU linker version script for symbol interception |
| `codegen/generate_stubs.py` | ~323 | Stub code generator (not functional) |
