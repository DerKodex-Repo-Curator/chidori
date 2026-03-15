# chidori-worker

Open-source CUDA-over-IP interception layer for the [Juice Labs](https://github.com/Juice-Labs/Juice-Labs) GPU remoting system.

This project replaces the proprietary `Renderer_Win` binary with an open-source implementation, enabling transparent remote GPU access for CUDA compute workloads over TCP/IP.

## Architecture

```
CLIENT (no GPU)                         SERVER (has GPU)
===============                         ================

  User App (unmodified)                  chidori-worker
       |                                      |
  [LD_PRELOAD]                           [Real CUDA API]
       |                                      |
  libchidori.so        --- TCP --->   dispatch loop
```

**libchidori.so** — Client library loaded via `LD_PRELOAD`. Intercepts all CUDA Driver API calls, serializes them, and sends them over TCP to the GPU worker.

**chidori-worker** — Server binary running on the GPU host. Receives CUDA API calls over the wire protocol, executes them on the real GPU, and returns results.

## Building

```bash
make all        # Build server + client
make test       # Run protocol tests (no GPU needed)
make clean      # Clean build artifacts
```

For real CUDA support, set `CUDA_PATH`:
```bash
CUDA_PATH=/usr/local/cuda make all
```

## Codegen

The codegen tool auto-generates client/server stubs from CUDA headers:

```bash
python3 codegen/generate_stubs.py --cuda-path /usr/local/cuda/include --output-dir src/generated/
```

## Integration with Juice Labs

This plugs into the existing Juice Labs control plane:
1. The Juice **Agent** spawns `chidori-worker` instead of `Renderer_Win`
2. The Agent forwards client TCP sockets via IPC (Unix SCM_RIGHTS)
3. **juicify** sets `LD_PRELOAD=libchidori.so` on client applications

## Status

- [x] Wire protocol (binary, 16B request header / 12B response header)
- [x] Network I/O primitives (reliable read/write)
- [x] Protocol unit tests (8/8 passing)
- [x] GPU worker stub (accepts IPC sockets, dispatches CUDA calls)
- [x] Client interception library (LD_PRELOAD, 30+ CUDA Driver API functions)
- [x] Codegen tool (parses cuda.h, generates stubs)
- [ ] Real CUDA API calls on server (requires CUDA toolkit)
- [ ] CUDA Runtime API interception
- [ ] cuBLAS / cuDNN interception
- [ ] Batching and pipelining optimizations
- [ ] Integration testing with PyTorch

## License

MIT
