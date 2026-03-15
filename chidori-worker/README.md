# chidori-worker

CUDA-over-IP interception layer — the core of [Chidori](https://github.com/fpolica91/chidori).

Replaces the proprietary `Renderer_Win` binary from Juice Labs with an open-source implementation. Enables transparent remote GPU access for CUDA compute workloads over TCP/IP.

## Components

**libchidori.so** — Client library loaded via `LD_PRELOAD`. Intercepts ~280 CUDA functions (Driver API, Runtime API, cuBLAS, cuDNN), serializes them, and sends them over TCP.

**chidori-worker** — Server binary running on the GPU host. Receives CUDA API calls, executes them on the real GPU, returns results.

## Building

```bash
make all        # Build server + client
make test       # Run protocol tests (no GPU needed)
make clean      # Clean build artifacts
```

## Status

- [x] Wire protocol (16B request / 12B response)
- [x] CUDA Driver API (~30 functions, real dispatch)
- [x] CUDA Runtime API (~26 functions, real dispatch)
- [x] cuBLAS (~27 functions, real dispatch)
- [x] cuDNN (~47 functions, real dispatch)
- [x] ~150 additional symbol stubs (PyTorch compatibility)
- [x] Multi-GPU support (8x A100 tested)
- [x] Kernel parameter serialization
- [x] PyTorch tensor ops working
- [x] 50/50 tests passing
- [x] Codegen tool (parses cuda.h)

## License

MIT
