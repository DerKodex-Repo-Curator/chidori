# 12. Build System

## Root Makefile

The root `Makefile` is a simple delegation layer to sub-projects. It does not contain any build logic itself.

| Target | Delegation | Description |
|--------|------------|-------------|
| `all` | `$(MAKE) -C chidori-worker all` | Build everything |
| `chidori-worker` | `$(MAKE) -C chidori-worker all` | Build worker + client |
| `control-plane` | `cd control-plane && go build ./...` | Build Go binaries |
| `test` | `$(MAKE) -C chidori-worker test` | Protocol tests |
| `test-e2e` | `$(MAKE) -C chidori-worker test-e2e` | End-to-end tests |
| `clean` | `$(MAKE) -C chidori-worker clean` | Clean build artifacts |

Notable absences:
- No `clean` target for the control-plane.
- No Docker or deployment targets at the root level.

---

## chidori-worker Makefile

### Variables

```makefile
CC ?= gcc
CFLAGS = -D_GNU_SOURCE -Wall -Wextra -O2 -fPIC -I include
CUDA_PATH ?= /usr/local/cuda            # Overridable: CUDA_PATH=/opt/cuda
CUDA_LDFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcublas -lcublasLt -lcudnn
```

### Source Files

| Component | Files |
|-----------|-------|
| Protocol | `src/protocol/chidori_net.c` |
| Server | `src/server/chidori_worker.c`, `src/server/cuda_dispatch.c`, `src/server/replay.c` |
| Client | `src/client/chidori_client.c` |

### Build Targets

| Target | Output | Description |
|--------|--------|-------------|
| `all` | server + client | Default target |
| `server` | `build/chidori-worker` | Links against real CUDA libs, libzstd, pthread |
| `client` | `build/libchidori.so` + 4 proxy libs + NVML proxy | Client interception library |
| `test` | `build/test_protocol` | Protocol tests, no GPU required |
| `test-e2e` | `build/test_e2e_cuda` | Real CUDA end-to-end tests, requires physical GPU |
| `test-all` | all + run `run_all.sh` | Full test suite |
| `clean` | `rm -rf build/` | Remove all build artifacts |

### Client Build Details

The `client` target produces multiple shared libraries:

**`libchidori.so`** -- Main interception library:
```
-ldl -shared -Wl,-soname,libchidori.so -Wl,--version-script=chidori_client.map
```

**cuBLAS Proxy Libraries** -- Same source compiled 4 times with different sonames:
- `libcublas.so.12` -- soname `libcublas.so.12`
- `libcublas.so.13` -- soname `libcublas.so.13`
- `libcublasLt.so.12` -- soname `libcublasLt.so.12`
- `libcublasLt.so.13` -- soname `libcublasLt.so.13`

**NVML Proxy Library:**
- `libnvidia-ml.so.1` -- Built from `chidori_nvml.c`, soname `libnvidia-ml.so.1`

### cuBLAS Proxy Pattern

All four cuBLAS/cublasLt proxy libraries are compiled from the same `chidori_client.c` source with different `-Wl,-soname` values. When `LD_LIBRARY_PATH=build`, the dynamic linker finds these proxy libraries instead of the real cuBLAS. The proxy then routes cuBLAS calls over the network to the remote server.

This is necessary because cuBLAS internally calls CUDA driver functions. Without the proxy, those internal calls would go to the intercepted (network-routed) versions instead of the real driver when running on the local GPU.

---

## Codegen (`generate_stubs.py`)

A Python 3 script in `codegen/` that auto-generates CUDA stubs from headers.

**How it works:**
- Parses CUDA headers via regex.
- Generates client stubs and server dispatch cases.

**Current state:**
- **NOT integrated into the build system** -- the Makefile never calls this script.
- Generated code is **SKELETON ONLY** -- contains `TODO` placeholders with no actual serialization logic.
- `TYPE_SIZES` dictionary is defined but never used.
- References stale filename `"juice_client.c"` (legacy name).
- Only parses `cuda.h` (Driver API) -- does not parse runtime, cuBLAS, or cuDNN headers.
- Report generation code exists but has never been run (no report file committed).

---

## Deployment (Inherited from Juice Labs)

### Docker

Docker configurations exist in `control-plane/docker/`:
- `agent` -- GPU host agent container
- `client` -- Client CLI launcher container
- `controller` -- Central scheduler container
- `server` -- GPU worker server container
- `nvidia-video-codec` -- Video codec container

### Helm Charts

Kubernetes Helm charts in `control-plane/charts/`:
- `agent` -- GPU host agent deployment
- `controller` -- Central scheduler deployment

### CI/CD

GitHub Actions workflows:
- `charts-release.yml` -- Helm chart release
- `charts-test.yml` -- Helm chart testing
- `docker-image.yml` -- Docker image build
- `controller-docker-image.yml` -- Controller Docker image build

### Notable Gaps

- No Docker build target for `chidori-worker` itself (the C component).
- No root-level Docker or deployment orchestration.
