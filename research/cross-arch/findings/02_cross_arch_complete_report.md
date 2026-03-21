# Cross-Architecture GPU-over-IP: Complete Technical Report

**Date**: 2026-03-21
**Project**: Chidori GPU-over-IP
**Result**: CROSS-ARCHITECTURE CONFIRMED WORKING
**Test**: Lenovo RTX 4090 (sm_89) → Blackwell RTX PRO 6000 (sm_120) over TCP

---

## Executive Summary

Chidori can now execute CUDA kernels on a GPU with a different architecture than the one the code was compiled for. A machine with an RTX 4090 (sm_89) runs PyTorch, and the actual GPU computation happens on a remote Blackwell RTX PRO 6000 (sm_120) over the network. The application requires zero code changes.

The solution required: (1) building PyTorch from source with nvcc 13.1 to get zstd-compressed fat binaries instead of NVIDIA's proprietary `a2` format, and (2) adding zstd decompression to the Chidori server to extract cubins that the CUDA Driver API can load.

---

## The Problem

CUDA fat binaries contain compiled GPU code (cubins) for specific GPU architectures. When a client compiled for sm_89 sends its fat binary to a server with sm_120, the server needs to either:
- Find an sm_120 cubin in the fat binary, or
- Find PTX (architecture-independent intermediate code) and JIT-compile it

The CUDA Driver API (`cuModuleLoadData`) cannot parse the `0xba55ed50` fat binary wrapper format — it expects raw cubins or PTX text. And the cubins inside pre-built PyTorch wheels are in a proprietary format that no public API can decode.

---

## What We Discovered

### Two Distinct Fat Binary Formats

NVIDIA uses different cubin encoding depending on the nvcc version:

| nvcc Version | Encoding | First Bytes | Decodable? |
|---|---|---|---|
| 12.8 (cu128 pip wheel) | Proprietary `a2` | `a2 7f 45 4c 46...` | NO — only `__cudaRegisterFatBinary` can decode |
| 13.1 (source-built) | Standard zstd | `28 b5 2f fd...` | YES — standard zstd decompression |

The `a2` format uses an ELF-like header with non-standard fields (`e_type=0x8000`, `e_machine=0x100`) that no public CUDA API or tool (cuobjdump, cuModuleLoadData, cuLibraryLoadData) can process.

The zstd format uses standard Zstandard compression (open-source, by Meta). Any zstd library can decompress it to get raw ELF cubins or PTX text.

### Fat Binary Entry Format

```
Offset  Size  Field
0       2     entry_type (1=PTX, 2=ELF/cubin)
2       2     sub_header_flags
4       4     entry_header_size (64 or 80 or 96 bytes)
8       8     uncompressed_data_size
16      4     compressed_data_size
20      4     flags (bit 6 = 0x40 = compressed)
24      4     reserved
28      4     sm_version (70, 75, 80, 86, 89, 90, 100, 120)
[hdr]   var   data payload
```

Entries are 8-byte aligned. Position advances by `uncompressed_data_size` (not compressed), because compressed entries still occupy the uncompressed footprint in the fat binary layout.

### Compression Detection

- Check `flags & 0x40` at entry offset +20
- If set, check first 4 bytes of data:
  - `28 b5 2f fd` = zstd → decompress with standard zstd
  - `a2 7f 45 4c` = NVIDIA proprietary → cannot decode with public APIs

---

## Everything We Tried (11 Approaches)

### Approaches That Failed

1. **cuModuleLoadData(fatbin)** — Error 209. Can't parse 0xba55ed50 wrapper.
2. **cuModuleLoadFatBinary(fatbin)** — Error 209. Deprecated, can't decompress.
3. **cuModuleLoadDataEx with JIT target** — Error 209. Same underlying issue.
4. **cuLibraryLoadData (CUDA 12+)** — Loads container, 0 kernels available. Can't decompress entries.
5. **__cudaRegisterFatBinary + dummy hostFun pointers** — Registration OK, cudaGetFuncBySymbol returns 209.
6. **__cudaRegisterFatBinary + mmap'd executable stubs** — Same result. Real executable memory doesn't help.
7. **__cudaRegisterFatBinary + real C function addresses** — Works with uncompressed PTX fatbins, fails with compressed.
8. **Extract ELF from fat binary, strip a2 prefix** — Error 200. The a2-prefixed data is NOT a valid ELF.
9. **Patch ELF OSABI from 0x33 to 0x41** — Error 200. e_type/e_machine are also non-standard.
10. **Install cu130 PyTorch (native sm_120 SASS)** — Segfaults due to cudaDeviceProp ABI mismatch (1032 vs 1008 bytes).
11. **dlmopen real NVIDIA driver in new namespace** — Segfaults from conflicting CUDA state.

### The Approach That Worked

**Build PyTorch from source with nvcc 13.1 + zstd decompression in the server.**

1. Clone PyTorch v2.10.0 on the Blackwell (256-core, fast internet)
2. Build with `TORCH_CUDA_ARCH_LIST="8.9+PTX" USE_MPI=0 USE_NCCL=0 USE_DISTRIBUTED=0`
3. Install Python 3.14 wheel on the Lenovo client
4. Server parses fat binary entries, finds matching architecture or PTX
5. Decompresses zstd-encoded entry with standard `ZSTD_decompress()`
6. Loads raw cubin/PTX via `cuModuleLoadData()` (standard Driver API)
7. Gets CUfunction via `cuModuleGetFunction()` — param info, kernel launches all work

---

## Implementation Details

### Server Changes (`cuda_dispatch.c`)

Added `#include <zstd.h>` and `-lzstd` to link flags.

In `CHIDORI_CUDA_REGISTER_FAT_BINARY` handler:
- **Fast path**: `cuModuleLoadData(payload)` — works for same-arch (uncompressed cubin matches server GPU)
- **Cross-arch fallback**: Parse 0xba55ed50 fat binary entries, find best match:
  1. Prefer ELF entry matching server's SM version
  2. Fallback to PTX entry (any version, JIT-compiled)
- If entry has zstd magic (`28 b5 2f fd`): decompress with `ZSTD_decompress()`
- Pass raw cubin/PTX to `cuModuleLoadData()` — standard Driver API from here

### Client Changes (`chidori_client.c`)

- Added `__cudaLaunchKernel` interception (source-built PyTorch uses double-underscore variant)
- Fixed `cudaDeviceProp` memset size: 1008 bytes for CUDA 13.x (was 1032 for 12.x)

### Build Changes (`Makefile`)

- Added `-lzstd` to `LDFLAGS_SERVER`

### New Dependencies

- `libzstd` (BSD license, ~100KB, available on all Linux distros)

---

## What Still Needs Work

### Short Term

1. **The `a2` proprietary format** — Pre-built pip wheels (cu124, cu126, cu128) use this format. Our zstd path doesn't help them. Customers using stock pip-installed PyTorch on CUDA 12.x need either:
   - A source-built PyTorch wheel (we can distribute this)
   - NVIDIA to document/open-source the `a2` format
   - Matching GPU architectures (no cross-arch needed)

2. **Full test suite on cross-arch** — Only `x + 1` tested. Need to run all 5 tests (matmul, FP16 training, FP32 training, inference).

3. **Same-arch regression** — Loopback 5/5 PASS confirmed. Need to verify same-arch remote still works.

### Long Term

1. **Reverse-engineer `a2` format** — ELF-like with `e_type=0x8000`, `e_machine=0x100`, OSABI `0x33`. The research directory has tools and extracted samples for further analysis.

2. **Ship pre-built wheels** — Build and distribute `chidori-torch` wheels with nvcc 13.1 (zstd format) for common Python versions.

3. **Support multiple CUDA versions** — Detect fat binary format at runtime and choose the right decompression path.

---

## Proof

```
$ ssh fabricio@lenovo.local '... LD_PRELOAD=build/libcuda.so.1 ... python -c "
import torch; torch._C._cuda_init()
print(\"GPU:\", torch.cuda.get_device_name(0))
x = torch.tensor([1.0, 2.0, 3.0]).cuda()
y = x + 1; torch.cuda.synchronize()
print(\"x+1 =\", y.cpu().tolist())
"'

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
x+1 = [2.0, 3.0, 4.0]
PASS
```

Client: Lenovo RTX 4090 Laptop (sm_89), PyTorch built with nvcc 13.1
Server: Blackwell RTX PRO 6000 (sm_120), CUDA 13.1
Network: SSH tunnel over internet
Fat binary path: sm_89 PTX → zstd decompress → cuModuleLoadData JIT → sm_120 execution

---

## Files Changed

| File | Change |
|---|---|
| `chidori-worker/src/server/cuda_dispatch.c` | zstd decompression, fatbin parser, cross-arch module loading |
| `chidori-worker/src/client/chidori_client.c` | `__cudaLaunchKernel` interception, cudaDeviceProp size fix |
| `chidori-worker/Makefile` | `-lzstd` link flag |
| `research/cross-arch/` | Research tools, findings, extracted samples |

## Branch

`feat/cross-arch-clean` — based on `main`, no regressions on loopback (5/5 PASS).

---

## Timeline

- **Day 1-2**: Tried Runtime API approaches (attempts 1-8). All failed with compressed cubins.
- **Day 3 morning**: Discovered zstd magic `28 b5 2f fd` in nvcc 13.1 fatbins. Confirmed standard zstd decompression works.
- **Day 3 afternoon**: Found that cu128 pre-built wheels use `a2` format (NOT zstd). Built PyTorch from source on Blackwell (256 cores, ~25 min per build).
- **Day 3 evening**: Fixed `__cudaLaunchKernel` interception and `cudaDeviceProp` size. Cross-arch confirmed working: `x+1 = [2.0, 3.0, 4.0]` on Blackwell via Lenovo.
