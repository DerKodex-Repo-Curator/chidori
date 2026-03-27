# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A research workspace for reverse-engineering CUDA internals that inform the Chidori GPU-over-IP project. Contains proof-of-concept programs, disassembly analysis, and binary format documentation. Not a production codebase — no tests to run before changes.

## Build & Run

### cudart-internals (CUDA Runtime API reverse-engineering)

```bash
cd cudart-internals
make              # Build test programs and test fatbin (sm_89)
make test         # Run all proof tests (requires GPU + CUDA SDK)
make clean        # Remove build artifacts
```

To build/run `runtime_probe_v2` (separate, not in Makefile `all`):
```bash
cd cudart-internals
gcc -Wall -O0 -g -I/opt/cuda/include -o runtime_probe_v2 runtime_probe_v2.c \
  -L/opt/cuda/targets/x86_64-linux/lib -lcuda -lcudart -ldl -lpthread \
  -Wl,-rpath,/opt/cuda/targets/x86_64-linux/lib
./runtime_probe_v2 simple_sm86.fatbin simple_add
```

Test programs link against `libcuda` and `libcudart`. Requires `CUDA_PATH` (default `/opt/cuda`).

### cross-arch (Fat binary compression research)

No build system — contains documentation and analysis scripts only.

## Architecture

### cudart-internals/

Investigates why `cudaGetFuncBySymbol` fails in a server context by probing libcudart.so internal state. Research spans two CUDA versions with different behaviors:

- **FINDINGS.md** — CUDA 13.1.80 (RTX 4090 Laptop, sm_89): initialization ordering doesn't matter, the 0x540 field is not a gate, `cudaGetFuncBySymbol` succeeds on sm_89 fatbin
- **FINDINGS_V2.md** — CUDA 12.8.90 (RTX A6000, sm_86): `cudaGetFuncBySymbol` returns error 98, `__cudaGetKernel` succeeds as alternative, two resolution paths (vtable vs direct)
- **DEEP_ANALYSIS.md** — Full disassembly of `cudaGetFuncBySymbol`, `__cudaRegisterFatBinary`, `__cudaRegisterFunction`, runtime state layout (CUDA 13.1)
- **EXTRACTED_CODE.md** — Reconstructed pseudocode from disassembly
- **STRING_TABLE.md** — Internal string table from libcudart.so
- **runtime_probe_v2.c** — Comprehensive state dumper for CUDA 12.8, probes mod_mgmt vtables, lifecycle fields, and registration pipeline at every stage
- **deep_trace.c** — Tracer that logs every internal call during fat binary registration and kernel resolution
- **probe_runtime_state.c** — Reads internal runtime state fields at different lifecycle stages
- **test_isolated_*.c** — Minimal proofs for initialization ordering (context-first vs register-first)
- **simple.cu** — Test kernel used to build fat binaries for experiments

### cross-arch/

Documents the discovery that NVIDIA's CUDA 13.x fat binary compression is standard **zstd** (magic `28 b5 2f fd`), not proprietary. This is the breakthrough that enables cross-architecture GPU-over-IP.

- **README.md** — Fat binary entry format with compression detection and zstd verification code
- **findings/01_cross_arch_breakthrough.md** — Full chronicle of 10 failed attempts and the zstd discovery
- **findings/02_cross_arch_complete_report.md** — Complete technical report

## Key Technical Details

### Fat Binary Entry Format

```
Offset  Size  Field
0       2     entry_type (1=PTX, 2=ELF/cubin)
2       2     sub_header_flags
4       4     entry_header_size
8       8     uncompressed_data_size
16      4     compressed_data_size (0 = uncompressed)
20      4     flags (bit 6 = 0x40 = COMPRESSED)
24      4     unknown/reserved
28      4     sm_version (89, 120, etc.)
[hdr]   ...   data (zstd if flags & 0x40, raw otherwise)
```

### Compression Types

- **zstd** (`28 b5 2f fd` magic): nvcc 13.1, standard algorithm, decompressible with `libzstd`
- **a2** (`0xa2` prefix): nvcc 12.x pre-built wheels, proprietary, only decompressible by libcudart.so internal paths — this is the remaining unsolved blocker for pre-built PyTorch cross-arch support

### cudaGetFuncBySymbol Two Resolution Paths (CUDA 12.8)

When `mod_mgmt+0x540 != 0`: vtable dispatch through resolution scope → module manager → core resolver
When `mod_mgmt+0x540 == 0`: direct call to core resolver at offset 0x22040, bypassing scope management

The probe confirms `0x540` stays at 0 throughout the lifecycle, so Path B (direct) is always taken. This matters for the Chidori server because it means the vtable-based resolution infrastructure is never initialized in our use case.
