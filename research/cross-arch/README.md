# Cross-Architecture Research: CUDA Fat Binary Compression

## KEY FINDING (2026-03-20)

**Fat binary compression = standard zstd (Zstandard)**

NVIDIA's CUDA 13.x nvcc compresses fatbin entries using **zstd** (magic `28 b5 2f fd`).
This is a standard, well-supported compression algorithm — NOT proprietary.

### Fatbin Entry Format

```
Offset  Size  Field
0       2     entry_type (1=PTX, 2=ELF/cubin)
2       2     flags_low (version/attributes)
4       4     entry_header_size
8       8     uncompressed_data_size
16      4     compressed_data_size (0 if uncompressed)
20      4     flags (bit 6 = 0x40 = compressed)
24      4     unknown
28      4     sm_version (e.g., 89, 120)
...     ...   (padding to entry_header_size)
[hdr]   comp  data (zstd if flags & 0x40, raw otherwise)
```

### Compression Detection

- `flags & 0x40` at entry offset 20 = compressed
- Compressed data starts with zstd magic: `28 b5 2f fd`
- Decompress with standard zstd library → raw ELF cubin or PTX text

### Verified With

```python
import zstandard
dctx = zstandard.ZstdDecompressor()
decompressed = dctx.decompress(compressed_entry_data)
# PTX: starts with ".version 9.1"
# ELF: starts with 0x7f454c46
```

### Impact on Chidori

The server can now:
1. Receive fat binary from client
2. Parse entries, find matching architecture (or PTX)
3. If compressed (flags & 0x40): decompress with zstd
4. Pass raw ELF cubin to `cuModuleLoadData` (Driver API)
5. Get CUfunction via `cuModuleGetFunction`
6. Query params via `cuFuncGetParamInfo`
7. Launch with `cuLaunchKernel`

No need for:
- `__cudaRegisterFatBinary` (Runtime API)
- Synthetic device stubs
- `cudaGetFuncBySymbol`
- Building PyTorch from source

Just add `-lzstd` to the server build and implement decompression.
