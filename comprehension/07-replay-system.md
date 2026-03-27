# 07 -- Replay System

## Purpose

The replay system records one training step's complete CUDA dispatch sequence, then replays it on subsequent steps sending only changed data (input batches). This eliminates per-call RPC overhead for repeated training iterations, which is the dominant performance cost in GPU-over-IP: a single PyTorch training step may issue hundreds of CUDA calls (kernel launches, memory copies, synchronization barriers), each requiring a separate TCP round-trip without replay. With replay, the entire sequence collapses to a single RPC carrying the changed inputs.

## Limits

| Constant | Value | Meaning |
|---|---|---|
| `REPLAY_MAX_VAR_SLOTS` | 8 | Maximum variable regions per call |
| `REPLAY_MAX_CALLS` | 8192 | Maximum calls per recorded sequence |
| `REPLAY_MAX_SEQUENCES` | 4 | Simultaneous recordings (e.g., forward + backward + optimizer + spare) |
| `REPLAY_MAX_H2D_SLOTS` | 64 | Host-to-device data slots per sequence |
| `REPLAY_MAX_D2D_SLOTS` | 16 | Device-to-host output slots per sequence |

## Call Classification (`replay_call_type_t`)

Every CUDA call seen during recording is classified into one of seven types. The classification determines how the call is handled during replay.

| Type | Value | Description |
|---|---|---|
| `REPLAY_CALL_FIXED` | 0 | Entirely reproducible. Same payload every step. Replayed verbatim. Examples: `cuMemsetD8`, `cuMemcpyDtoD`, `cuEventRecord`. |
| `REPLAY_CALL_KERNEL` | 1 | `cuLaunchKernel`. Fixed structure (grid dims, block dims, shared mem, stream), but the kernel parameter buffer contains pointers that may change between steps. |
| `REPLAY_CALL_MATMUL` | 2 | `cublasLtMatmul`. Fixed algorithm descriptors and compute types, but the four data pointers (A, B, C, D matrices) are variable -- they point to the current input/output buffers. |
| `REPLAY_CALL_H2D` | 3 | `cuMemcpyHtoD` / `cuMemcpyHtoDAsync` / `cudaMemcpy(kind=HtoD)`. The data payload must be sent fresh each step because the input batch changes. The destination address and size remain fixed. |
| `REPLAY_CALL_D2H` | 4 | `cuMemcpyDtoH` / `cuMemcpyDtoHAsync` / `cudaMemcpy(kind=DtoH)`. Output to return to the client (e.g., loss value, gradients for debugging). |
| `REPLAY_CALL_SYNC` | 5 | Synchronization point (`cuStreamSynchronize`, `cuCtxSynchronize`, `cudaDeviceSynchronize`). Must still execute on replay to maintain correct ordering. |
| `REPLAY_CALL_SKIP` | 6 | One-time setup operations that should not be replayed: `cuModuleLoadFatBinary`, `cuRegisterFatBinary`, `cuRegisterFunction`, `cuMemAlloc`, `cuMemAllocManaged`, `cublasCreate`, `cublasLtCreate`, `cudnnCreate`, etc. |

## How Classification Works (`classify_call`)

Classification happens in `cuda_dispatch.c` via the `classify_call()` function:

- **Direct match by `func_id`**: Most operations have a fixed function ID that maps to a single call type. For example, `0x0072` (`cuMemcpyHtoD`) is always `REPLAY_CALL_H2D`.
- **`cudaMemcpy` / `cudaMemcpyAsync`**: These use a single function ID but the direction depends on the `cudaMemcpyKind` field. The classifier reads the `kind` field at payload byte offset 24 to determine direction:
  - `kind = 1` (`cudaMemcpyHostToDevice`) --> `REPLAY_CALL_H2D`
  - `kind = 2` (`cudaMemcpyDeviceToHost`) --> `REPLAY_CALL_D2H`
  - Other kinds (DtoD, HtoH, Default) --> `REPLAY_CALL_FIXED` or `REPLAY_CALL_SKIP`

## Variable Slot Identification (`identify_var_slots`)

For each recorded call, the system must identify which bytes in the payload are "variable" (change between steps) and which are "fixed" (same every step). This is done by `identify_var_slots()`, which populates an array of `{offset, size}` descriptors.

### H2D Calls (variable data starts at specific offsets)

The data payload in a host-to-device copy follows a header of destination pointer, size, and optionally stream. Everything after the header is the variable data that changes each step.

| func_id | Function | Header | Variable offset |
|---|---|---|---|
| `0x0072` | `cuMemcpyHtoD` | dst(8) + size(8) = 16 bytes | 16 |
| `0x0075` | `cuMemcpyHtoDAsync` | dst(8) + size(8) + stream(8) = 24 bytes | 24 |
| `0x0102` | `cudaMemcpy` | dst(8) + src(8) + size(8) + kind(4) = 28 bytes | 28 |
| `0x0103` | `cudaMemcpyAsync` | dst(8) + src(8) + size(8) + kind(4) + stream(8) = 36 bytes | 36 |

### D2H Calls

Payload is `src(8) + size(8)`. The `d2h_size` is extracted from bytes 8-15. No variable slots -- the entire payload is fixed; the server captures the output separately.

### KERNEL Calls

`cuLaunchKernel` payloads have a 44-byte header (gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream) followed by a parameter buffer. Layout:

```
[0..43]   header (44 bytes, fixed)
[44]      has_params (1 byte)
[45..48]  buf_size (4 bytes)
[49..49+buf_size]  parameter buffer (VARIABLE)
```

Variable region: buffer at offset 49, size = `buf_size`.

### MATMUL Calls

`cublasLtMatmul` payloads are 308 bytes with a specific layout:

```
[0..187]     Fixed descriptors (operation, preferred config, etc.)
[188..195]   alpha (8 bytes)
[196..203]   beta (8 bytes)
[204..211]   A pointer (VARIABLE)
[212..219]   B pointer (VARIABLE)
[220..227]   C pointer (VARIABLE)
[228..235]   D pointer (VARIABLE)
[236..243]   stream
[244..307]   algorithm workspace (FIXED)
```

Variable region: 32 bytes at offset 204 (the four data matrix pointers).

## Recording Flow

### 1. Client Initiates Recording

The client sends `REPLAY_BEGIN_RECORD` (function ID `0x0400`) with a `sequence_id`.

### 2. Server Prepares Recording Slot

`replay_begin_record()` finds a free slot in the sequence array, zeroes it, and sets the recording state.

### 3. Normal RPCs Flow with Recording

As each CUDA call arrives via `cuda_dispatch()`, the dispatch function also calls `replay_record_call()`:

- Snapshots the payload via `malloc` + `memcpy` (the payload is copied, not referenced)
- Classifies the call type via `classify_call()`
- Calls `identify_var_slots()` to mark which regions of the payload are variable
- Stores `needs_response` based on the original flags (whether the caller expected a response)

### 4. Client Ends Recording

The client sends `REPLAY_END_RECORD` (function ID `0x0401`).

### 5. Server Finalizes the Sequence

`replay_end_record()` performs several operations:

- **FNV-1a fingerprint**: Computes a hash over all recorded calls. For each call, it hashes the `func_id` and the fixed portions of the payload (skipping variable slots). This fingerprint is used to validate that the replay request matches the recorded sequence.
  - For calls with variable slots: the hash covers segments in order: before the first slot, gaps between slots, after the last slot.
- **H2D manifest**: Builds a list of `{call_index, expected_size}` for every H2D call in the sequence. This tells the client exactly how much data to send for each variable slot on replay.
- **D2H manifest**: Builds a list of `{call_index, expected_size}` for every D2H call. (Currently used for bookkeeping; actual output capture is TODO.)

**Response**: 24 bytes total:

```
fingerprint(8) + call_count(4) + h2d_count(4) + total_var_bytes(8)
```

## Replay Flow

### 1. Client Sends Replay Request

The client sends `REPLAY_EXECUTE` (function ID `0x0402`) containing:
- The fingerprint (to validate sequence match)
- New H2D data for all variable slots (concatenated)

### 2. Server Validates and Executes

`replay_execute()`:

1. **Fingerprint validation**: Compares the received fingerprint against the recorded one. Returns `-2` on mismatch (the training graph has changed, client must re-record).
2. **H2D slot count validation**: Verifies the number of H2D data slots matches expectations. Returns `-3` on mismatch.
3. **Data substitution**: Copies new H2D data into the recorded payloads at the variable slot offsets. This modifies the recorded payload in-place.
4. **Dispatch loop**: Iterates through all recorded calls:
   - `REPLAY_CALL_SKIP`: continues (one-time setup, already done)
   - All other types: dispatches the call with `CHIDORI_FLAG_ASYNC` (suppresses per-call responses, since the single replay response covers everything)
   - `REPLAY_CALL_D2H`: dispatched with original flags (output capture is TODO)
5. **Completion**: Increments `replay_count`, returns 0 on success.

## FNV-1a Hashing

The replay system uses FNV-1a for fingerprinting sequences:

```
FNV_OFFSET_BASIS = 0xcbf29ce484222325
FNV_PRIME        = 0x100000001b3

Algorithm (per byte):
    hash = hash XOR byte
    hash = hash * FNV_PRIME
```

FNV-1a was chosen for its speed (single multiply per byte, no table lookups) and reasonable distribution properties. It is not cryptographically secure, but that is not required -- the purpose is to detect accidental sequence changes, not to prevent attacks.

## Client-Side Replay State Machine

The client library (`chidori_client.c`) manages replay state with three variables:

- `g_replay_mode`: `0` = off, `1` = recording, `2` = replaying
- `g_replay_step_count`: counts training steps
- `g_replay_warmup`: number of steps before recording starts (configurable via `CHIDORI_REPLAY_WARMUP` environment variable)

State transitions are triggered by `cudaDeviceSynchronize`:

| Current Mode | Condition | Action | New Mode |
|---|---|---|---|
| 0 (off) | `step_count >= warmup` | Send `REPLAY_BEGIN_RECORD` | 1 (recording) |
| 0 (off) | `step_count < warmup` | No action | 0 (off) |
| 1 (recording) | Always (after one step) | Send `REPLAY_END_RECORD`, receive fingerprint | 2 (replaying) |
| 2 (replaying) | Always | Collect H2D data, build `REPLAY_EXECUTE` payload, send single RPC | 2 (replaying) |

### Replay Step Execution (Mode 2)

When in replay mode, instead of sending individual CUDA calls over the wire, the client:

1. Intercepts H2D memory copies and buffers the data locally (using the H2D manifest to know expected sizes)
2. Skips sending all other calls (kernels, DtoD, syncs, etc.)
3. When `cudaDeviceSynchronize` fires, assembles a single `REPLAY_EXECUTE` message containing the fingerprint and all buffered H2D data
4. Sends one RPC instead of hundreds
5. Processes the single response

## Current Limitations

1. **D2H output capture is TODO**: The `d2h_out` and `d2h_out_sizes` arrays are never populated. Device-to-host copies during replay are dispatched but their output is not captured and returned to the client. This means replay currently only works for training steps where the loss/gradient is read on the server side.

2. **In-place payload modification**: Variable data substitution modifies the recorded payload in-place. After the first replay, the original H2D data is lost. If the client sends different-sized data on a subsequent replay, the system cannot detect this (it relies on fingerprint validation of the fixed portions only).

3. **Max 4 simultaneous sequences**: This limits the number of distinct CUDA call patterns that can be replayed concurrently. Typical usage: forward pass, backward pass, optimizer step, plus one spare.

4. **No sequence invalidation on model change**: If the model architecture changes (e.g., different layer sizes), the fingerprint will change and the client must detect this and re-record. The current implementation detects fingerprint mismatch but the client-side logic for automatic re-recording is not fully robust.
