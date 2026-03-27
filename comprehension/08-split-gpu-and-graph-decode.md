# 08 -- Split GPU and Graph Decode

## Overview

Two complementary server-side optimizations that eliminate per-token round-trips during autoregressive inference. Graph decode runs the full token generation loop on the server in a single RPC. Split GPU decode enables pipeline-parallel inference where the client's local GPU runs the first half of the model and the server runs the second half, with a single RPC per pipeline step.

## Server-Side Graph Decode (`CHIDORI_GRAPH_DECODE`, 0x0501)

### Purpose

In autoregressive token generation (e.g., LLM decoding), each token requires a forward pass through the model. Without graph decode, each token requires a full round-trip: client sends the token, server runs the model, server returns the logits, client runs argmax, client sends the next token. At 100 tokens with 1ms network latency, that is 200ms of pure network overhead.

Graph decode moves the entire generation loop to the server. One RPC call generates N tokens. The server captures the model as a CUDA graph (which is very fast to launch -- microseconds vs. milliseconds for a regular forward pass) and runs a tight CPU loop: launch graph, read logits, argmax, write next token, repeat.

### Payload (52 bytes)

```
graph_exec(8) + stream(8) + input_ptr(8) + position_ptr(8) +
logits_ptr(8) + vocab_size(4) + num_tokens(4) + start_position(4)
```

| Field | Description |
|---|---|
| `graph_exec` | Handle to an instantiated CUDA graph executable (from `cuGraphInstantiate`) |
| `stream` | CUDA stream for all operations |
| `input_ptr` | GPU memory pointer for the current input token (int64) |
| `position_ptr` | GPU memory pointer for the current position (int64) |
| `logits_ptr` | GPU memory pointer where the model writes output logits |
| `vocab_size` | Vocabulary size (determines logits read size: `vocab_size * 2` bytes for FP16) |
| `num_tokens` | Maximum tokens to generate |
| `start_position` | Position in the KV cache to start from (for continuation) |

### Per-Token Loop

For each token from `start_position` to `start_position + num_tokens`:

1. **Write position**: `cuMemcpyHtoD(position_ptr, &position, 8)` -- writes the current position (int64) to GPU memory
2. **Launch graph**: `cuGraphLaunch(exec, stream)` -- executes the captured CUDA graph, which runs the full model forward pass
3. **Synchronize**: `cuStreamSynchronize(stream)` -- waits for the graph to complete
4. **Read logits**: `cuMemcpyDtoH(host_buffer, logits_ptr, vocab_size * 2)` -- reads FP16 logits to pinned host memory (~25 GB/s on PCIe 4.0, so 128K vocab * 2 bytes = 256 KB takes ~10 microseconds)
5. **CPU argmax**: Finds the token with highest logit score using an FP16 bit trick (see below)
6. **Write next token**: `cuMemcpyHtoD(input_ptr, &next_token, 8)` -- writes the selected token ID (int64) to GPU memory for the next iteration
7. **EOS check**: Breaks if the selected token ID is 2 (standard EOS token)

### FP16 Argmax Bit Trick

Rather than converting FP16 values to float32 for comparison (expensive for large vocabularies), the argmax uses a direct comparison of the raw uint16 representation:

```
For each FP16 value stored as uint16:
  - Positive values: larger raw uint16 = bigger float value
  - Negative values: smaller raw uint16 = less negative = bigger float value
  - Comparison: check sign bit to choose comparison direction
    - If both positive: max(uint16_a, uint16_b)
    - If both negative: min(uint16_a, uint16_b)
    - If mixed: positive always wins
```

This works because IEEE 754 FP16 has the same sign-magnitude ordering property as FP32: the raw bit pattern preserves ordering within the same sign. No float conversion needed.

### Response

```
num_generated(4) + token_ids(4 * N)
```

Returns the actual number of tokens generated (may be less than `num_tokens` if EOS was hit) followed by the token IDs.

## Split GPU Decode (`CHIDORI_GRAPH_DECODE_SPLIT`, 0x0502)

### Purpose

Enables pipeline-parallel inference where the client machine has a local GPU running the first half of the model and the remote server runs the second half. This is useful when both machines have GPUs but neither alone can fit the full model, or when the client GPU can handle early layers efficiently.

One RPC call performs one pipeline step: the client sends intermediate activations from its local GPU, the server runs its layers plus argmax, and returns a single token ID.

### Payload

```
graph_exec(8) + stream(8) + activation_dst(8) + position_ptr(8) +
logits_ptr(8) + vocab_size(4) + position(4) + activation_size(4) +
activation_data(activation_size)
```

| Field | Description |
|---|---|
| `graph_exec` | Handle to CUDA graph for the server's half of the model |
| `stream` | CUDA stream |
| `activation_dst` | GPU memory pointer where activations should be placed |
| `position_ptr` | GPU memory pointer for position |
| `logits_ptr` | GPU memory pointer for output logits |
| `vocab_size` | Vocabulary size |
| `position` | Current position (scalar, not a pointer) |
| `activation_size` | Size of the activation data in bytes |
| `activation_data` | The intermediate activations from the client's local GPU (variable length) |

### Steps

1. **HtoD activations**: `cuMemcpyHtoD(activation_dst, activation_data, activation_size)` -- copies the intermediate activations from the client's local GPU to the server's GPU
2. **Update position**: Writes the position value to GPU memory
3. **Launch graph**: `cuGraphLaunch(exec, stream)` -- runs the server's half of the model (the captured CUDA graph)
4. **Synchronize**: `cuStreamSynchronize(stream)`
5. **DtoH logits**: `cuMemcpyDtoH(host_buffer, logits_ptr, vocab_size * 2)` -- reads logits
6. **CPU argmax**: Same FP16 bit trick as graph decode
7. **Return**: Single `token_id(4)` response

### Key Difference from Graph Decode

Graph decode runs a complete model loop for multiple tokens. Split decode runs one half of the model for a single token. The client drives the outer loop, calling split decode repeatedly and handling the first-half forward pass on its local GPU.

## CUDA Graph Capture and Replay

Both decode modes rely on CUDA graphs captured on the server side. The client initiates capture via intercepted CUDA stream capture APIs.

### StreamBeginCapture (0x0120)

- Forces **relaxed mode** (`mode=1`): global capture mode (`cudaStreamCaptureModeGlobal`) fails because non-capture operations (like network I/O) invalidate the capture. Relaxed mode allows non-capture-related operations on other streams.
- Sets `g_capturing_stream` to track which stream is being captured.

### StreamEndCapture (0x0121)

- Ends capture on the tracked stream, clears `g_capturing_stream`.
- Stores the resulting `CUgraph` in the `g_graphs` array.

### GraphInstantiate (0x0125)

- Calls `cuGraphInstantiate(&exec, graph, 0)` to create an executable graph from the captured graph.
- Stores the executable handle in `g_graph_execs` array.

### GraphLaunch (0x0127)

- Calls `cuGraphLaunch(exec, stream)` to execute the graph.
- This is a very fast operation (microseconds) because the GPU command buffer is pre-built.

## Client-Side Graph Decode Usage

### test_inference_multi.py

Captures a decode graph on the server, then calls `chidori_graph_decode` for batch decode of 5 prompts. Each prompt is decoded with a single RPC call that generates all tokens.

### test_inference_graph.py

Manual graph capture and replay per token. Demonstrates the lower-level approach where the client manages the per-token loop but still uses CUDA graphs for fast kernel launch.

### test_inference_cached.py

Compares three inference strategies:
- **Dynamic cache**: Standard HuggingFace `DynamicCache`, per-token allocation RPCs
- **Static cache**: Pre-allocated KV cache, eliminates per-token allocation overhead
- **Graph decode**: Full server-side generation loop, single RPC

### test_llama8b.py

Production-scale test with Llama-3.1-8B. Demonstrates graph decode working with a real model at useful scale.

## StaticCache Integration

StaticCache is a HuggingFace Transformers feature that pre-allocates the KV cache to a fixed maximum sequence length, eliminating the per-token `cudaMalloc` / `cudaFree` calls that would otherwise occur during autoregressive generation. In the Chidori context, this is critical because each allocation would be a separate RPC.

Configuration:
- `cache_implementation="static"` in `from_pretrained()`
- `use_cache=True` (default for generation)
- `cache_position` parameter for specifying the current position in the pre-allocated cache

Combined with graph decode, StaticCache ensures the entire decode loop has zero allocation RPCs.

## Test Files

| File | Description |
|---|---|
| `test_split_inference.py` | TinyLlama split across 2 GPUs via `accelerate.dispatch_model` |
| `test_split_graph.py` | Split inference with StaticCache |
| `test_inference_multi.py` | Server-side graph decode, 5 prompts batch decoded |
| `test_inference_flash.py` | Flash attention / SDPA / eager attention variants |
| `test_inference_cached.py` | Dynamic cache vs. StaticCache vs. graph decode comparison |
| `test_inference_graph.py` | CUDA graph capture/replay for decode |
| `test_llama8b.py` | Llama-3.1-8B production-scale graph decode test |

## Performance Characteristics

### Graph Decode

- **Per-token cost on server**: ~10-50 microseconds (graph launch + logit read + argmax + token write)
- **Network cost**: 1 RPC for the entire sequence (e.g., 100 tokens)
- **Without graph decode**: 100+ RPCs, each with ~0.5-2ms network round-trip = 50-200ms overhead
- **With graph decode**: 1 RPC, ~1-2ms network overhead total

### Split GPU Decode

- **Per-token cost**: 1 RPC per token (client drives the loop)
- **Activation transfer**: `activation_size` bytes sent per token (typically a few KB for intermediate layer outputs)
- **Use case**: Both machines have GPUs, model does not fit on either alone
- **Tradeoff**: More RPCs than graph decode, but enables larger models by splitting across two GPUs
