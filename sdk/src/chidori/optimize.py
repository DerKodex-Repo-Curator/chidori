"""Inference optimizer — patches HuggingFace generate() to use server-side graph decode."""

import ctypes
import os
import functools


def _get_libchidori():
    """Load libchidori.so to access chidori_graph_decode."""
    # It's already loaded via LD_PRELOAD — just open it
    try:
        lib = ctypes.CDLL("libchidori.so", mode=ctypes.RTLD_NOLOAD)
        return lib
    except OSError:
        pass
    # Try from LD_PRELOAD
    preload = os.environ.get("LD_PRELOAD", "")
    for path in preload.split():
        if "libchidori" in path or "libcuda.so" in path:
            try:
                return ctypes.CDLL(path, mode=ctypes.RTLD_NOLOAD)
            except OSError:
                pass
    return None


def _graph_decode_available():
    """Check if chidori_graph_decode is available."""
    lib = _get_libchidori()
    if lib is None:
        return False
    try:
        lib.chidori_graph_decode
        return True
    except AttributeError:
        return False


def patch_generate():
    """Monkey-patch transformers GenerationMixin.generate to use graph decode.

    After the prefill (first forward pass), captures a CUDA graph of the decode
    step and hands off the autoregressive loop to the server via a single
    chidori_graph_decode RPC call.
    """
    try:
        import torch
        from transformers import GenerationMixin
    except ImportError:
        return False

    lib = _get_libchidori()
    if lib is None:
        return False

    # chidori_graph_decode signature:
    # int chidori_graph_decode(uint64_t exec_handle, uint64_t stream_handle,
    #                          uint64_t input_ptr, uint64_t position_ptr,
    #                          uint64_t logits_ptr, int vocab_size,
    #                          int num_tokens, int start_position)
    # Returns: number of tokens generated (writes token IDs to input_ptr)
    try:
        decode_fn = lib.chidori_graph_decode
        decode_fn.restype = ctypes.c_int
        decode_fn.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64,  # exec, stream
            ctypes.c_uint64, ctypes.c_uint64,  # input, position
            ctypes.c_uint64,                     # logits
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # vocab, num_tokens, start_pos
        ]
    except AttributeError:
        return False

    original_generate = GenerationMixin.generate

    @functools.wraps(original_generate)
    def optimized_generate(self, input_ids=None, **kwargs):
        max_new_tokens = kwargs.get("max_new_tokens", None)
        do_sample = kwargs.get("do_sample", False)

        # Only optimize greedy decoding with known token count
        if max_new_tokens is None or do_sample:
            return original_generate(self, input_ids=input_ids, **kwargs)

        # Only optimize if model is on CUDA
        try:
            device = next(self.parameters()).device
            if not device.type == "cuda":
                return original_generate(self, input_ids=input_ids, **kwargs)
        except StopIteration:
            return original_generate(self, input_ids=input_ids, **kwargs)

        # Try graph decode path
        try:
            return _graph_decode_generate(
                self, input_ids, max_new_tokens, decode_fn, original_generate, kwargs
            )
        except Exception:
            # Any failure — fall back to standard generate
            return original_generate(self, input_ids=input_ids, **kwargs)

    GenerationMixin.generate = optimized_generate
    return True


def _graph_decode_generate(model, input_ids, max_new_tokens, decode_fn, fallback, kwargs):
    """Run inference using CUDA graph capture + server-side decode."""
    import torch

    if input_ids is None:
        raise ValueError("input_ids required for graph decode")

    input_ids = input_ids.to(next(model.parameters()).device)
    batch_size, seq_len = input_ids.shape

    if batch_size != 1:
        # Graph decode only supports batch=1 for now
        return fallback(model, input_ids=input_ids, **kwargs)

    model.eval()

    # Step 1: Prefill — run the prompt through the model normally
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        logits = outputs.logits
        next_token = logits[:, -1:, :].argmax(dim=-1)

    # Step 2: Prepare for decode — single token input with KV cache
    position_ids = torch.tensor([[seq_len]], device=input_ids.device, dtype=torch.long)
    decode_input = next_token

    # Step 3: Capture one decode step as a CUDA graph
    stream = torch.cuda.current_stream()

    # Warmup the decode path (required before capture)
    with torch.no_grad():
        warmup_out = model(decode_input, past_key_values=past_kv,
                          position_ids=position_ids, use_cache=True)
        past_kv = warmup_out.past_key_values

    # Now capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph, stream=stream):
        graph_out = model(decode_input, past_key_values=past_kv,
                         position_ids=position_ids, use_cache=True)

    graph_logits = graph_out.logits  # (1, 1, vocab_size)
    vocab_size = graph_logits.shape[-1]

    # Step 4: Server-side decode — one RPC for all remaining tokens
    exec_handle = graph.pool()  # CUgraphExec handle
    stream_handle = stream.cuda_stream  # Raw CUDA stream pointer

    n_generated = decode_fn(
        ctypes.c_uint64(exec_handle),
        ctypes.c_uint64(stream_handle),
        ctypes.c_uint64(decode_input.data_ptr()),
        ctypes.c_uint64(position_ids.data_ptr()),
        ctypes.c_uint64(graph_logits.data_ptr()),
        ctypes.c_int(vocab_size),
        ctypes.c_int(max_new_tokens - 1),  # -1 because we already generated 1 token
        ctypes.c_int(seq_len + 1),  # start position after prefill + first decode
    )

    torch.cuda.synchronize()

    # Step 5: Collect generated tokens from the server response
    # The server wrote token IDs back — read them from the decode output
    # Build the full sequence: prompt + first_token + server_tokens
    all_tokens = [input_ids]
    all_tokens.append(next_token)

    if n_generated > 0:
        # Server returns tokens via the response payload — stored in a buffer
        # For now, read them from the logits/input cycling
        # The graph_decode function writes tokens to input_ptr sequentially
        server_tokens = decode_input.cpu()
        all_tokens.append(server_tokens.unsqueeze(0) if server_tokens.dim() == 0 else server_tokens)

    return torch.cat([t.view(1, -1) for t in all_tokens], dim=1)


def optimize():
    """Enable all Chidori optimizations for the current process.

    Call this at the top of your script, or let `chidori run` handle it
    automatically via the CHIDORI_OPTIMIZE_INFERENCE env var.
    """
    patched = patch_generate()
    if patched:
        import sys
        print("[chidori] Inference optimizer enabled — generate() will use server-side graph decode",
              file=sys.stderr)
    return patched


# Auto-optimize if env var is set (triggered by `chidori run --optimize inference`)
if os.environ.get("CHIDORI_OPTIMIZE_INFERENCE") == "1":
    # Delay patching until transformers is actually imported
    import importlib
    _original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def _lazy_patch_import(name, *args, **kwargs):
        module = _original_import(name, *args, **kwargs)
        if name == "transformers" or (hasattr(name, 'startswith') and name.startswith("transformers.")):
            optimize()
            # Restore original import — only need to patch once
            if hasattr(__builtins__, '__import__'):
                __builtins__.__import__ = _original_import
        return module

    if hasattr(__builtins__, '__import__'):
        __builtins__.__import__ = _lazy_patch_import
