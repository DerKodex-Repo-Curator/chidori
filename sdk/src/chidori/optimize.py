"""Chidori inference optimizer — patches _sample() to use server-side graph decode.

Instead of patching generate() and reimplementing its setup logic, we patch
the inner decode loop (_sample) and let generate() handle everything else:
GenerationConfig parsing, cache creation, prefill, stopping criteria.

After prefill, we capture one decode step as a CUDA graph and hand off the
entire autoregressive loop to the server via chidori_graph_decode().
"""
import ctypes
import os
import functools
import sys

RTLD_NOLOAD = getattr(ctypes, 'RTLD_NOLOAD', 0x00004)


def _load_lib():
    """Load libchidori / libcuda.so.1 from LD_LIBRARY_PATH."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    for d in ld_path.split(":"):
        for name in ["libcuda.so.1", "libchidori.so"]:
            path = os.path.join(d, name)
            if os.path.exists(path):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    pass
    for name in ["libcuda.so.1", "libchidori.so"]:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    return None


def server_decode(model, tokenizer, prompt, max_new_tokens=200):
    """Direct API: run inference with server-side graph decode. Returns text.

    Exact copy of the proven test_inference_cached.py pattern (104-213 tok/s).
    """
    import torch
    from transformers.cache_utils import StaticCache

    lib = _load_lib()
    if lib is None or not hasattr(lib, 'chidori_graph_decode'):
        raise RuntimeError("chidori_graph_decode not available")

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]
    max_cache_len = prompt_len + max_new_tokens + 16

    static_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    with torch.no_grad():
        pf = model(inputs["input_ids"], past_key_values=static_cache, use_cache=True)
        nt = pf.logits[:, -1:, :].argmax(-1)
    cp = torch.tensor([prompt_len], device="cuda")
    with torch.no_grad():
        model(nt, past_key_values=static_cache, use_cache=True, cache_position=cp)
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    si = nt.clone()
    sp = torch.tensor([prompt_len + 1], device="cuda", dtype=torch.long)
    with torch.cuda.stream(s):
        with torch.no_grad():
            so = model(si, past_key_values=static_cache, use_cache=True, cache_position=sp)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    sptr = ctypes.c_void_p(s.cuda_stream)
    lib.cuStreamBeginCapture_v2(sptr, ctypes.c_int(0))
    with torch.cuda.stream(s):
        with torch.no_grad():
            so = model(si, past_key_values=static_cache, use_cache=True, cache_position=sp)
    gptr = ctypes.c_void_p(0)
    lib.cuStreamEndCapture(sptr, ctypes.byref(gptr))
    eptr = ctypes.c_void_p(0)
    lib.cuGraphInstantiate(ctypes.byref(eptr), gptr, ctypes.c_ulonglong(0))

    out_tokens = (ctypes.c_int32 * max_new_tokens)()
    out_count = ctypes.c_int32(0)
    torch.cuda.synchronize()

    import time
    t0 = time.time()
    lib.chidori_graph_decode(
        ctypes.c_uint64(eptr.value), ctypes.c_uint64(sptr.value),
        ctypes.c_uint64(si.data_ptr()), ctypes.c_uint64(sp.data_ptr()),
        ctypes.c_uint64(so.logits.data_ptr()),
        ctypes.c_int32(model.config.vocab_size),
        ctypes.c_int32(max_new_tokens - 2), ctypes.c_int32(prompt_len + 2),
        out_tokens, ctypes.byref(out_count),
    )
    dt = time.time() - t0
    n = out_count.value
    tps = n / dt if dt > 0 else 0

    token_ids = [nt.item()] + [out_tokens[i] for i in range(n)]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"[chidori] Server decode: {n} tokens in {dt:.1f}s = {tps:.1f} tok/s", file=sys.stderr)
    return text


def optimize():
    """Patch _sample() to use server-side graph decode for greedy inference.

    Lets generate() handle all setup (GenerationConfig, cache, prefill).
    We only replace the inner decode loop inside _sample().
    """
    try:
        import torch
        from transformers import GenerationMixin
        from transformers.cache_utils import StaticCache
    except ImportError:
        return False

    lib = _load_lib()
    if lib is None or not hasattr(lib, 'chidori_graph_decode'):
        return False

    original_sample = GenerationMixin._sample

    @functools.wraps(original_sample)
    def patched_sample(self, input_ids, logits_processor=None, stopping_criteria=None,
                       generation_config=None, synced_gpus=False, streamer=None,
                       **model_kwargs):
        """Replaces the decode loop with CUDA graph capture + server-side decode.

        generate() already did: GenerationConfig parsing, cache creation,
        _prepare_model_inputs, _prepare_special_tokens. input_ids contains
        the prompt. model_kwargs has past_key_values, attention_mask, etc.

        We let the prefill happen normally via _sample's first iteration,
        then hijack the decode loop.
        """
        # Only optimize: greedy, batch=1, StaticCache, CUDA
        past_kv = model_kwargs.get("past_key_values")
        if (generation_config.do_sample
            or input_ids.shape[0] != 1
            or not isinstance(past_kv, StaticCache)
            or input_ids.device.type != "cuda"):
            return original_sample(self, input_ids, logits_processor,
                                   stopping_criteria, generation_config,
                                   synced_gpus, streamer, **model_kwargs)

        try:
            return _graph_decode_sample(
                self, input_ids, logits_processor, stopping_criteria,
                generation_config, synced_gpus, streamer, lib, model_kwargs
            )
        except Exception as e:
            print(f"[chidori] Graph decode failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Reset cache and fall back
            past_kv = model_kwargs.get("past_key_values")
            if hasattr(past_kv, 'reset'):
                past_kv.reset()
            return original_sample(self, input_ids, logits_processor,
                                   stopping_criteria, generation_config,
                                   synced_gpus, streamer, **model_kwargs)

    GenerationMixin._sample = patched_sample
    print("[chidori] Server-side graph decode enabled for generate()", file=sys.stderr)
    return True


def _graph_decode_sample(self, input_ids, logits_processor, stopping_criteria,
                         generation_config, synced_gpus, streamer, lib, model_kwargs):
    """Graph capture + server-side decode. Uses the EXACT proven pattern from
    test_inference_cached.py — no prepare_inputs_for_generation, just direct
    model calls with StaticCache and cache_position.
    """
    import torch

    static_cache = model_kwargs.get("past_key_values")
    prompt_len = input_ids.shape[1]
    max_new = generation_config.max_new_tokens
    if max_new is None:
        max_new = generation_config.max_length - prompt_len

    # === Prefill ===
    with torch.no_grad():
        pf = self(input_ids, past_key_values=static_cache, use_cache=True)
        nt = pf.logits[:, -1:, :].argmax(-1)

    # === Warmup decode ===
    cp = torch.tensor([prompt_len], device="cuda")
    with torch.no_grad():
        self(nt, past_key_values=static_cache, use_cache=True, cache_position=cp)
    torch.cuda.synchronize()

    if max_new <= 1:
        return torch.cat([input_ids, nt], dim=-1)

    # === Stream warmup ===
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    si = nt.clone()
    sp = torch.tensor([prompt_len + 1], device="cuda", dtype=torch.long)
    with torch.cuda.stream(s):
        with torch.no_grad():
            so = self(si, past_key_values=static_cache, use_cache=True, cache_position=sp)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # === CUDA graph capture ===
    sptr = ctypes.c_void_p(s.cuda_stream)
    lib.cuStreamBeginCapture_v2(sptr, ctypes.c_int(0))
    with torch.cuda.stream(s):
        with torch.no_grad():
            so = self(si, past_key_values=static_cache, use_cache=True, cache_position=sp)
    gptr = ctypes.c_void_p(0)
    lib.cuStreamEndCapture(sptr, ctypes.byref(gptr))
    eptr = ctypes.c_void_p(0)
    lib.cuGraphInstantiate(ctypes.byref(eptr), gptr, ctypes.c_ulonglong(0))

    # === Save IDs to CPU before graph decode (CUDA state gets dirty) ===
    saved_prompt_ids = input_ids[0].tolist()
    first_token_id = nt.item()

    # === Server-side decode ===
    remaining = max_new - 2
    out_tokens = (ctypes.c_int32 * max(remaining, 1))()
    out_count = ctypes.c_int32(0)
    torch.cuda.synchronize()

    import time
    t0 = time.time()
    lib.chidori_graph_decode(
        ctypes.c_uint64(eptr.value), ctypes.c_uint64(sptr.value),
        ctypes.c_uint64(si.data_ptr()), ctypes.c_uint64(sp.data_ptr()),
        ctypes.c_uint64(so.logits.data_ptr()),
        ctypes.c_int32(self.config.vocab_size),
        ctypes.c_int32(remaining), ctypes.c_int32(prompt_len + 2),
        out_tokens, ctypes.byref(out_count),
    )
    dt = time.time() - t0
    n = out_count.value
    tps = n / dt if dt > 0 else 0
    print(f"[chidori] Server decode: {n} tokens in {dt:.1f}s = {tps:.1f} tok/s", file=sys.stderr)

    # === Assemble output (same format as _sample returns) ===
    # Keep on CPU — CUDA state is dirty after graph decode
    all_ids = saved_prompt_ids + [first_token_id] + [out_tokens[i] for i in range(n)]
    return torch.tensor([all_ids], dtype=torch.long)
