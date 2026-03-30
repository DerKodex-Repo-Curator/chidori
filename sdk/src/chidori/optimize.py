"""Chidori inference optimizer — server-side graph decode for HuggingFace generate().

Copies the exact proven pattern from test_inference_cached.py (111.8 tok/s).
"""
import ctypes
import os
import functools
import sys


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
    """Run inference with server-side graph decode. Returns generated text.

    This is the public API. Exact copy of the proven test_inference_cached.py
    pattern that achieves 111-213 tok/s.
    """
    import torch
    from transformers.cache_utils import StaticCache

    lib = _load_lib()
    if lib is None or not hasattr(lib, 'chidori_graph_decode'):
        raise RuntimeError("chidori_graph_decode not available — is LD_PRELOAD set?")

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]
    max_cache_len = prompt_len + max_new_tokens + 16

    # StaticCache — exact same constructor as the working test
    static_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)

    # Prefill
    with torch.no_grad():
        pf = model(inputs["input_ids"], past_key_values=static_cache, use_cache=True)
        nt = pf.logits[:, -1:, :].argmax(-1)

    # Warmup decode
    cp = torch.tensor([prompt_len], device="cuda")
    with torch.no_grad():
        model(nt, past_key_values=static_cache, use_cache=True, cache_position=cp)
    torch.cuda.synchronize()

    # Stream warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    si = nt.clone()
    sp = torch.tensor([prompt_len + 1], device="cuda", dtype=torch.long)
    with torch.cuda.stream(s):
        with torch.no_grad():
            so = model(si, past_key_values=static_cache, use_cache=True, cache_position=sp)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # CUDA graph capture
    sptr = ctypes.c_void_p(s.cuda_stream)
    lib.cuStreamBeginCapture_v2(sptr, ctypes.c_int(0))
    with torch.cuda.stream(s):
        with torch.no_grad():
            so = model(si, past_key_values=static_cache, use_cache=True, cache_position=sp)
    gptr = ctypes.c_void_p(0)
    lib.cuStreamEndCapture(sptr, ctypes.byref(gptr))
    eptr = ctypes.c_void_p(0)
    lib.cuGraphInstantiate(ctypes.byref(eptr), gptr, ctypes.c_ulonglong(0))

    # Server-side decode — single RPC
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

    # Decode tokens
    token_ids = [nt.item()] + [out_tokens[i] for i in range(n)]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)

    print(f"[chidori] Server decode: {n} tokens in {dt:.1f}s = {tps:.1f} tok/s", file=sys.stderr)
    return text


def patch_generate():
    """Monkey-patch generate() to use server_decode automatically."""
    try:
        import torch  # noqa: F401
        from transformers import GenerationMixin
    except ImportError:
        return False

    lib = _load_lib()
    if lib is None or not hasattr(lib, 'chidori_graph_decode'):
        return False

    original_generate = GenerationMixin.generate

    @functools.wraps(original_generate)
    def optimized_generate(self, inputs=None, generation_config=None, **kwargs):
        import torch
        from transformers.cache_utils import StaticCache

        # Get input_ids
        input_ids = inputs
        if input_ids is None:
            input_ids = kwargs.pop("input_ids", None)
        if input_ids is None:
            return original_generate(self, inputs=inputs, generation_config=generation_config, **kwargs)

        # Get max_new_tokens from kwargs or generation_config
        max_new_tokens = kwargs.get("max_new_tokens", None)
        if max_new_tokens is None and generation_config is not None:
            max_new_tokens = getattr(generation_config, "max_new_tokens", None)

        do_sample = kwargs.get("do_sample", None)
        if do_sample is None and generation_config is not None:
            do_sample = getattr(generation_config, "do_sample", False)

        # Only optimize: greedy, known length, batch=1, CUDA
        if max_new_tokens is None or do_sample:
            return original_generate(self, inputs=input_ids, generation_config=generation_config, **kwargs)
        if input_ids.shape[0] != 1:
            return original_generate(self, inputs=input_ids, generation_config=generation_config, **kwargs)
        try:
            if next(self.parameters()).device.type != "cuda":
                return original_generate(self, inputs=input_ids, generation_config=generation_config, **kwargs)
        except StopIteration:
            return original_generate(self, inputs=input_ids, generation_config=generation_config, **kwargs)

        # === EXACT COPY OF test_inference_cached.py ===
        try:
            self.eval()
            prompt_len = input_ids.shape[1]
            max_cache_len = prompt_len + max_new_tokens + 16

            static_cache = StaticCache(config=self.config, max_cache_len=max_cache_len)
            with torch.no_grad():
                pf = self(input_ids, past_key_values=static_cache, use_cache=True)
                nt = pf.logits[:, -1:, :].argmax(-1)
            cp = torch.tensor([prompt_len], device="cuda")
            with torch.no_grad():
                self(nt, past_key_values=static_cache, use_cache=True, cache_position=cp)
            torch.cuda.synchronize()

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            si = nt.clone()
            sp = torch.tensor([prompt_len + 1], device="cuda", dtype=torch.long)
            with torch.cuda.stream(s):
                with torch.no_grad():
                    so = self(si, past_key_values=static_cache, use_cache=True, cache_position=sp)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            sptr = ctypes.c_void_p(s.cuda_stream)
            lib.cuStreamBeginCapture_v2(sptr, ctypes.c_int(0))
            with torch.cuda.stream(s):
                with torch.no_grad():
                    so = self(si, past_key_values=static_cache, use_cache=True, cache_position=sp)
            gptr = ctypes.c_void_p(0)
            lib.cuStreamEndCapture(sptr, ctypes.byref(gptr))
            eptr = ctypes.c_void_p(0)
            lib.cuGraphInstantiate(ctypes.byref(eptr), gptr, ctypes.c_ulonglong(0))

            out_tokens = (ctypes.c_int32 * max_new_tokens)()
            out_count = ctypes.c_int32(0)
            torch.cuda.synchronize()

            lib.chidori_graph_decode(
                ctypes.c_uint64(eptr.value), ctypes.c_uint64(sptr.value),
                ctypes.c_uint64(si.data_ptr()), ctypes.c_uint64(sp.data_ptr()),
                ctypes.c_uint64(so.logits.data_ptr()),
                ctypes.c_int32(self.config.vocab_size),
                ctypes.c_int32(max_new_tokens - 2), ctypes.c_int32(prompt_len + 2),
                out_tokens, ctypes.byref(out_count),
            )

            n = out_count.value
            all_ids = input_ids[0].tolist() + [nt.item()] + [out_tokens[i] for i in range(n)]
            return torch.tensor([all_ids], device=input_ids.device, dtype=torch.long)

        except Exception as e:
            import traceback
            print(f"[chidori] Graph decode failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return original_generate(self, inputs=input_ids, generation_config=generation_config, **kwargs)

    GenerationMixin.generate = optimized_generate
    return True


def optimize():
    """Enable server-side graph decode for HuggingFace generate()."""
    patched = patch_generate()
    if patched:
        print("[chidori] Server-side graph decode enabled for generate()", file=sys.stderr)
    return patched
