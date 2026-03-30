#!/usr/bin/env python3
"""Simple vLLM test with Chidori GPU-over-IP."""
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams

def main():
    print("=== Simple vLLM Test ===")
    prompts = ["The capital of France is"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    print(f"Loading model...")
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
    )
    print("Generating...")
    outputs = llm.generate(prompts, sampling_params)
 for output in outputs:
        generated_text = output.outputs[0].text
        print(f"\nResult: {generated_text}")
    print("=== SUCCESS ===")
