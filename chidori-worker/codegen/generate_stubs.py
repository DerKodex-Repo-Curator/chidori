#!/usr/bin/env python3
"""
Chidori CUDA Codegen — Auto-generates client/server stubs from CUDA headers.

Parses cuda.h and cuda_runtime_api.h to generate:
  1. Client stubs (serialize args → send RPC → deserialize return)
  2. Server dispatch (deserialize → call real CUDA API → serialize return)
  3. Function ID enum additions

Usage:
  python3 generate_stubs.py --cuda-path /usr/local/cuda/include --output-dir ../src/generated/

This follows the SCUDA approach of generating from headers for sustainable
coverage of the CUDA API surface.
"""

import re
import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class CudaParam:
    """A single parameter of a CUDA function."""
    type: str           # e.g., "CUdeviceptr *", "size_t", "const void *"
    name: str           # e.g., "dptr", "bytesize"
    is_pointer: bool    # True if this is an output parameter
    is_const: bool      # True if const pointer (input buffer)
    base_type: str      # Type without pointer/const, e.g., "CUdeviceptr"

@dataclass
class CudaFunction:
    """A parsed CUDA API function declaration."""
    name: str           # e.g., "cuMemAlloc_v2"
    return_type: str    # Always "CUresult" for driver API
    params: List[CudaParam]
    original_decl: str  # Original declaration string
    category: str = ""  # memory, context, stream, etc.

# Map CUDA types to wire sizes
TYPE_SIZES = {
    "CUdeviceptr": 8,
    "CUdevice": 4,
    "CUcontext": 8,
    "CUmodule": 8,
    "CUfunction": 8,
    "CUstream": 8,
    "CUevent": 8,
    "size_t": 8,
    "unsigned int": 4,
    "int": 4,
    "float": 4,
    "double": 8,
    "unsigned long long": 8,
    "CUjit_option": 4,
}

# Functions we want to generate stubs for (Phase 1 target list)
TARGET_FUNCTIONS = {
    # Initialization
    "cuInit", "cuDriverGetVersion",
    # Device
    "cuDeviceGet", "cuDeviceGetCount", "cuDeviceGetName",
    "cuDeviceTotalMem_v2", "cuDeviceGetAttribute",
    "cuDeviceGetUuid",
    # Context
    "cuCtxCreate_v2", "cuCtxDestroy_v2", "cuCtxSetCurrent",
    "cuCtxGetCurrent", "cuCtxSynchronize", "cuCtxGetDevice",
    "cuDevicePrimaryCtxRetain", "cuDevicePrimaryCtxRelease_v2",
    # Module
    "cuModuleLoad", "cuModuleLoadData", "cuModuleLoadDataEx",
    "cuModuleUnload", "cuModuleGetFunction", "cuModuleGetGlobal_v2",
    # Memory
    "cuMemAlloc_v2", "cuMemFree_v2",
    "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2", "cuMemcpyDtoD_v2",
    "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoDAsync_v2",
    "cuMemsetD8_v2", "cuMemsetD16_v2", "cuMemsetD32_v2",
    "cuMemGetInfo_v2",
    # Execution
    "cuLaunchKernel",
    # Stream
    "cuStreamCreate", "cuStreamCreateWithPriority",
    "cuStreamDestroy_v2", "cuStreamSynchronize", "cuStreamQuery",
    "cuStreamWaitEvent",
    # Event
    "cuEventCreate", "cuEventDestroy_v2", "cuEventRecord",
    "cuEventSynchronize", "cuEventQuery", "cuEventElapsedTime",
}

def parse_param(param_str: str) -> Optional[CudaParam]:
    """Parse a single function parameter string."""
    param_str = param_str.strip()
    if not param_str or param_str == "void":
        return None

    is_const = param_str.startswith("const ")
    clean = param_str.replace("const ", "").strip()

    # Handle pointer types
    is_pointer = "*" in clean

    # Split type and name
    # Try to find the name (last word)
    parts = clean.replace("*", " * ").split()
    if len(parts) < 2:
        return CudaParam(
            type=param_str, name="arg", is_pointer=is_pointer,
            is_const=is_const, base_type=clean.replace("*", "").strip()
        )

    name = parts[-1]
    type_parts = parts[:-1]
    base_type = " ".join(p for p in type_parts if p != "*").strip()
    full_type = param_str.rsplit(name, 1)[0].strip()

    return CudaParam(
        type=full_type, name=name, is_pointer=is_pointer,
        is_const=is_const, base_type=base_type
    )

def parse_cuda_header(header_path: str) -> List[CudaFunction]:
    """Parse a CUDA header file and extract function declarations."""
    if not os.path.exists(header_path):
        print(f"Warning: {header_path} not found, skipping")
        return []

    with open(header_path, "r") as f:
        content = f.read()

    # Remove comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)

    # Find function declarations: CUresult CUDAAPI cuXxx(...)
    pattern = r'(CUresult)\s+(?:CUDAAPI\s+)?(\w+)\s*\(([^)]*)\)'

    functions = []
    for match in re.finditer(pattern, content):
        ret_type = match.group(1)
        func_name = match.group(2)
        params_str = match.group(3)

        if func_name not in TARGET_FUNCTIONS:
            continue

        params = []
        for p in params_str.split(","):
            param = parse_param(p)
            if param:
                params.append(param)

        functions.append(CudaFunction(
            name=func_name,
            return_type=ret_type,
            params=params,
            original_decl=match.group(0),
        ))

    return functions

def categorize(func_name: str) -> str:
    """Determine the category of a CUDA function."""
    name = func_name.lower()
    if "device" in name or name == "cuinit" or "driver" in name:
        return "device"
    elif "ctx" in name or "primaryctx" in name:
        return "context"
    elif "module" in name or "func" in name:
        return "module"
    elif "mem" in name or "pointer" in name:
        return "memory"
    elif "launch" in name:
        return "execution"
    elif "stream" in name:
        return "stream"
    elif "event" in name:
        return "event"
    return "misc"

def func_id_name(func_name: str) -> str:
    """Convert cuFunctionName to CHIDORI_CU_FUNCTION_NAME."""
    # Remove _v2 suffix
    name = re.sub(r'_v\d+$', '', func_name)
    # cuMemcpyHtoD -> CU_MEMCPY_H_TO_D
    # Insert underscore before uppercase letters
    result = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    result = result.upper()
    # cu_ -> CU_
    result = result.replace("CU_", "CU_", 1)
    return f"CHIDORI_{result}"

def generate_report(functions: List[CudaFunction], output_dir: str):
    """Generate a report of what would be generated."""
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "codegen_report.txt")
    with open(report_path, "w") as f:
        f.write("Chidori CUDA Codegen Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Functions found: {len(functions)}\n")
        f.write(f"Target functions: {len(TARGET_FUNCTIONS)}\n")
        f.write(f"Coverage: {len(functions)}/{len(TARGET_FUNCTIONS)} "
                f"({100*len(functions)//max(len(TARGET_FUNCTIONS),1)}%)\n\n")

        by_category = {}
        for func in functions:
            func.category = categorize(func.name)
            by_category.setdefault(func.category, []).append(func)

        for cat in sorted(by_category.keys()):
            f.write(f"\n## {cat.upper()}\n")
            for func in by_category[cat]:
                f.write(f"  {func.name}(")
                f.write(", ".join(f"{p.type} {p.name}" for p in func.params))
                f.write(f")\n")
                f.write(f"    -> {func_id_name(func.name)}\n")

        # List missing functions
        found_names = {f.name for f in functions}
        missing = TARGET_FUNCTIONS - found_names
        if missing:
            f.write(f"\n## MISSING ({len(missing)} functions not found in headers)\n")
            for name in sorted(missing):
                f.write(f"  {name}\n")

    print(f"Report written to {report_path}")
    return functions

def generate_server_dispatch(functions: List[CudaFunction], output_dir: str):
    """Generate server-side dispatch code that calls real CUDA APIs."""
    path = os.path.join(output_dir, "server_dispatch_generated.c")

    with open(path, "w") as f:
        f.write("/* AUTO-GENERATED by codegen/generate_stubs.py — DO NOT EDIT */\n")
        f.write("/* Server-side CUDA dispatch: deserialize, call real API, serialize response */\n\n")
        f.write("#include <cuda.h>\n")
        f.write("#include \"chidori_protocol.h\"\n\n")

        f.write("int dispatch_cuda_call(int fd, chidori_request_t *req, void *payload) {\n")
        f.write("    switch (req->func_id) {\n\n")

        for func in functions:
            fid = func_id_name(func.name)
            f.write(f"    case {fid}: {{\n")
            f.write(f"        /* {func.original_decl} */\n")
            f.write(f"        // TODO: Deserialize payload, call {func.name}(), serialize response\n")
            f.write(f"        return chidori_send_response(fd, req->seq_num, 0, NULL, 0);\n")
            f.write(f"    }}\n\n")

        f.write("    default:\n")
        f.write("        return chidori_send_response(fd, req->seq_num, 801, NULL, 0);\n")
        f.write("    }\n")
        f.write("}\n")

    print(f"Server dispatch written to {path}")

def generate_client_stubs(functions: List[CudaFunction], output_dir: str):
    """Generate client-side interception stubs."""
    path = os.path.join(output_dir, "client_stubs_generated.c")

    with open(path, "w") as f:
        f.write("/* AUTO-GENERATED by codegen/generate_stubs.py — DO NOT EDIT */\n")
        f.write("/* Client-side CUDA interception: serialize, send RPC, deserialize response */\n\n")
        f.write("#include \"chidori_protocol.h\"\n\n")
        f.write("/* chidori_rpc() is defined in juice_client.c */\n")
        f.write("extern int chidori_rpc(uint16_t func_id, const void *payload, size_t payload_len,\n")
        f.write("                     void *out_payload, size_t *out_payload_len);\n\n")

        for func in functions:
            fid = func_id_name(func.name)

            # Function signature
            param_strs = [f"{p.type} {p.name}" for p in func.params]
            f.write(f"CUresult {func.name}({', '.join(param_strs)}) {{\n")
            f.write(f"    /* -> {fid} */\n")
            f.write(f"    // TODO: Serialize parameters and call chidori_rpc()\n")
            f.write(f"    return chidori_rpc({fid}, NULL, 0, NULL, NULL);\n")
            f.write(f"}}\n\n")

    print(f"Client stubs written to {path}")

def main():
    parser = argparse.ArgumentParser(description="Chidori CUDA Codegen")
    parser.add_argument("--cuda-path", default="/usr/local/cuda/include",
                        help="Path to CUDA include directory")
    parser.add_argument("--output-dir", default="../src/generated",
                        help="Output directory for generated code")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate the report, no code")
    args = parser.parse_args()

    print(f"Chidori CUDA Codegen")
    print(f"CUDA headers: {args.cuda_path}")
    print(f"Output: {args.output_dir}\n")

    # Parse headers
    functions = []

    cuda_h = os.path.join(args.cuda_path, "cuda.h")
    functions.extend(parse_cuda_header(cuda_h))

    # Categorize
    for func in functions:
        func.category = categorize(func.name)

    print(f"Parsed {len(functions)} target functions from headers")

    # Generate report
    functions = generate_report(functions, args.output_dir)

    if not args.report_only:
        if functions:
            generate_server_dispatch(functions, args.output_dir)
            generate_client_stubs(functions, args.output_dir)
        else:
            print("\nNo functions parsed (CUDA headers not found?).")
            print("Generated report with target function list.")
            print("Install CUDA toolkit to enable full codegen.")

if __name__ == "__main__":
    main()
