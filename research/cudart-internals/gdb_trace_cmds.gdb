# Break on every key function in the resolution chain
break cudaGetFuncBySymbol
break getRuntimeState
break checkShuttingDown
break *0x23160
break *0x466d0
break *0x47350
break *0x46bb0
break *0x28cc0
break *0x28e30
break *0x27a00
break *0x27020
break *0x27fd0
break *0x900a0

break *0x8e360
break *0x4cac0
break *0x46510

break *0x26730
break *0x267a0
break *0x28e08
break *0x26920
echo ""
echo "Running full trace..."
echo ""

echo "set pagination off"
echo ""

# Build the test program that uses the runtime API directly
echo "#include <stdio.h>" > /tmp/gdb_test.c
echo "#include <stdlib.h>" >> /tmp/gdb_test.c
echo "#include <string.h>" >> /tmp/gdb_test.c
echo "#include <cuda.h>" >> /tmp/gdb_test.c
echo "#include <cuda_runtime.h>" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "// Forward declarations for registration functions" >> /tmp/gdb_test.c
echo "extern void **__cudaRegisterFatBinary(void *);" >> /tmp/gdb_test.c
echo "extern void __cudaRegisterFatBinaryEnd(void **);" >> /tmp/gdb_test.c
echo "extern void __cudaRegisterFunction(void **, hostFun, char *name," >> /tmp/gdb_test.c
echo "    char *name2, int thread_limit," >> /tmp/gdb_test.c
echo "    void *extra1, void *extra2, void *extra3," >> /tmp/gdb_test.c
echo "    void *extra4, void *extra5);" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "// External fatbin data (compile with nvcc)" >> /tmp/gdb_test.c
echo "extern const void *__fat_device_text[];" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "int main(int argc, char **argv) {" >> /tmp/gdb_test.c
echo "    const char *fatbin_path = argv[1] ? argv[1] : \"$FATBIN\";" >> /tmp/gdb_test.c
echo "    const char *kernel_name = argv[2]? argv[2] : \"$KERNEL_NAME\";" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "    // Load fat binary data from file" >> /tmp/gdb_test.c
echo "    FILE *f = fopen(fatbin_path, \"rb\");" >> /tmp/gdb_test.c
echo "    if (!f) { fprintf(stderr, \"Cannot open %s\\n\", fatbin_path); return 1; }" >> /tmp/gdb_test.c
echo "    fseek(f, 0, SEEK_END);" >> /tmp/gdb_test.c
echo "    long size = ftell(f);" >> /tmp/gdb_test.c
echo "    void *fat_data = malloc(size);" >> /tmp/gdb_test.c
echo "    fread(fat_data, 1, size, f);" >> /tmp/gdb_test.c
echo "    fclose(f);" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "    // Step 1: Initialize CUDA runtime" >> /tmp/gdb_test.c
echo "    void *dummy;" >> /tmp/gdb_test.c
echo "    cudaError_t err = cudaMalloc(&dummy, 256);" >> /tmp/gdb_test.c
echo "    if (err != cudaSuccess) { fprintf(stderr, \"cudaMalloc failed: %d\\n\", err); return 1; }" >> /tmp/gdb_test.c
echo "    cudaFree(dummy);" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "    // Step 2: Register fat binary" >> /tmp/gdb_test.c
echo "    // Build wrapper struct" >> /tmp/gdb_test.c
echo "    struct { int magic; int version; const void *data; void *filename; } wrapper;" >> /tmp/gdb_test.c
echo "    wrapper.magic = 0x466243b1;   // __cudaFatMAGIC2" >> /tmp/gdb_test.c
echo "    wrapper.version = 1;" >> /tmp/gdb_test.c
echo "    wrapper.data = fat_data;" >> /tmp/gdb_test.c
echo "    wrapper.filename = NULL;" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "    void **handle = __cudaRegisterFatBinary(&wrapper);" >> /tmp/gdb_test.c
echo "    fprintf(stderr, \"__cudaRegisterFatBinary: handle=%p\\n\", (void*)handle);" >> /tmp/gdb_test.c
echo "    __cudaRegisterFatBinaryEnd(handle);" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "    // Step 3: Create Driver API context" >> /tmp/gdb_test.c
echo "    CUcontext ctx;" >> /tmp/gdb_test.c
echo "    CUresult res = cuDevicePrimaryCtxRetain(&ctx, 0);" >> /tmp/gdb_test.c
echo "    fprintf(stderr, \"cuDevicePrimaryCtxRetain: %d ctx=%p\\n\", res, (void*)ctx);" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "    // Step 4: Register function with stub" >> /tmp/gdb_test.c
echo "    void *stub = malloc(64);" >> /tmp/gdb_test.c
echo "    memset(stub, 0, 64);" >> /tmp/gdb_test.c
echo "    __cudaRegisterFunction(handle, (const char*)stub, (char*)kernel_name," >> /tmp/gdb_test.c
echo "        (char*)kernel_name, -1, NULL, NULL, NULL, NULL, NULL);" >> /tmp/gdb_test.c
echo "    fprintf(stderr, \"__cudaRegisterFunction: stub=%p name='%s'\\n\", stub, kernel_name);" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "    // Step 5: Resolve via cudaGetFuncBySymbol" >> /tmp/gdb_test.c
echo "    CUfunction func;" >> /tmp/gdb_test.c
echo "    cudaError_t rt_err = cudaGetFuncBySymbol(&func, stub);" >> /tmp/gdb_test.c
echo "    fprintf(stderr, \"cudaGetFuncBySymbol: err=%d (%s) func=%p\\n\"," >> /tmp/gdb_test.c
echo "            (int)rt_err, cudaGetErrorString(rt_err), (void*)func);" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "    if (rt_err == cudaSuccess) {" >> /tmp/gdb_test.c
echo "        fprintf(stderr, \"SUCCESS: Resolved function!\\n\");" >> /tmp/gdb_test.c
echo "    }" >> /tmp/gdb_test.c
echo "" >> /tmp/gdb_test.c
echo "    return (rt_err == cudaSuccess) ? 0 : 1;" >> /tmp/gdb_test.c
echo "}" >> /tmp/gdb_test.c

# Compile and run under gdb
echo "Compiling test program..."
