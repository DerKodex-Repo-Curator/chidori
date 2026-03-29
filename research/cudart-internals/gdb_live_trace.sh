#!/bin/bash
# GDB Live Trace — Proves the actual execution path through cudaGetFuncBySymbol
#
# Strategy: Set breakpoints at every function in the call chain, When cudaGetFuncBySymbol
# is hit, record the backtrace to prove which functions were actually called.
#
# Usage: ./gdb_live_trace.sh [fatbin] [kernel_name]

FATBIN="${1:-simple_sm89.fatbin}"
KERNEL="${2:-simple_add}"

echo "================================================================"
echo " GDB Live Trace — cudaGetFuncBySymbol Actual Execution Path"
echo " Fat binary: $FATBIN"
echo " Kernel: $KERNEL"
echo "================================================================"

# Step 1: Build test program
cat > _gdb_test.c <<'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

extern void **__cudaRegisterFatBinary(void *);
extern void __cudaRegisterFatBinaryEnd(void **);
extern void __cudaRegisterFunction(void **, const char *, char *, char *, int,
    void *, void *, void *, void *, void *);

// Marker functions — GDB breakpoints on these prove we entered the resolver
__attribute__((noinline)) void marker_before_resolve(void *stub) {
    volatile int x = *(volatile int*)stub; // force use
    (void)x;
}

int main(int argc, char **argv) {
    const char *fatbin_path = argv[1];
    const char *kernel_name = argv[2];

    // Force runtime init
    void *dummy;
    cudaMalloc(&dummy, 256);
    cudaFree(dummy);

    // Load fat binary from file
    FILE *f = fopen(fatbin_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fatbin_path); return 1; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    void *fat_data = malloc(size);
    fread(fat_data, 1, size, f);
    fclose(f);

    // Build wrapper
    struct { int magic; int version; const void *data; void *filename; } wrapper;
    wrapper.magic = 0x466243b1;
    wrapper.version = 1;
    wrapper.data = fat_data;
    wrapper.filename = NULL;

    // Register
    void **handle = __cudaRegisterFatBinary(&wrapper);
    fprintf(stderr, "[TRACE] __cudaRegisterFatBinary: handle=%p\n", (void*)handle);
    __cudaRegisterFatBinaryEnd(handle);

    // Create driver context
    CUcontext ctx;
    CUresult res = cuDevicePrimaryCtxRetain(&ctx, 0);
    fprintf(stderr, "[TRACE] cuDevicePrimaryCtxRetain: res=%d ctx=%p\n", res, (void*)ctx);

    // Register function with stub
    void *stub = malloc(64);
    memset(stub, 0, 64);
    __cudaRegisterFunction(handle, (const char*)stub, (char*)kernel_name,
        (char*)kernel_name, -1, NULL, NULL, NULL, NULL, NULL);
    fprintf(stderr, "[TRACE] __cudaRegisterFunction: stub=%p name='%s'\n", stub, kernel_name);

    // THE CRITICAL CALL
    CUfunction func;
    marker_before_resolve(stub);
    cudaError_t rt_err = cudaGetFuncBySymbol(&func, stub);
    fprintf(stderr, "[TRACE] cudaGetFuncBySymbol: err=%d func=%p\n",
            (int)rt_err, (void*)func);

    return (rt_err == cudaSuccess) ? 0 : 1;
}
CEOF

echo "Compiling..."
gcc -g -O0 -fPIC -o _gdb_test _gdb_test.c \
    -I/opt/cuda/include \
    -L/opt/cuda/targets/x86_64-linux/lib -lcuda -lcudart -ldl -lpthread \
    -Wl,-rpath,/opt/cuda/targets/x86_64-linux/lib 2>&1
if [ $? -ne 0 ]; then
    echo "Compile FAILED"; exit 1; fi
echo ""

# Step 2: Create GDB script
# Key insight: We can't use function names for internal functions because they're not exported.
# But we CAN set breakpoints by address. We find the addresses from the binary.
# First find libcudart base address range

LIBCUDART=$(ldconfig -p 2>/dev/null | grep libcudart | head -1)
if [ -z "$LIBCUDART" ]; then
    # Try common paths
    for p in /opt/cuda/targets/x86_64-linux/lib/libcudart.so.*; do
        [ -f "$p" ] && LIBCUDART="$p" && break
    done
fi

echo "Using libcudart: $LIBCUDART"
echo ""

# Get the .text section offset for the library
TEXT_OFFSET=$(readelf -S "$LIBCUDART" | grep '\.text' | awk '{print $5}' | head -1)
echo "libcudart .text section offset: 0x$TEXT_OFFSET"
echo ""

# We need to calculate the actual virtual addresses
# The disassembly offsets we have (0x79600, etc.) are relative to the library's base
# In GDB, we can set breakpoints using the library base + offset

# Key function offsets (from our disassembly)
OFFSET_GET_FUNC_BY_SYM=0x79600
OFFSET_GET_RUNTIME_STATE=0x2a800
OFFSET_CHECK_SHUTTING_DOWN=0x42b00
OFFSET_ALT_RESOLVER=0x23160
OFFSET_BOOTSTRAP=0x466d0
OFFSET_REAL_BOOTSTRAP=0x47350
OFFSET_CREATE_MODULE_MGMT=0x46bb0
OFFSET_HASH_LOOKUP=0x28cc0
OFFSET_CORE_RESOLUTION=0x28e30
OFFSET_HASH_INSERT=0x27a00
OFFSET_REGISTER_FAT_BIN_HANDLER=0x27020
OFFSET_REGISTER_FUNC_HANDLER=0x27fd0

# Step 3: Create the GDB command file
cat > _gdb_cmds.gdb <<ENDOF
set pagination off
set confirm off

# Set breakpoints at the marker function (our code, and key internal functions
# We use 'break *address' to break in the shared library
break marker_before_resolve

# Break at cudaGetFuncBySymbol entry
break *${OFFSET_GET_FUNC_BY_SYM}
# Break at getRuntimeState
break *${OFFSET_GET_RUNTIME_STATE}
# Break at checkShuttingDown
break *${OFFSET_CHECK_SHUTTING_DOWN}
# Break at the alternative resolver
break *${OFFSET_ALT_RESOLVER}
# Break at bootstrap
break *${OFFSET_BOOTSTRAP}
# Break at real bootstrap
break *${OFFSET_REAL_BOOTSTRAP}
# Break at hash lookup
break *${OFFSET_HASH_LOOKUP}
# Break at core resolution
break *${OFFSET_CORE_RESOLUTION}

# When we hit marker_before_resolve, we're about to call cudaGetFuncBySymbol
# Continue to the first breakpoint hit
continue

# At this point we've hit one of the breakpoints. Show which one.
printf "\n================================================================\n"
printf " BREAKPOINT HIT — checking which function...\n"
printf "================================================================\n"
info line
bt
printf "\n"

# Continue to each breakpoint and log
set \$BP_COUNT =0
set \$LOG=""

# Loop: continue, log breakpoint, stop after 20 hits or after cudaGetFuncBySymbol returns
define _log_bp
  set \$BP_COUNT += 1
  printf "[BP #%d] ", \$BP_COUNT
  info line
  bt
  printf "\n"
  # If we hit cudaGetFuncBySymbol's return (leave), stop
  # cudaGetFuncBySymbol is at offset 0x79600, ret is at 0x797fb
  # Check if current PC is in the range of cudaGetFuncBySymbol
  if \$pc >= 0x79600 && \$pc <= 0x797fb
    printf "  >>> Still inside cudaGetFuncBySymbol\n"
  end
  # If we hit a breakpoint after 0x79810 (= cudaGetKernel), we've left cudaGetFuncBySymbol
  if \$pc > 0x79810
    printf "  >>> Left cudaGetFuncBySymbol region — done!\n"
    set \$BP_COUNT=100
  end
  # Continue if not done
  if \$BP_COUNT < 100
    _log_bp
  end
end

# Start the logging loop
_log_bp
ENDOF

echo "Running under GDB..."
echo ""

# Step 4: Run
LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib gdb -batch -x _gdb_cmds.gdb --args ./_gdb_test "$FATBIN" "$KERNEL" 2>&1

echo ""
echo "================================================================"
echo " Cleaning up..."
echo "================================================================"
rm -f _gdb_test _gdb_test.c _gdb_cmds.gdb
