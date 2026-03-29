#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <link.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

extern void **__cudaRegisterFatBinary(void *);
extern void __cudaRegisterFatBinaryEnd(void **);
extern void __cudaRegisterFunction(void **, const char *, char *, char *, int,
    void *, void *, void *, void *, void *);

/* Get libcudart.so base address from the link_map */
static void *get_libcudart_base(void) {
    /* First call something to trigger loading */
    void *dummy;
    cudaMalloc(&dummy, 256);
    cudaFree(dummy);

    /* Now find libcudart in link_map */
    void *handle = dlopen("libcudart.so.13", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) return NULL;

    struct link_map *lm = NULL;
    dlinfo(handle, RTLD_DI_LINKMAP, &lm);
    void *base = (void *)lm->l_addr;

    dlclose(handle);
    return base;
}

/* Read memory at a given offset from libcudart */
static uint64_t read_u64(void *base, uint64_t offset) {
    uint64_t val;
    memcpy(&val, (uint8_t *)base + offset, sizeof(val));
    return val;
}

static uint32_t read_u32(void *base, uint64_t offset) {
    uint32_t val;
    memcpy(&val, (uint8_t *)base + offset, sizeof(val));
    return val;
}

/* Convert absolute address to offset within libcudart */
#define ADDR_TO_OFFSET(base, addr) ((uint64_t)(addr) - (uint64_t)(base))

/* Timestamp helper */
static struct timespec ts_start;
static void ts_mark_start(void) { clock_gettime(CLOCK_MONOTONIC, &ts_start); }
static uint64_t ts_elapsed_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - ts_start.tv_sec) * 1000000000ULL +
           (now.tv_nsec - ts_start.tv_nsec);
}

int main(int argc, char **argv) {
    const char *fatbin_path = argv[1];
    const char *kernel_name = argv[2];
    void *libcudart_base;

    fprintf(stderr, "\n");
    fprintf(stderr, "================================================================\n");
    fprintf(stderr, " PERF Live Trace: Resolving %s from %s\n", kernel_name, fatbin_path);
    fprintf(stderr, "================================================================\n\n");

    /* Get libcudart base */
    libcudart_base = get_libcudart_base();
    if (!libcudart_base) {
        fprintf(stderr, "FATAL: Cannot find libcudart base address\n");
        return 1;
    }
    fprintf(stderr, "[PERF] libcudart base: %p\n", libcudart_base);

    /* Load fat binary from file */
    FILE *f = fopen(fatbin_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fatbin_path); return 1; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    void *fat_data = malloc(size);
    fread(fat_data, 1, size, f);
    fclose(f);

    /* Build wrapper */
    struct { int magic; int version; const void *data; void *filename; } wrapper;
    wrapper.magic = 0x466243b1;
    wrapper.version = 1;
    wrapper.data = fat_data;
    wrapper.filename = NULL;

    /* ===== PHASE 1: Registration ===== */
    fprintf(stderr, "\n--- PHASE 1: Registration ---\n");

    void **handle = __cudaRegisterFatBinary(&wrapper);
    fprintf(stderr, "[PERF] __cudaRegisterFatBinary: handle=%p\n", (void*)handle);
    __cudaRegisterFatBinaryEnd(handle);

    fprintf(stderr, "[PERF] __cudaRegisterFatBinaryEnd done\n");

    /* Create driver context */
    CUcontext ctx;
    CUresult res = cuDevicePrimaryCtxRetain(&ctx, 0);
    fprintf(stderr, "[PERF] cuDevicePrimaryCtxRetain: res=%d ctx=%p\n", res, (void*)ctx);

    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "FATAL: Cannot create context: %d\n", res);
        return 1;
    }

    /* Register function with stub */
    void *stub = malloc(64);
    memset(stub, 0, 64);
    __cudaRegisterFunction(handle, (const char*)stub, (char*)kernel_name,
        (char*)kernel_name, -1, NULL, NULL, NULL, NULL, NULL);
    fprintf(stderr, "[PERF] __cudaRegisterFunction: stub=%p name='%s'\n", stub, kernel_name);

    /* ===== Read runtime state BEFORE cudaGetFuncBySymbol ===== */
    fprintf(stderr, "\n--- PHASE 2: Runtime state before resolution ---\n");

    /* Read getRuntimeState pointer */
    /* getRuntimeState at 0x2a800 loads state from *(base + 0x2b93c8) */
    /* The singleton data is at offsets derived from the disassembly */
    uint64_t state_ptr_addr_offset = 0x2b93c8;  /* GOT entry for runtime state pointer */
    uint64_t runtime_state_ptr = read_u64(libcudart_base, state_ptr_addr_offset);

    fprintf(stderr, "[STATE] *(base+0x%lx) = %p → runtime_state_ptr\n",
            (unsigned long)state_ptr_addr_offset, (void*)runtime_state_ptr);

    if (!runtime_state_ptr) {
        fprintf(stderr, "[STATE] Runtime state pointer is NULL — runtime not initialized?\n");
        /* Force init by calling cudaGetDevice */
        int dev;
        cudaGetDevice(&dev);
        runtime_state_ptr = read_u64(libcudart_base, state_ptr_addr_offset);
        fprintf(stderr, "[STATE] After forced init: runtime_state_ptr=%p\n", (void*)runtime_state_ptr);
    }

    if (!runtime_state_ptr) {
        fprintf(stderr, "[STATE] Still NULL. Trying read from *(base+0x2b93c4) singleton init...\n");
        /* The singleton_once_t is at 0x2b93c4 */
        /* And the state pointer at 0x2b93c8 */
        fprintf(stderr, "[STATE] FATAL: Cannot locate runtime state\n");
        return 1;
    }

    /* Read key state fields */
    uint32_t lifecycle = read_u32((void*)runtime_state_ptr, 0xd0);
    uint32_t shutdown_err = read_u32((void*)runtime_state_ptr, 0xd4);
    uint64_t vtable_module = read_u64((void*)runtime_state_ptr, 0xf8);
    uint64_t vtable_func = read_u64((void*)runtime_state_ptr, 0x100);
    uint64_t mod_mgmt = read_u64((void*)runtime_state_ptr, 0x108);

    uint64_t init_module = read_u64((void*)runtime_state_ptr, 0x110);

    uint32_t init_flag_540 = 0;
    if (mod_mgmt) {
        init_flag_540 = read_u32((void*)mod_mgmt, 0x540);
    }
    uint32_t init_flag_4 = 0;
    if (init_module) {
        init_flag_4 = read_u32((void*)init_module, 0x4);
    }

    fprintf(stderr, "[STATE] lifecycle=%d shutdown_err=%d\n", lifecycle, shutdown_err);
    fprintf(stderr, "[STATE] vtable_module=%p vtable_func=%p\n",
            (void*)vtable_module, (void*)vtable_func);
    fprintf(stderr, "[STATE] mod_mgmt=%p initModule=%p\n",
            (void*)mod_mgmt, (void*)init_module);
    fprintf(stderr, "[STATE] mod_mgmt->0x540=%d  (0=alternative resolver path)\n", init_flag_540);
    fprintf(stderr, "[STATE] initModule->0x4=%d\n", init_flag_4);

    /* ===== PHASE 3: THE CRITICAL CALL — cudaGetFuncBySymbol ===== */
    fprintf(stderr, "\n--- PHASE 3: Calling cudaGetFuncBySymbol ---\n");

    /* Read the function pointer at 0x2b91e0 (the indirect resolver) */
    uint64_t resolver_fn = read_u64(libcudart_base, 0x2b91e0);
    fprintf(stderr, "[STATE] *(base+0x2b91e0) = %p (core resolution indirect call target)\n",
            (void*)resolver_fn);

    /* Read the function pointer at 0x2b91b0 (used by cudaFuncGetAttributes) */
    uint64_t attr_resolver_fn = read_u64(libcudart_base, 0x2b91b0);
    fprintf(stderr, "[STATE] *(base+0x2b91b0) = %p (attribute query resolver)\n",
            (void*)attr_resolver_fn);

    ts_mark_start();

    CUfunction func;
    cudaError_t rt_err = cudaGetFuncBySymbol(&func, stub);

    uint64_t elapsed = ts_elapsed_ns();

    fprintf(stderr, "\n--- PHASE 4: Result ---\n");
    fprintf(stderr, "[PERF] cudaGetFuncBySymbol returned in %lu ns\n", elapsed);
    fprintf(stderr, "[PERF] err=%d (%s) func=%p\n",
            (int)rt_err, cudaGetErrorString(rt_err), (void*)func);

    /* ===== PHASE 5: Verify state AFTER resolution ===== */
    fprintf(stderr, "\n--- PHASE 5: State after resolution ---\n");

    uint32_t lifecycle_after = read_u32((void*)runtime_state_ptr, 0xd0);
    uint64_t mod_mgmt_after = read_u64((void*)runtime_state_ptr, 0x108);
    uint32_t init_flag_540_after = 0;
    if (mod_mgmt_after) {
        init_flag_540_after = read_u32((void*)mod_mgmt_after, 0x540);
    }

    fprintf(stderr, "[STATE] lifecycle=%d mod_mgmt=%p\n", lifecycle_after, (void*)mod_mgmt_after);
    fprintf(stderr, "[STATE] mod_mgmt->0x540=%d  (was %d before)\n",
            init_flag_540_after, init_flag_540);

    /* ===== VERDICT ===== */
    fprintf(stderr, "\n");
    fprintf(stderr, "================================================================\n");
    fprintf(stderr, " VERDICT\n");
    fprintf(stderr, "================================================================\n");
    if (init_flag_540 == 0) {
        fprintf(stderr, " 0x540 field stayed 0 throughout → alternative resolver path taken\n");
        fprintf(stderr, " Proof: cudaGetFuncBySymbol took the DEINIT path at 0x79800\n");
        fprintf(stderr, "        → calls 0x23160 (alternative resolver) directly\n");
        fprintf(stderr, "        → NO vtable dispatches on state->0x100 or state->0xf8\n");
    } else {
        fprintf(stderr, " 0x540 field = %d → NORMAL path with vtable dispatches\n", init_flag_540);
        fprintf(stderr, "        → vtable[2] on state->0x100, then 0x23160, then vtable[2] cleanup\n");
    }
    if (rt_err == cudaSuccess && func != NULL) {
        fprintf(stderr, " Resolution: SUCCESS (func=%p)\n", (void*)func);
    } else {
        fprintf(stderr, " Resolution: FAILED (err=%d)\n", (int)rt_err);
    }
    fprintf(stderr, "================================================================\n");

    return 0;
}
