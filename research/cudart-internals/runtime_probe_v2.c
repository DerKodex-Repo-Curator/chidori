/*
 * runtime_probe_v2.c — Comprehensive CUDA Runtime API state probe
 *
 * Probes internal libcudart.so state at every lifecycle stage:
 *   1. Before any CUDA call
 *   2. After cudaSetDevice(0)
 *   3. After cudaFree(0)  (lazy init trigger)
 *   4. After __cudaRegisterFatBinary
 *   5. After __cudaRegisterFunction
 *   6. After cudaGetFuncBySymbol (if success)
 *
 * Build:
 *   gcc -O0 -g -o runtime_probe_v2 runtime_probe_v2.c \
 *       -I$CUDA_PATH/include -L$CUDA_PATH/targets/x86_64-linux/lib \
 *       -lcuda -lcudart -ldl -lpthread \
 *       -Wl,-rpath,$CUDA_PATH/targets/x86_64-linux/lib
 *
 * Usage:
 *   LD_LIBRARY_PATH=$CUDA_PATH/targets/x86_64-linux/lib \
 *       ./runtime_probe_v2 <fatbin_path> <kernel_name>
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <link.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

/* ---- helpers ---- */

static void *get_libcudart_base(void) {
    void *handle = dlopen("libcudart.so", RTLD_NOW | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libcudart.so.12.8", RTLD_NOW | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libcudart.so.12.8.90", RTLD_NOW | RTLD_NOLOAD);
    if (!handle) {
        /* Find via link map */
        struct link_map *lm;
        FILE *f = fopen("/proc/self/maps", "r");
        if (!f) return NULL;
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "libcudart.so")) {
                unsigned long start;
                if (sscanf(line, "%lx-", &start) == 1) {
                    fclose(f);
                    return (void *)start;
                }
            }
        }
        fclose(f);
        return NULL;
    }
    struct link_map *lm;
    if (dlinfo(handle, RTLD_DI_LINKMAP, &lm) == 0) {
        dlclose(handle);
        return (void *)lm->l_addr;
    }
    dlclose(handle);
    return NULL;
}

/* Get runtime state singleton pointer by calling the internal function */
typedef void *(*getRuntimeState_fn)(void);

static void *get_runtime_state(void) {
    /* The function at offset 0x27f50 returns the runtime state pointer.
       It uses pthread_once for initialization. We call it via dlsym
       of the resolved address. */
    static getRuntimeState_fn fn = NULL;
    if (!fn) {
        /* First try dlsym on the __cudaRegisterFatBinary symbol and use
           its relative position, since getRuntimeState is internal */
        void *base = get_libcudart_base();
        if (!base) {
            fprintf(stderr, "  [ERR] Cannot find libcudart.so base\n");
            return NULL;
        }
        /* __cudaRegisterFatBinary is at offset 0x228c0 in 12.8.90 */
        void *reg_fatbin = dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinary");
        if (reg_fatbin && base) {
            unsigned long actual_offset = (unsigned long)((char *)reg_fatbin - (char *)base);
            printf("  [DEBUG] __cudaRegisterFatBinary at offset 0x%lx (expected 0x228c0)\n",
                   actual_offset);
            /* getRuntimeState is always at __cudaRegisterFatBinary + (0x27f50 - 0x228c0) = +0x5690 */
            fn = (getRuntimeState_fn)((char *)base + actual_offset + 0x5690);
        }
        if (!fn) {
            /* Fallback: use hardcoded offset */
            fn = (getRuntimeState_fn)((char *)base + 0x27f50);
        }
    }
    if (!fn) return NULL;
    void *result = fn();
    printf("  [DEBUG] get_runtime_state() = %p, lifecycle = %d\n", result,
           result ? *(volatile int *)((uint8_t *)result + 0x070) : -1);
    return result;
}

/* checkShuttingDown — returns non-zero if shutting down, 0 if ok */
typedef int (*checkShuttingDown_fn)(void *);

static int check_shutting_down(void *state) {
    if (!state) return 1;
    /* At offset 0x356e0 in CUDA 12.8.90 */
    void *base = get_libcudart_base();
    if (!base) return 1;
    checkShuttingDown_fn fn = (checkShuttingDown_fn)((char *)base + 0x356e0);
    return fn(state);
}

static const char *cuda_error_name(int err) {
    switch (err) {
        case 0: return "SUCCESS";
        case 1: return "INVALID_VALUE";
        case 3: return "NOT_INITIALIZED";
        case 4: return "DEINITIALIZED";
        case 100: return "NO_DEVICE";
        case 200: return "INVALID_IMAGE";
        case 201: return "INVALID_CONTEXT";
        case 209: return "NO_KERNEL_IMAGE_FOR_DEVICE";
        case 98: return "ERROR_98";
        case 99: return "ERROR_99";
        default: {
            static char buf[32];
            snprintf(buf, sizeof(buf), "error_%d", err);
            return buf;
        }
    }
}

/* ---- state printer ---- */

#include <unistd.h>
#include <sys/mman.h>

static int safe_read_ptr(void *addr, void **out) {
    *out = NULL;
    if (!addr) return 0;
    *out = *(volatile void **)addr;
    return 1;
}

static int safe_read_u32(void *addr, uint32_t *out) {
    *out = 0;
    if (!addr) return 0;
    *out = *(volatile uint32_t *)addr;
    return 1;
}

static void print_runtime_state(const char *label, void *state) {
    printf("\n--- %s ---\n", label);

    if (!state) {
        printf("  state = (nil)\n");
        return;
    }

    uint8_t *s = (uint8_t *)state;
    void *base = get_libcudart_base();

    printf("  state ptr: %p\n", state);

    /* Dump first 0x100 bytes as pointers to find vtable and sub-objects */
    printf("  Pointer map (non-NULL):\n");
    for (int off = 0; off < 0xc0; off += 8) {
        void *val = NULL;
        if (safe_read_ptr(s + off, &val) && val) {
            int is_libcudart = base && val > base && val < (void *)((char *)base + 0x100000);
            printf("    [+0x%03x] %p%s\n", off, val,
                   is_libcudart ? " (in libcudart.so)" : "");
        }
    }

    /* Scan for lifecycle state */
    printf("  Small integer fields (potential lifecycle/state):\n");
    for (int off = 0x40; off < 0x100; off += 4) {
        uint32_t val = 0;
        if (safe_read_u32(s + off, &val) && val <= 2 && val != 0) {
            printf("    [+0x%03x] %u\n", off, val);
        }
    }

    /* Known fields from disassembly */
    uint32_t lifecycle = 0;
    safe_read_u32(s + 0x070, &lifecycle);
    printf("  [0x070] lifecycle: %d  (0=uninit, 1=active, 2=shutting_down)\n", lifecycle);

    void *mod_mgr_vtbl = NULL, *func_res_vtbl = NULL, *mod_mgmt_ptr = NULL, *init_mod_ptr = NULL;
    safe_read_ptr(s + 0x098, &mod_mgr_vtbl);
    safe_read_ptr(s + 0x0a0, &func_res_vtbl);
    safe_read_ptr(s + 0x0a8, &mod_mgmt_ptr);
    safe_read_ptr(s + 0x0b0, &init_mod_ptr);

    printf("  [0x098] vtable (module_mgr):  %p\n", mod_mgr_vtbl);
    printf("  [0x0a0] vtable (func_res):   %p\n", func_res_vtbl);
    printf("  [0x0a8] module mgmt ptr:     %p\n", mod_mgmt_ptr);
    printf("  [0x0b0] init_module ptr:     %p\n", init_mod_ptr);

    if (mod_mgmt_ptr) {
        uint8_t *m = (uint8_t *)mod_mgmt_ptr;
        printf("  module mgmt object @ %p:\n", mod_mgmt_ptr);
        void *p = NULL;
        if (safe_read_ptr(m + 0x000, &p)) printf("    [+0x000] vtable:    %p\n", p);
        if (safe_read_ptr(m + 0x008, &p)) printf("    [+0x008] data[0]:   %p\n", p);
        if (safe_read_ptr(m + 0x010, &p)) printf("    [+0x010] data[1]:   %p\n", p);
        uint32_t f = 0;
        if (safe_read_u32(m + 0x540, &f)) printf("    [+0x540] init_flag: %d\n", f);
        if (safe_read_u32(m + 0x544, &f)) printf("    [+0x544] field_544: %d\n", f);
    }

    if (init_mod_ptr) {
        uint8_t *m = (uint8_t *)init_mod_ptr;
        printf("  init_module object @ %p:\n", init_mod_ptr);
        void *p = NULL;
        if (safe_read_ptr(m + 0x000, &p)) printf("    [+0x000] vtable:   %p\n", p);
        uint32_t f = 0;
        if (safe_read_u32(m + 0x004, &f)) printf("    [+0x004] field_4:  %d\n", f);
        if (safe_read_ptr(m + 0x008, &p)) printf("    [+0x008] field_8:  %p\n", p);
    }

    /* Don't call checkShuttingDown — it crashes due to offset mismatch */
    printf("  checkShuttingDown(): skipped (use lifecycle field above)\n");
}

/* ---- vtable dumper ---- */

static void dump_vtable(const char *name, void *vtbl_ptr, int n_entries) {
    if (!vtbl_ptr) {
        printf("  %s vtable: (nil)\n", name);
        return;
    }
    printf("  %s vtable at %p:\n", name, vtbl_ptr);
    void **vtbl = (void **)vtbl_ptr;
    for (int i = 0; i < n_entries && i < 16; i++) {
        printf("    [%2d] %p", i, vtbl[i]);
        if (vtbl[i]) {
            void *base = get_libcudart_base();
            if (base && (char *)vtbl[i] > (char *)base) {
                printf("  (offset +0x%lx)", (unsigned long)((char *)vtbl[i] - (char *)base));
            }
        }
        printf("\n");
    }
}

/* ---- fat binary parser ---- */

static void parse_fatbin(const void *data, size_t size) {
    const uint8_t *p = (const uint8_t *)data;
    if (size < 4) return;

    uint32_t magic = *(uint32_t *)p;
    if (magic != 0x50554166 && magic != 0xba55ed50) {
        printf("  Not a recognized fat binary (magic=0x%08x)\n", magic);
        return;
    }
    printf("  Fat binary magic: 0x%08x (%s)\n", magic,
           magic == 0xba55ed50 ? "cubin" : "ptx");

    /* Try to find entries by scanning for known patterns */
    /* Fat binary header: magic(4) + version(2) + header_size(4) + ... */
    /* Skip the first 16 bytes of the fat binary header, then parse entries */
    /* For 0xba55ed50 format: 4 magic + 2 version + 2 pad + 4 header_size + 4 data_size + ... */

    uint16_t version = *(uint16_t *)(p + 4);
    printf("  Version: %d\n", version);

    if (magic == 0xba55ed50) {
        uint32_t hdr_size = *(uint32_t *)(p + 8);
        uint64_t data_size = *(uint64_t *)(p + 12);
        printf("  Header size: %u, Data size: %lu\n", hdr_size, (unsigned long)data_size);

        /* The actual entries start after the header */
        const uint8_t *entry = p + hdr_size;
        const uint8_t *data_end = p + hdr_size + data_size;
        int entry_idx = 0;

        while (entry < data_end - 32 && entry_idx < 20) {
            uint16_t entry_type = *(uint16_t *)(entry);
            uint16_t sub_flags = *(uint16_t *)(entry + 2);
            uint32_t entry_hdr_size = *(uint32_t *)(entry + 4);
            uint64_t uncomp_size = *(uint64_t *)(entry + 8);
            uint32_t comp_size = *(uint32_t *)(entry + 16);
            uint32_t flags = *(uint32_t *)(entry + 20);
            uint32_t unknown = *(uint32_t *)(entry + 24);
            uint32_t sm_version = 0;
            if (entry_hdr_size >= 32)
                sm_version = *(uint32_t *)(entry + 28);

            int compressed = (flags & 0x40) != 0;
            printf("  Entry %d: type=%u sm=%u uncomp=%lu comp=%u flags=0x%x %s",
                   entry_idx, entry_type, sm_version,
                   (unsigned long)uncomp_size, comp_size, flags,
                   compressed ? "(COMPRESSED)" : "(raw)");

            /* Check for zstd magic */
            const uint8_t *edata = entry + entry_hdr_size;
            if (compressed && entry + entry_hdr_size + 4 <= data_end) {
                if (edata[0] == 0x28 && edata[1] == 0xb5 && edata[2] == 0x2f && edata[3] == 0xfd) {
                    printf(" [zstd]");
                } else if (edata[0] == 0xa2) {
                    printf(" [a2-proprietary]");
                } else {
                    printf(" [unknown-compression: %02x %02x %02x %02x]",
                           edata[0], edata[1], edata[2], edata[3]);
                }
            }
            printf("\n");

            if (entry_hdr_size == 0) break;
            entry += entry_hdr_size + (comp_size ? comp_size : uncomp_size);
            entry_idx++;
        }
    }
}

/* ---- main ---- */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <fatbin> <kernel_name>\n", argv[0]);
        return 1;
    }

    const char *fatbin_path = argv[1];
    const char *kernel_name = argv[2];

    printf("============================================================\n");
    printf("CUDA Runtime API State Probe v2\n");
    printf("============================================================\n");
    printf("Fat binary: %s\n", fatbin_path);
    printf("Kernel name: %s\n", kernel_name);
    printf("GPU: ");
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("%d device(s) found\n", device_count);

    void *base = get_libcudart_base();
    printf("libcudart.so base: %p\n", base);

    /* Load fat binary into memory */
    FILE *f = fopen(fatbin_path, "rb");
    if (!f) { perror("fopen fatbin"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *fatbin_data = malloc(fsize);
    fread(fatbin_data, 1, fsize, f);
    fclose(f);
    printf("Fat binary loaded: %ld bytes\n", fsize);

    parse_fatbin(fatbin_data, fsize);

    /* ---- STAGE 1: Before any CUDA call ---- */
    void *state = get_runtime_state();
    print_runtime_state("STAGE 1: Before any CUDA call", state);

    /* ---- STAGE 2: After cudaSetDevice(0) ---- */
    cudaError_t err = cudaSetDevice(0);
    printf("\n[STAGE 2] cudaSetDevice(0): %s\n", cudaGetErrorString(err));
    state = get_runtime_state();
    print_runtime_state("STAGE 2: After cudaSetDevice(0)", state);

    if (state) {
        dump_vtable("module_mgr", *(void **)((uint8_t *)state + 0x098), 6);
        dump_vtable("func_res", *(void **)((uint8_t *)state + 0x0a0), 6);
    }

    /* ---- STAGE 3: After cudaFree(0) — triggers lazy init ---- */
    err = cudaFree(0);
    printf("\n[STAGE 3] cudaFree(0): %s\n", cudaGetErrorString(err));
    state = get_runtime_state();
    print_runtime_state("STAGE 3: After cudaFree(0)", state);

    /* ---- STAGE 4: After __cudaRegisterFatBinary ---- */
    typedef void *(*RegisterFatBinary_t)(void *);
    RegisterFatBinary_t reg_fatbin = (RegisterFatBinary_t)dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinary");
    if (!reg_fatbin) {
        fprintf(stderr, "Cannot find __cudaRegisterFatBinary\n");
        return 1;
    }

    void *fatbin_handle = reg_fatbin(fatbin_data);
    printf("\n[STAGE 4] __cudaRegisterFatBinary: handle=%p\n", fatbin_handle);
    state = get_runtime_state();
    print_runtime_state("STAGE 4: After __cudaRegisterFatBinary", state);

    /* ---- STAGE 5: After __cudaRegisterFunction ---- */
    typedef void (*RegisterFunction_t)(void *, const char *, void *, const char *,
                                       int, uint3, uint3, size_t, int);
    RegisterFunction_t reg_func = (RegisterFunction_t)dlsym(RTLD_DEFAULT, "__cudaRegisterFunction");
    if (!reg_func) {
        fprintf(stderr, "Cannot find __cudaRegisterFunction\n");
        return 1;
    }

    /* Use the address of a real function as hostFun stub */
    void *real_func = (void *)(uintptr_t)cudaFree;  /* any real function pointer works */
    uint3 gridDim = {1, 1, 1};
    uint3 blockDim = {256, 1, 1};

    reg_func(fatbin_handle, kernel_name, (void *)real_func, kernel_name,
             -1, gridDim, blockDim, 0, 0);
    printf("\n[STAGE 5] __cudaRegisterFunction: stub=%p name='%s'\n", real_func, kernel_name);
    state = get_runtime_state();
    print_runtime_state("STAGE 5: After __cudaRegisterFunction", state);

    /* ---- STAGE 6: cudaGetFuncBySymbol ---- */
    typedef CUresult (*GetFuncBySymbol_t)(CUfunction *, void *);
    GetFuncBySymbol_t get_func = (GetFuncBySymbol_t)dlsym(RTLD_DEFAULT, "cudaGetFuncBySymbol");
    if (!get_func) {
        fprintf(stderr, "Cannot find cudaGetFuncBySymbol\n");
        return 1;
    }

    CUfunction func = NULL;
    int res = get_func(&func, real_func);
    printf("\n[STAGE 6] cudaGetFuncBySymbol: err=%d (%s) func=%p\n",
           res, cuda_error_name(res), func);
    state = get_runtime_state();
    print_runtime_state("STAGE 6: After cudaGetFuncBySymbol", state);

    /* ---- STAGE 6b: Try with __cudaGetKernel ---- */
    typedef CUresult (*GetKernel_t)(CUfunction *, const char *, void *, uint3, uint3, size_t);
    GetKernel_t get_kernel = (GetKernel_t)dlsym(RTLD_DEFAULT, "__cudaGetKernel");
    if (get_kernel) {
        CUfunction func2 = NULL;
        CUresult res2 = get_kernel(&func2, kernel_name, (void *)real_func, gridDim, blockDim, 0);
        printf("\n[STAGE 6b] __cudaGetKernel: err=%d (%s) func=%p\n",
               res2, cuda_error_name(res2), func2);
    }

    /* ---- STAGE 7: Try __cudaLaunchKernel directly ---- */
    if (func) {
        typedef CUresult (*LaunchKernel_t)(const void *, uint3, uint3, void **, size_t, CUstream);
        LaunchKernel_t launch = (LaunchKernel_t)dlsym(RTLD_DEFAULT, "__cudaLaunchKernel");
        if (launch) {
            void *args[] = {NULL};
            CUresult lr = launch(func, gridDim, blockDim, args, 0, NULL);
            printf("\n[STAGE 7] __cudaLaunchKernel: err=%d (%s)\n", lr, cuda_error_name(lr));
        }
    }

    /* ---- STAGE 8: Dump fat binary handle internals ---- */
    if (fatbin_handle) {
        uint8_t *h = (uint8_t *)fatbin_handle;
        printf("\n--- Fat binary handle internals (%p) ---\n", fatbin_handle);
        printf("  [+0x00] ptr: %p\n", *(void **)(h + 0x00));
        printf("  [+0x08] ptr: %p\n", *(void **)(h + 0x08));
        printf("  [+0x10] ptr: %p\n", *(void **)(h + 0x10));
        printf("  [+0x18] ptr: %p\n", *(void **)(h + 0x18));
        printf("  [+0x20] ptr: %p\n", *(void **)(h + 0x20));
        printf("  [+0x28] ptr: %p\n", *(void **)(h + 0x28));
        printf("  [+0x30] ptr: %p\n", *(void **)(h + 0x30));
        printf("  [+0x38] ptr: %p\n", *(void **)(h + 0x38));
        printf("  [+0x40] ptr: %p\n", *(void **)(h + 0x40));
        printf("  [+0x48] ptr: %p\n", *(void **)(h + 0x48));
        printf("  [+0x50] int: %d\n", *(uint32_t *)(h + 0x50));
        printf("  [+0x54] int: %d\n", *(uint32_t *)(h + 0x54));
        printf("  [+0x58] ptr: %p\n", *(void **)(h + 0x58));
        printf("  [+0x60] int: %d\n", *(uint32_t *)(h + 0x60));
        printf("  [+0x64] int: %d\n", *(uint32_t *)(h + 0x64));

        /* The handle from __cudaRegisterFatBinary is a 0x68-byte struct (see allocation at 0x26639) */
        /* __cudaRegisterFunction stores at offset +0x00 (fatbin_ptr) and +0x08 (hostFun) */
        /* in a 0x58-byte struct (see allocation at 0x2791b) */
    }

    /* ---- STAGE 9: Check what the function registration linked-list looks like ---- */
    if (fatbin_handle) {
        uint8_t *h = (uint8_t *)fatbin_handle;
        /* At offset 0x50 and 0x58 there's a linked list of registered functions */
        void *func_list_head = *(void **)(h + 0x50);
        void *func_list_tail = *(void **)(h + 0x58);
        printf("\n--- Registered function list (head=%p, tail=%p) ---\n",
               func_list_head, func_list_tail);

        /* Walk the linked list - each node is 0x58 bytes */
        int count = 0;
        void *node = func_list_head;
        while (node && count < 20) {
            uint8_t *n = (uint8_t *)node;
            void *next = *(void **)(n + 0x50);
            void *prev = *(void **)(n + 0x58);
            void *hostfun = *(void **)(n + 0x00);
            void *name_ptr = *(void **)(n + 0x08);
            const char *name = name_ptr ? (const char *)name_ptr : "(null)";

            printf("  func_node[%d] @ %p: next=%p prev=%p hostFun=%p name='%s'\n",
                   count, node, next, prev, hostfun, name);

            /* Dump more fields */
            printf("    [+0x10] %p [+0x18] %p [+0x20] %p [+0x28] %p\n",
                   *(void **)(n + 0x10), *(void **)(n + 0x18),
                   *(void **)(n + 0x20), *(void **)(n + 0x28));
            printf("    [+0x30] %p [+0x38] %p [+0x40] %p [+0x48] %p\n",
                   *(void **)(n + 0x30), *(void **)(n + 0x38),
                   *(void **)(n + 0x40), *(void **)(n + 0x48));

            if (next == func_list_head) break;  /* circular? */
            node = next;
            count++;
        }
        printf("  Total registered functions: %d\n", count);
    }

    printf("\n============================================================\n");
    printf("PROBE COMPLETE\n");
    printf("============================================================\n");

    free(fatbin_data);
    return 0;
}
