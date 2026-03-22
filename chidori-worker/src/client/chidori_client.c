/*
 * Chidori CUDA Client Interception Library (libchidori.so)
 *
 * This shared library is loaded via LD_PRELOAD to intercept CUDA Driver API
 * calls. Each intercepted call is serialized and sent over TCP to the
 * chidori-worker running on the remote GPU host.
 *
 * The library reads connection info from CHIDORI_CFG_OVERRIDE env var
 * (set by juicify) to know which server to connect to.
 *
 * Phase 0: Stub implementation with basic protocol integration.
 * Phase 2: Full CUDA Driver API interception via codegen.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <limits.h>
#ifndef RTLD_DEEPBIND
#define RTLD_DEEPBIND 0  /* Not available on this platform */
#endif

#include "chidori_protocol.h"

/*
 * ============================================================================
 * dlopen interception — redirect libcuda.so.1 to real driver for cuBLAS.
 *
 * cuBLAS resolves CUDA symbols via dlopen("libcuda.so.1") + dlsym().
 * With LD_LIBRARY_PATH=build, this finds our symlink and gets our
 * intercepted functions. This causes cublasCreate to fail (error 14)
 * because cuBLAS's init handshake doesn't work through our routing.
 *
 * Fix: intercept dlopen and redirect "libcuda.so" requests to the real
 * NVIDIA driver handle (loaded by init_real_cuda). cuBLAS gets real
 * driver functions via dlsym, completely bypassing our interception.
 * ============================================================================
 */
static void *g_real_cuda_handle_for_dlopen = NULL; /* set by init_real_cuda */
static __thread int g_dlopen_redirect_active = 0;  /* 1 = redirect libcuda dlopen to real driver */
static __thread int g_in_real_cuda = 0;   /* reentrance guard: >0 = inside real CUDA call, forward to real driver */

/*
 * ============================================================================
 * dlsym bootstrap — get the REAL dlsym before we override it.
 *
 * We use dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5") to obtain the real
 * dlsym function pointer. We can't use dlsym(RTLD_NEXT, "dlsym") from
 * within our own dlsym interceptor — that would be infinite recursion.
 * All internal dlsym calls in this file go through __real_dlsym().
 * ============================================================================
 */
static void *(*g_real_dlsym_fn)(void *, const char *) = NULL;
static void *(*g_real_dlvsym_fn)(void *, const char *, const char *) = NULL;

/* Get the real glibc dlvsym via _dl_sym or linker tricks.
 * Called once at startup BEFORE our dlvsym override is active. */
extern void *_dl_sym(void *, const char *, void *) __attribute__((weak));

static void init_real_dlsym(void) {
    if (g_real_dlsym_fn) return;
    /* First, get glibc's real dlvsym so we can resolve versioned dlsym.
     * We can't call dlvsym() directly because our override intercepts it.
     * Use _dl_sym (glibc internal, weak) as bootstrap. */
    if (_dl_sym) {
        g_real_dlvsym_fn = (void *(*)(void *, const char *, const char *))
            _dl_sym(RTLD_NEXT, "dlvsym", (void *)init_real_dlsym);
    }
    if (g_real_dlvsym_fn) {
        g_real_dlsym_fn = (void *(*)(void *, const char *))
            g_real_dlvsym_fn(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    if (!g_real_dlsym_fn) {
        /* Last resort: try _dl_sym for dlsym directly */
        if (_dl_sym) {
            g_real_dlsym_fn = (void *(*)(void *, const char *))
                _dl_sym(RTLD_NEXT, "dlsym", (void *)init_real_dlsym);
        }
    }
    if (!g_real_dlsym_fn) {
        fprintf(stderr, "[chidori] FATAL: could not bootstrap real dlsym\n");
    }
}

/* Safe wrapper: all internal dlsym calls in this file MUST use this. */
static void *__real_dlsym(void *handle, const char *symbol) {
    if (!g_real_dlsym_fn) init_real_dlsym();
    if (g_real_dlsym_fn) return g_real_dlsym_fn(handle, symbol);
    return NULL;
}

void *dlopen(const char *filename, int flags) {
    static void *(*real_dlopen)(const char *, int) = NULL;
    if (!real_dlopen) {
        real_dlopen = __real_dlsym(RTLD_NEXT, "dlopen");
    }

    /* Redirect libcuda.so.1 dlopen to real driver when:
     * 1. Explicitly enabled (g_dlopen_redirect_active, during cuBLAS create), OR
     * 2. Inside a real CUDA call (g_in_real_cuda > 0, during cuBLAS GemmEx etc.)
     * This ensures cuBLAS gets real driver functions for ALL operations,
     * not just initialization. Needed for fp16 GemmEx which lazily loads kernels. */
    if ((g_dlopen_redirect_active || g_in_real_cuda > 0) &&
        g_real_cuda_handle_for_dlopen && filename &&
        (strcmp(filename, "libcuda.so.1") == 0 ||
         strcmp(filename, "libcuda.so") == 0)) {
        return g_real_cuda_handle_for_dlopen;
    }

    return real_dlopen ? real_dlopen(filename, flags) : NULL;
}

/*
 * ============================================================================
 * dlsym interception — return OUR cuBLAS/cublasLt functions when PyTorch
 * resolves them via dlsym(cublas_handle, "cublasGemmEx").
 *
 * Without this, PyTorch's dlsym bypasses LD_PRELOAD entirely and gets
 * real cuBLAS function pointers. Those real functions don't know about
 * our local/remote device routing, so FP16 matmul on the local device
 * fails with CUBLAS_STATUS_NOT_INITIALIZED.
 *
 * When g_in_real_cuda > 0, we do NOT intercept — cuBLAS internal
 * operations need real function pointers to work correctly.
 * ============================================================================
 */
static const char *g_intercepted_cublas_symbols[] = {
    /* cuBLAS core */
    "cublasCreate_v2", "cublasDestroy_v2",
    "cublasSetStream_v2", "cublasGetStream_v2",
    "cublasSetMathMode", "cublasGetMathMode",
    "cublasGetVersion_v2",
    "cublasSetWorkspace_v2",
    "cublasSetPointerMode_v2", "cublasGetPointerMode_v2",
    "cublasSetAtomicsMode", "cublasGetAtomicsMode",
    "cublasLoggerConfigure",
    "cublasGetStatusString",
    "cublasGetProperty",
    /* GEMM family */
    "cublasSgemm_v2", "cublasDgemm_v2", "cublasHgemm",
    "cublasSgemmEx", "cublasGemmEx",
    "cublasSgemmBatched", "cublasSgemmStridedBatched",
    "cublasGemmStridedBatchedEx", "cublasGemmBatchedEx",
    /* BLAS2 */
    "cublasSgemv_v2", "cublasDgemv_v2",
    /* BLAS1 */
    "cublasSaxpy_v2", "cublasDaxpy_v2",
    "cublasSnrm2_v2", "cublasDnrm2_v2",
    "cublasSdot_v2", "cublasDdot_v2",
    "cublasSscal_v2", "cublasDscal_v2",
    "cublasScopy_v2", "cublasDcopy_v2",
    /* cublasLt */
    "cublasLtCreate", "cublasLtDestroy",
    "cublasLtMatmul",
    "cublasLtMatmulDescCreate", "cublasLtMatmulDescDestroy",
    "cublasLtMatmulDescSetAttribute", "cublasLtMatmulDescGetAttribute",
    "cublasLtMatrixLayoutCreate", "cublasLtMatrixLayoutDestroy",
    "cublasLtMatrixLayoutSetAttribute", "cublasLtMatrixLayoutGetAttribute",
    "cublasLtMatmulPreferenceCreate", "cublasLtMatmulPreferenceDestroy",
    "cublasLtMatmulPreferenceSetAttribute",
    "cublasLtMatmulAlgoGetHeuristic",
    "cublasLtMatmulAlgoInit",
    "cublasLtMatmulAlgoConfigSetAttribute", "cublasLtMatmulAlgoConfigGetAttribute",
    "cublasLtMatmulAlgoCheck", "cublasLtMatmulAlgoGetIds",
    "cublasLtGetCudartVersion", "cublasLtGetVersion",
    "cublasLtGetStatusName", "cublasLtGetStatusString",
    "cublasLtGetProperty",
    "cublasLtLoggerSetCallback", "cublasLtLoggerSetLevel",
    "cublasLtLoggerSetMask", "cublasLtLoggerSetFile",
    NULL
};

void *dlsym(void *handle, const char *symbol) {
    /* Bootstrap: ensure we have the real dlsym */
    if (!g_real_dlsym_fn) init_real_dlsym();

    /* If inside a real CUDA/cuBLAS call (reentrance), do NOT intercept.
     * cuBLAS internal operations need to resolve their own symbols to
     * real implementations, not our routing wrappers. */
    if (g_in_real_cuda > 0) {
        return __real_dlsym(handle, symbol);
    }

    /* Check if this is a cuBLAS/cublasLt symbol we intercept */
    if (symbol) {
        for (int i = 0; g_intercepted_cublas_symbols[i] != NULL; i++) {
            if (strcmp(symbol, g_intercepted_cublas_symbols[i]) == 0) {
                /* Return OUR version so local/remote routing works */
                void *ours = __real_dlsym(RTLD_DEFAULT, symbol);
                if (ours) {
                    fprintf(stderr, "[chidori-dlsym] %s → OUR %p (handle=%p)\n", symbol, ours, handle);
                    return ours;
                }
                fprintf(stderr, "[chidori-dlsym] %s MATCH but RTLD_DEFAULT returned NULL!\n", symbol);
                break;  /* symbol matched but we couldn't find ours — fall through */
            }
        }
    }

    /* All other symbols: return the real result */
    return __real_dlsym(handle, symbol);
}

/* Also intercept dlvsym — PyTorch may use versioned symbol resolution */
void *dlvsym(void *handle, const char *symbol, const char *version) {
    /* Use the real dlvsym stored during bootstrap */
    if (!g_real_dlvsym_fn) init_real_dlsym();

    /* Same interception logic as dlsym for cuBLAS symbols */
    if (g_in_real_cuda == 0 && symbol) {
        for (int i = 0; g_intercepted_cublas_symbols[i] != NULL; i++) {
            if (strcmp(symbol, g_intercepted_cublas_symbols[i]) == 0) {
                void *ours = __real_dlsym(RTLD_DEFAULT, symbol);
                if (ours) {
                    fprintf(stderr, "[chidori-dlvsym] %s@%s → OUR %p\n", symbol, version ? version : "?", ours);
                    return ours;
                }
                break;
            }
        }
    }

    return g_real_dlvsym_fn ? g_real_dlvsym_fn(handle, symbol, version) : NULL;
}

/*
 * ============================================================================
 * Centralized logging — controlled by environment variables:
 *   CHIDORI_DEBUG=1    — Connection events, device routing, errors
 *   CHIDORI_TRACE=1    — Every sync/async RPC with func_id and timing
 *   CHIDORI_VERBOSE=1  — Per-RPC payload/result details
 * DEBUG=1 implies VERBOSE=1. All output goes to stderr.
 * ============================================================================
 */
static int g_debug = -1;
static int g_log_trace = -1;
static int g_log_verbose = -1;

static void check_log_levels(void) {
    if (g_debug == -1) {
        const char *d = getenv("CHIDORI_DEBUG");
        g_debug = (d && d[0] == '1') ? 1 : 0;
    }
    if (g_log_trace == -1) {
        const char *v = getenv("CHIDORI_TRACE");
        g_log_trace = (v && v[0] == '1') ? 1 : 0;
    }
    if (g_log_verbose == -1) {
        const char *v = getenv("CHIDORI_VERBOSE");
        g_log_verbose = (v && v[0] == '1') ? 1 : 0;
    }
    if (g_debug && !g_log_verbose) g_log_verbose = 1;
}

#define LOG_DEBUG(fmt, ...) do { \
    if (__builtin_expect(g_debug > 0, 0)) \
        fprintf(stderr, "[chidori] " fmt "\n", ##__VA_ARGS__); \
} while(0)

#define LOG_TRACE(fmt, ...) do { \
    if (__builtin_expect(g_log_trace > 0, 0)) { \
        struct timeval _tv; gettimeofday(&_tv, NULL); \
        fprintf(stderr, "[TRACE %ld.%03ld] " fmt "\n", \
                (long)_tv.tv_sec % 1000, (long)_tv.tv_usec / 1000, ##__VA_ARGS__); \
    } \
} while(0)

#define LOG_VERBOSE(fmt, ...) do { \
    if (__builtin_expect(g_log_verbose > 0, 0)) \
        fprintf(stderr, "[chidori-rpc] " fmt "\n", ##__VA_ARGS__); \
} while(0)

/* CUDA types (minimal definitions to avoid requiring cuda.h on client) */
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_INVALID_DEVICE 101
#define CUDA_ERROR_NOT_FOUND 500
#define CUDA_ERROR_NOT_SUPPORTED 801

/*
 * ============================================================================
 * Partial Offloading — Real CUDA passthrough for local GPU devices.
 *
 * When CHIDORI_LOCAL_DEVICES is set (e.g., "0"), device 0 uses the real
 * CUDA driver (loaded via dlsym) and all other devices route to the remote
 * Chidori server. When unset, ALL devices route to the remote server
 * (original behavior).
 * ============================================================================
 */

/* Real CUDA function dispatch table — loaded lazily via dlsym */
static void *g_real_cuda_handle = NULL;   /* dlopen handle for real libcuda.so */
static void *g_real_cudart_handle = NULL; /* dlopen handle for real libcudart.so */
static void *g_real_cublas_handle = NULL; /* dlopen handle for real libcublas.so */
static int g_partial_offload = 0;         /* 1 if CHIDORI_LOCAL_DEVICES is set */
static int g_fatbin_passthrough = -1;    /* 1 if CHIDORI_FATBIN_PASSTHROUGH=1: skip all our fatbin tracking */
static int g_device_count_cache = -1;    /* cached device count, -1 = not cached */
static int g_real_cuda_inited = 0;       /* 1 after real cuInit(0) called */
static int g_current_device = 0;         /* current active device (global) */

/*
 * Deferred init mode (CHIDORI_DEFERRED_INIT=1):
 * Delays init_real_cuda() until cuInit() is called, instead of calling it
 * from __cudaRegisterFatBinary during early library loading. This ensures
 * the real CUDA driver is only initialized AFTER the normal CUDA init sequence
 * starts, which may fix cuBLAS FP16 issues caused by early initialization.
 *
 * During deferred mode, __cudaRegisterFatBinary/End/Function calls store
 * their arguments in pending arrays. When cuInit fires, it calls
 * init_real_cuda() and then replays all pending registrations.
 */
static int g_deferred_init = -1;         /* -1=unchecked, 0=off, 1=on */
static int g_deferred_replay_done = 0;   /* 1 after pending fatbins replayed */

static inline int is_deferred_init(void) {
    if (g_deferred_init == -1) {
        const char *v = getenv("CHIDORI_DEFERRED_INIT");
        g_deferred_init = (v && v[0] == '1') ? 1 : 0;
        if (g_deferred_init)
            fprintf(stderr, "[chidori] DEFERRED INIT MODE: init_real_cuda() will be deferred until cuInit()\n");
    }
    return g_deferred_init;
}

/* Pending fat binary registrations for deferred replay */
#define MAX_PENDING_FATBINS 1024
static struct {
    void *fatCubin;     /* original fatCubin pointer passed to __cudaRegisterFatBinary */
    int idx;            /* index into g_fat_binaries where we stored the parsed data */
} g_pending_fatbins[MAX_PENDING_FATBINS];
static int g_pending_fatbin_count = 0;

/* Pending __cudaRegisterFatBinaryEnd calls */
static int g_pending_fatbin_ends[MAX_PENDING_FATBINS]; /* fat_binary index */
static int g_pending_fatbin_end_count = 0;

/* Pending __cudaRegisterFunction calls */
#define MAX_PENDING_FUNCS 65536
static struct {
    int fat_idx;                /* index into g_fat_binaries */
    const char *hostFun;
    char *deviceFun;
    const char *deviceName;
    int thread_limit;
    void *tid;
    void *bid;
    void *bDim;
    void *gDim;
    int *wSize;
} g_pending_funcs[MAX_PENDING_FUNCS];
static int g_pending_func_count = 0;

/* Pointer-to-device tracker for cuPointerGetAttribute cross-device queries */
#define MAX_ALLOC_TRACK 16384
static struct { uintptr_t ptr; size_t size; int device; } g_alloc_track[MAX_ALLOC_TRACK];
static int g_alloc_track_count = 0;
static pthread_mutex_t g_alloc_track_lock = PTHREAD_MUTEX_INITIALIZER;

static void alloc_track_add(uintptr_t ptr, size_t size, int device) {
    pthread_mutex_lock(&g_alloc_track_lock);
    if (g_alloc_track_count < MAX_ALLOC_TRACK) {
        g_alloc_track[g_alloc_track_count].ptr = ptr;
        g_alloc_track[g_alloc_track_count].size = size;
        g_alloc_track[g_alloc_track_count].device = device;
        g_alloc_track_count++;
    }
    pthread_mutex_unlock(&g_alloc_track_lock);
}

static int alloc_track_device(uintptr_t ptr) {
    pthread_mutex_lock(&g_alloc_track_lock);
    for (int i = g_alloc_track_count - 1; i >= 0; i--) {
        if (ptr >= g_alloc_track[i].ptr &&
            ptr < g_alloc_track[i].ptr + g_alloc_track[i].size) {
            pthread_mutex_unlock(&g_alloc_track_lock);
            return g_alloc_track[i].device;
        }
    }
    pthread_mutex_unlock(&g_alloc_track_lock);
    return -1; /* unknown */
}
static uint32_t g_local_device_mask = 0;  /* bitmask: bit N = device N is local */
static int g_local_device_count = 0;      /* number of local GPUs */
/* g_in_real_cuda defined above (before dlopen interceptor) */
static CUcontext g_local_ctx = NULL;      /* Real CUDA primary context for local device */

/* Real CUDA function pointers (populated by init_real_cuda) */
static struct {
    /* Driver API — all functions that real cudart may call internally */
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDriverGetVersion)(int *);
    CUresult (*cuDeviceGetCount)(int *);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetName)(char *, int, CUdevice);
    CUresult (*cuDeviceTotalMem_v2)(size_t *, CUdevice);
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*cuCtxCreate_v2)(CUcontext *, unsigned int, CUdevice);
    CUresult (*cuCtxDestroy_v2)(CUcontext);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuCtxGetCurrent)(CUcontext *);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*cuDevicePrimaryCtxRelease_v2)(CUdevice);
    CUresult (*cuMemAlloc_v2)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree_v2)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
    CUresult (*cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
    CUresult (*cuMemcpyDtoD_v2)(CUdeviceptr, CUdeviceptr, size_t);
    CUresult (*cuMemGetInfo_v2)(size_t *, size_t *);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleUnload)(CUmodule);
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*cuFuncGetAttribute)(int *, int, CUfunction);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, CUstream, void **, void **);
    CUresult (*cuStreamCreate)(CUstream *, unsigned int);
    CUresult (*cuStreamDestroy_v2)(CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuEventCreate)(CUevent *, unsigned int);
    CUresult (*cuEventDestroy_v2)(CUevent);
    CUresult (*cuEventRecord)(CUevent, CUstream);
    CUresult (*cuEventSynchronize)(CUevent);
    CUresult (*cuEventElapsedTime)(float *, CUevent, CUevent);
    /* Internal driver API — used by cudart for initialization handshake */
    CUresult (*cuGetExportTable)(const void **, const void *);
    /* Runtime API */
    int (*cudaSetDevice)(int);
    int (*cudaGetDevice)(int *);
    int (*cudaGetDeviceCount)(int *);
    int (*cudaMalloc)(void **, size_t);
    int (*cudaFree)(void *);
    int (*cudaMemcpy)(void *, const void *, size_t, int);
    int (*cudaDeviceSynchronize)(void);
    int (*cudaStreamSynchronize)(void *);
    void *cudaLaunchKernel_ptr; /* raw function pointer, cast at call site */
    /* Fat binary / kernel registration */
    void **(*__cudaRegisterFatBinary)(void *);
    void (*__cudaRegisterFatBinaryEnd)(void **);
    void (*__cudaUnregisterFatBinary)(void **);
    void (*__cudaRegisterFunction)(void **, const char *, char *, const char *,
                                   int, void *, void *, void *, void *, int *);
} g_real = {0};

static int g_real_cuda_initialized = 0;

/* Forward declaration — defined after fat binary data structures */
static void replay_deferred_registrations(void);

static void init_real_cuda(void) {
    if (g_real_cuda_initialized) return;
    g_real_cuda_initialized = 1;

    const char *env = getenv("CHIDORI_LOCAL_DEVICES");
    fprintf(stderr, "[chidori] init_real_cuda: CHIDORI_LOCAL_DEVICES=%s\n", env ? env : "(not set)");
    if (!env) return; /* No partial offloading — all remote */

    g_partial_offload = 1;
    g_device_count_cache = -1; /* Invalidate — will be recomputed with local+remote */

    /* Parse local device mask (e.g., "0" or "0,1") */
    const char *p = env;
    while (*p) {
        int dev = atoi(p);
        if (dev >= 0 && dev < 32)
            g_local_device_mask |= (1u << dev);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }

    /* Load real CUDA driver via explicit dlopen.
     * We CANNOT use RTLD_NEXT because we ARE libcuda.so.1 via LD_PRELOAD,
     * so RTLD_NEXT resolves back to ourselves. Instead, we dlopen the real
     * system libcuda.so using its versioned path. We use RTLD_LOCAL to avoid
     * replacing PLT entries in PyTorch's libtorch_cuda.so.
     * For cudart, RTLD_NEXT works since we don't replace libcudart. */
    {
        /* Try to find and dlopen the real CUDA driver by versioned path */
        static const char *cuda_paths[] = {
            "/usr/lib/libcuda.so.1",          /* Standard Linux */
            "/usr/lib64/libcuda.so.1",         /* Some distros */
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1", /* Debian/Ubuntu multiarch */
            NULL
        };
        g_real_cuda_handle = NULL;
        for (int i = 0; cuda_paths[i]; i++) {
            /* Resolve symlinks to get the versioned .so (e.g., libcuda.so.590.48.01)
             * This avoids dlopen loading our own libcuda.so.1 symlink. */
            char resolved[PATH_MAX];
            if (realpath(cuda_paths[i], resolved) && strcmp(resolved, cuda_paths[i]) != 0) {
                g_real_cuda_handle = dlopen(resolved, RTLD_LAZY | RTLD_LOCAL);
                if (g_real_cuda_handle) {
                    fprintf(stderr, "[chidori] Loaded real CUDA driver from %s (resolved: %s)\n", cuda_paths[i], resolved);
                    break;
                }
            }
            /* Also try the direct path */
            g_real_cuda_handle = dlopen(cuda_paths[i], RTLD_LAZY | RTLD_LOCAL);
            if (g_real_cuda_handle) {
                fprintf(stderr, "[chidori] Loaded real CUDA driver from %s\n", cuda_paths[i]);
                break;
            }
        }
        if (!g_real_cuda_handle) {
            fprintf(stderr, "[chidori] WARNING: Could not load real CUDA driver — local GPU passthrough disabled\n");
            g_partial_offload = 0;
            return;
        }
    }
    /* Enable dlopen interception: cuBLAS's dlopen("libcuda.so.1") returns real driver.
     * Also permanently enable the redirect — ALL dlopen("libcuda.so.1") calls after
     * init will get the real driver. This is safe because our LD_PRELOAD'd symbols
     * still intercept PLT calls, and cuGetProcAddress returns our functions for routing.
     * Libraries that use dlsym on the real driver handle get real driver functions,
     * which is correct for local device operations. */
    g_real_cuda_handle_for_dlopen = g_real_cuda_handle;

    /* For cudart: use RTLD_NEXT since we don't replace libcudart */
    g_real_cudart_handle = RTLD_NEXT;

    /* Populate dispatch table — Driver API from dlopen'd handle, Runtime API from RTLD_NEXT */
    #define LOAD_SYM(handle, name) g_real.name = __real_dlsym(handle, #name)
    #define LOAD_CUDA(name) LOAD_SYM(g_real_cuda_handle, name)
    #define LOAD_CUDART(name) LOAD_SYM(RTLD_NEXT, name)

    LOAD_CUDA(cuInit);
    LOAD_CUDA(cuDriverGetVersion);
    LOAD_CUDA(cuDeviceGetCount);
    LOAD_CUDA(cuDeviceGet);
    LOAD_CUDA(cuDeviceGetName);
    LOAD_CUDA(cuDeviceTotalMem_v2);
    LOAD_CUDA(cuDeviceGetAttribute);
    LOAD_CUDA(cuCtxCreate_v2);
    LOAD_CUDA(cuCtxDestroy_v2);
    LOAD_CUDA(cuCtxSetCurrent);
    LOAD_CUDA(cuCtxGetCurrent);
    LOAD_CUDA(cuCtxSynchronize);
    LOAD_CUDA(cuDevicePrimaryCtxRetain);
    LOAD_CUDA(cuDevicePrimaryCtxRelease_v2);
    LOAD_CUDA(cuMemAlloc_v2);
    LOAD_CUDA(cuMemFree_v2);
    LOAD_CUDA(cuMemcpyHtoD_v2);
    LOAD_CUDA(cuMemcpyDtoH_v2);
    LOAD_CUDA(cuMemcpyDtoD_v2);
    LOAD_CUDA(cuMemGetInfo_v2);
    LOAD_CUDA(cuModuleLoadData);
    LOAD_CUDA(cuModuleUnload);
    LOAD_CUDA(cuModuleGetFunction);
    LOAD_CUDA(cuFuncGetAttribute);
    LOAD_CUDA(cuLaunchKernel);
    LOAD_CUDA(cuStreamCreate);
    LOAD_CUDA(cuStreamDestroy_v2);
    LOAD_CUDA(cuStreamSynchronize);
    LOAD_CUDA(cuEventCreate);
    LOAD_CUDA(cuEventDestroy_v2);
    LOAD_CUDA(cuEventRecord);
    LOAD_CUDA(cuEventSynchronize);
    LOAD_CUDA(cuEventElapsedTime);
    LOAD_CUDA(cuGetExportTable);
    LOAD_CUDART(cudaSetDevice);
    LOAD_CUDART(cudaGetDevice);
    LOAD_CUDART(cudaGetDeviceCount);
    LOAD_CUDART(cudaMalloc);
    LOAD_CUDART(cudaFree);
    LOAD_CUDART(cudaMemcpy);
    LOAD_CUDART(cudaDeviceSynchronize);
    LOAD_CUDART(cudaStreamSynchronize);
    if (g_real_cudart_handle) g_real.cudaLaunchKernel_ptr = __real_dlsym(g_real_cudart_handle, "cudaLaunchKernel");
    LOAD_CUDART(__cudaRegisterFatBinary);
    LOAD_CUDART(__cudaRegisterFatBinaryEnd);
    LOAD_CUDART(__cudaUnregisterFatBinary);
    LOAD_CUDART(__cudaRegisterFunction);

    #undef LOAD_SYM
    #undef LOAD_CUDA
    #undef LOAD_CUDART

    /* Count local devices WITHOUT calling cuInit on the real driver.
     * cuInit will be called lazily when cudaSetDevice(local_device) is first called.
     * We must NOT call g_real.cuInit here because PyTorch's _cuda_init() hasn't
     * happened yet, and calling cuInit on the real driver would set PyTorch's
     * internal device count to just the local GPUs (missing the remote ones). */
    /* Count local GPUs by checking which bits are set in the mask */
    for (int i = 0; i < 32; i++) {
        if (g_local_device_mask & (1u << i))
            g_local_device_count++;
    }
    LOG_DEBUG("Partial offload: local_devices=%s (%d local GPUs)", env, g_local_device_count);
    LOG_DEBUG("init_real_cuda: CHIDORI_LOCAL_DEVICES=%s", env);

    /* In deferred init mode, replay all pending fat binary registrations now.
     * init_real_cuda() was deferred until a normal CUDA API call (cuDeviceGetCount, etc.)
     * instead of being called from __cudaRegisterFatBinary during library loading.
     * The real CUDA driver and dispatch table are now loaded, so we can register
     * all the fat binaries that were queued during the deferred period. */
    if (is_deferred_init() && !g_deferred_replay_done) {
        replay_deferred_registrations();
    }
}

static inline int is_local_device(int device) {
    return g_partial_offload && ((g_local_device_mask >> device) & 1);
}

static inline int is_fatbin_passthrough(void) {
    if (g_fatbin_passthrough == -1) {
        const char *v = getenv("CHIDORI_FATBIN_PASSTHROUGH");
        g_fatbin_passthrough = (v && v[0] == '1') ? 1 : 0;
    }
    return g_fatbin_passthrough;
}

/*
 * Lazy-load real cuBLAS — called at first cuBLAS function entry on a local device.
 * We can't load cuBLAS in init_real_cuda() because PyTorch hasn't loaded it yet
 * at that point. By the time a cuBLAS function is called, PyTorch's libtorch_cuda.so
 * has already dlopen'd libcublas.so, so RTLD_NOLOAD will find it.
 */
static int g_cublas_load_attempted = 0;
static inline void *ensure_real_cublas(void) {
    if (g_real_cublas_handle) return g_real_cublas_handle;
    if (g_cublas_load_attempted) return NULL;
    g_cublas_load_attempted = 1;

    /* cuBLAS resolves CUDA symbols via dlopen("libcuda.so.1") + dlsym() at runtime.
     * We enable the dlopen redirect DURING cuBLAS loading so cuBLAS's internal
     * dlopen("libcuda.so.1") returns the real driver. This ensures cuBLAS caches
     * real driver function pointers, not our interceptors — needed for GemmEx fp16
     * which does lazy kernel loading through cached pointers.
     *
     * Use PyTorch's already-loaded cuBLAS (RTLD_NOLOAD) first. If cuBLAS was
     * already loaded before our redirect was active, force a re-resolution by
     * loading from system path with redirect active. */
    g_dlopen_redirect_active = 1;
    g_real_cublas_handle = dlopen("libcublas.so.12", RTLD_LAZY | RTLD_NOLOAD);
    if (!g_real_cublas_handle)
        g_real_cublas_handle = dlopen("libcublas.so", RTLD_LAZY | RTLD_NOLOAD);

    /* Fallback: try system cuBLAS paths (with redirect active) */
    if (!g_real_cublas_handle) {
        static const char *cublas_paths[] = {
            "/opt/cuda/lib64/libcublas.so",
            "/usr/local/cuda/lib64/libcublas.so",
            "/usr/lib/x86_64-linux-gnu/libcublas.so",
            "/usr/lib64/libcublas.so",
            NULL
        };
        for (int i = 0; cublas_paths[i] && !g_real_cublas_handle; i++) {
            g_real_cublas_handle = dlopen(cublas_paths[i], RTLD_LAZY);
        }
    }
    g_dlopen_redirect_active = 0;
    if (g_real_cublas_handle)
        LOG_DEBUG("Loaded real cuBLAS: %p", g_real_cublas_handle);
    else
        LOG_DEBUG("WARNING: Could not load real cuBLAS — local device cuBLAS calls will fall through to RPC");
    return g_real_cublas_handle;
}

/* Client state */
static int g_server_fd = -1;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t g_seq_num = 0;
static int g_initialized = 0;

/*
 * Connect to the Chidori GPU worker.
 * Parses CHIDORI_CFG_OVERRIDE for server address, or falls back to
 * connecting to the Agent's address directly.
 */
/*
 * Extract a JSON string value by key (minimal parser).
 * Returns pointer to malloc'd string or NULL.
 */
static char *json_get_string(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":\"", key);
    const char *start = strstr(json, search);
    if (!start) return NULL;
    start += strlen(search);
    const char *end = strchr(start, '"');
    if (!end) return NULL;
    size_t len = (size_t)(end - start);
    char *val = (char *)malloc(len + 1);
    memcpy(val, start, len);
    val[len] = '\0';
    return val;
}

static char *json_get_array_first(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":[\"", key);
    const char *start = strstr(json, search);
    if (!start) return NULL;
    start += strlen(search);
    const char *end = strchr(start, '"');
    if (!end) return NULL;
    size_t len = (size_t)(end - start);
    char *val = (char *)malloc(len + 1);
    memcpy(val, start, len);
    val[len] = '\0';
    return val;
}

static int tcp_connect(const char *host, int port) {
    struct addrinfo hints = {0}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    int rc = getaddrinfo(host, port_str, &hints, &res);
    if (rc != 0) {
        fprintf(stderr, "[chidori] getaddrinfo(%s): %s\n", host, gai_strerror(rc));
        return -1;
    }

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) {
        perror("[chidori] socket");
        freeaddrinfo(res);
        return -1;
    }

    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    /* Large socket buffers for WAN throughput (BDP = bandwidth * RTT).
     * At 1Gbps / 55ms RTT, BDP = 6.875MB. We set 16MB to fill the pipe. */
    int bufsize = 16 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));

    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        perror("[chidori] connect");
        close(fd);
        freeaddrinfo(res);
        return -1;
    }

    freeaddrinfo(res);

    /* Move fd to a high number to avoid PyTorch/cuBLAS closing it.
     * Low fds (3, 4, 5) get reused and closed by CUDA runtime internals. */
    int high_fd = dup(fd);
    if (high_fd >= 0) {
        close(fd);
        fd = high_fd;
        /* Try to get an even higher fd */
        int higher = fcntl(fd, F_DUPFD, 500);
        if (higher >= 0) {
            close(fd);
            fd = higher;
        }
    }
    LOG_DEBUG("Socket fd=%d", fd);

    /* TCP keepalive to detect dead connections */
    int keepalive = 1;
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive));
#ifdef TCP_KEEPIDLE
    int idle = 10;
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
    int interval = 5;
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof(interval));
    int cnt = 3;
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));
#endif

    return fd;
}

static int chidori_connect(void) {
    if (g_server_fd >= 0)
        return 0;

    const char *cfg = getenv("CHIDORI_CFG_OVERRIDE");
    if (!cfg) {
        fprintf(stderr, "[chidori] CHIDORI_CFG_OVERRIDE not set\n");
        return -1;
    }

    /* Parse session ID and server address from config */
    char *session_id = json_get_string(cfg, "id");
    char *server = json_get_array_first(cfg, "servers");

    if (!session_id || !server) {
        fprintf(stderr, "[chidori] Missing id or servers in config\n");
        free(session_id);
        free(server);
        return -1;
    }

    /* Split host:port */
    char host[256] = "127.0.0.1";
    int port = 43210;
    char *colon = strrchr(server, ':');
    if (colon) {
        *colon = '\0';
        strncpy(host, server, sizeof(host) - 1);
        port = atoi(colon + 1);
    } else {
        strncpy(host, server, sizeof(host) - 1);
    }

    /*
     * Check if CHIDORI_DIRECT_CONNECT is set — if so, connect directly to
     * chidori-worker's standalone TCP port (no HTTP/agent middleman).
     */
    const char *direct = getenv("CHIDORI_DIRECT_CONNECT");

    if (direct && direct[0] == '1') {
        /* Direct TCP connection to gpu-worker standalone mode */
        LOG_DEBUG("Direct connect to gpu-worker at %s:%d", host, port);

        g_server_fd = tcp_connect(host, port);
        if (g_server_fd < 0) {
            free(session_id);
            free(server);
            return -1;
        }

        LOG_DEBUG("Connected directly to GPU worker");
    } else {
        /* Agent-mediated connection via HTTP session protocol */
        LOG_DEBUG("Connecting to agent at %s:%d (session %s)", host, port, session_id);

        g_server_fd = tcp_connect(host, port);
        if (g_server_fd < 0) {
            free(session_id);
            free(server);
            return -1;
        }

        char http_req[2048];
        const char *body = "{\"id\":\"conn-1\",\"pid\":\"0\",\"processName\":\"chidori-client\"}";
        int body_len = (int)strlen(body);

        int req_len = snprintf(http_req, sizeof(http_req),
            "POST /v1/connect/session/%s HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: %d\r\n"
            "Connection: Upgrade\r\n"
            "\r\n"
            "%s",
            session_id, host, port, body_len, body);

        if (chidori_write_full(g_server_fd, http_req, (size_t)req_len) != 0) {
            fprintf(stderr, "[chidori] Failed to send HTTP connect request\n");
            close(g_server_fd);
            fprintf(stderr, "[chidori] SOCKET CLOSED at %s:%d\n", __FILE__, __LINE__); g_server_fd = -1;
            free(session_id);
            free(server);
            return -1;
        }

        /*
         * Read the HTTP 101 Switching Protocols response from the agent.
         * This confirms the agent has hijacked the connection and forwarded
         * the socket to gpu-worker.  We read byte-by-byte looking for the
         * end-of-headers marker "\r\n\r\n".  After this point the TCP
         * stream carries raw Chidori CUDA protocol.
         *
         * Timeout: we give the agent up to 30 seconds to respond (CUDA
         * init on the worker side can take a while on first launch).
         */
        {
            struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
            setsockopt(g_server_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

            char resp_buf[1024];
            int resp_len = 0;
            int found_end = 0;

            while (resp_len < (int)sizeof(resp_buf) - 1) {
                ssize_t n = read(g_server_fd, resp_buf + resp_len, 1);
                if (n <= 0) {
                    fprintf(stderr, "[chidori] Failed to read HTTP response from agent (n=%zd errno=%d)\n", n, errno);
                    close(g_server_fd);
                    fprintf(stderr, "[chidori] SOCKET CLOSED at %s:%d\n", __FILE__, __LINE__); g_server_fd = -1;
                    free(session_id);
                    free(server);
                    return -1;
                }
                resp_len += (int)n;
                /* Look for \r\n\r\n end-of-headers */
                if (resp_len >= 4 &&
                    resp_buf[resp_len - 4] == '\r' &&
                    resp_buf[resp_len - 3] == '\n' &&
                    resp_buf[resp_len - 2] == '\r' &&
                    resp_buf[resp_len - 1] == '\n') {
                    found_end = 1;
                    break;
                }
            }

            /* Clear the receive timeout for subsequent CUDA RPC traffic */
            tv.tv_sec = 0;
            tv.tv_usec = 0;
            setsockopt(g_server_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

            if (!found_end) {
                fprintf(stderr, "[chidori] Invalid HTTP response from agent\n");
                close(g_server_fd);
                fprintf(stderr, "[chidori] SOCKET CLOSED at %s:%d\n", __FILE__, __LINE__); g_server_fd = -1;
                free(session_id);
                free(server);
                return -1;
            }

            resp_buf[resp_len] = '\0';
            if (strstr(resp_buf, "101") == NULL) {
                fprintf(stderr, "[chidori] Agent did not return 101: %s\n", resp_buf);
                close(g_server_fd);
                fprintf(stderr, "[chidori] SOCKET CLOSED at %s:%d\n", __FILE__, __LINE__); g_server_fd = -1;
                free(session_id);
                free(server);
                return -1;
            }
        }

        LOG_DEBUG("Connected to remote GPU (session %s)", session_id);
    }

    free(session_id);
    free(server);
    return 0;
}

/*
 * Send an RPC and receive the response.
 * Thread-safe via mutex (simple approach; could use per-thread connections later).
 */
/*
 * ============================================================================
 * Centralized logging — CHIDORI_DEBUG=1 enables all debug output.
 *
 * Log levels:
 *   CHIDORI_DEBUG=1    — Connection events, device routing, errors, fat binary loads
 *   CHIDORI_TRACE=1    — Every sync/async RPC with func_id and timing
 *   CHIDORI_VERBOSE=1  — Per-RPC details (payload size, pending count, result)
 *
 * All output goes to stderr. Use LOG_DEBUG/LOG_TRACE/LOG_VERBOSE macros.
 * ============================================================================
 */
/* Aliases for existing code that references g_verbose/g_trace directly */
static int g_verbose = -1;
static int g_trace = -1;
static int g_pending_responses = 0;
static uint64_t g_async_count = 0;
static uint64_t g_sync_count = 0;

static void check_trace(void) {
    check_log_levels();
    g_verbose = g_log_verbose;
    g_trace = g_log_trace;
}

/*
 * Flush all pending async responses from the socket.
 * Must be called with g_lock held.
 */
static void chidori_flush_responses(void) {
    char drain[4096];
    while (g_pending_responses > 0 && g_server_fd >= 0) {
        chidori_response_t resp;
        if (chidori_recv_response(g_server_fd, &resp) != 0) {
            /* Connection died — reset */
            close(g_server_fd);
            fprintf(stderr, "[chidori] SOCKET CLOSED at %s:%d\n", __FILE__, __LINE__); g_server_fd = -1;
            g_pending_responses = 0;
            return;
        }
        /* Drain any response payload */
        size_t resp_payload = resp.msg_len - CHIDORI_RESPONSE_HEADER_SIZE;
        while (resp_payload > 0) {
            size_t chunk = resp_payload < sizeof(drain) ? resp_payload : sizeof(drain);
            if (chidori_read_full(g_server_fd, drain, chunk) != 0) {
                close(g_server_fd);
                fprintf(stderr, "[chidori] SOCKET CLOSED at %s:%d\n", __FILE__, __LINE__); g_server_fd = -1;
                g_pending_responses = 0;
                return;
            }
            resp_payload -= chunk;
        }
        g_pending_responses--;
    }
}

/*
 * Async RPC: send request, DON'T wait for response.
 * The response will be drained by the next sync RPC or explicit flush.
 */
static int g_cork_active = 0; /* TCP_CORK batching state */

static CUresult chidori_rpc_async(uint16_t func_id, const void *payload, size_t payload_len) {
    check_log_levels();
    check_trace();
    if (g_trace) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        fprintf(stderr, "[TRACE %ld.%03ld] ASYNC func=0x%04x len=%zu\n",
                (long)tv.tv_sec % 1000, (long)tv.tv_usec / 1000,
                func_id, payload_len);
    }
    g_async_count++;

    pthread_mutex_lock(&g_lock);

    if (g_server_fd < 0) {
        if (chidori_connect() != 0) {
            pthread_mutex_unlock(&g_lock);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    /* Enable TCP_CORK to batch multiple async writes into one TCP segment */
    if (!g_cork_active) {
#ifdef TCP_CORK
        int cork = 1;
        setsockopt(g_server_fd, IPPROTO_TCP, TCP_CORK, &cork, sizeof(cork));
#endif
        g_cork_active = 1;
    }

    chidori_request_t req = {0};
    req.func_id = func_id;
    req.flags = CHIDORI_FLAG_ASYNC;
    req.seq_num = ++g_seq_num;

    if (chidori_send_request(g_server_fd, &req, payload, payload_len) != 0) {
        fprintf(stderr, "[chidori] Async send failed func=0x%04x: %s\n", func_id, strerror(errno));
        close(g_server_fd);
        fprintf(stderr, "[chidori] SOCKET CLOSED at %s:%d\n", __FILE__, __LINE__); g_server_fd = -1;
        g_cork_active = 0;
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

/*
 * Synchronous RPC: send request, wait for response.
 * Flushes any pending async responses first.
 */
static CUresult chidori_rpc(uint16_t func_id, const void *payload, size_t payload_len,
                          void *out_payload, size_t *out_payload_len) {
    check_log_levels();
    check_trace();
    if (g_trace) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        fprintf(stderr, "[TRACE %ld.%03ld] SYNC  func=0x%04x len=%zu (async_total=%lu sync_total=%lu)\n",
                (long)tv.tv_sec % 1000, (long)tv.tv_usec / 1000,
                func_id, payload_len, (unsigned long)g_async_count, (unsigned long)g_sync_count);
    }
    g_sync_count++;
    LOG_VERBOSE("func=0x%04x payload=%zu pending=%d", func_id, payload_len, g_pending_responses);

    pthread_mutex_lock(&g_lock);

    /* Uncork TCP to flush any batched async writes before sync call */
    if (g_cork_active && g_server_fd >= 0) {
#ifdef TCP_CORK
        int cork = 0;
        setsockopt(g_server_fd, IPPROTO_TCP, TCP_CORK, &cork, sizeof(cork));
#endif
        g_cork_active = 0;
    }

    /* Flush pending async responses before sync call */
    if (g_pending_responses > 0)
        chidori_flush_responses();

    if (g_server_fd < 0) {
        if (chidori_connect() != 0) {
            pthread_mutex_unlock(&g_lock);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    /* Build and send request */
    chidori_request_t req = {0};
    req.func_id = func_id;
    req.flags = CHIDORI_FLAG_SYNC;
    req.seq_num = ++g_seq_num;

    if (chidori_send_request(g_server_fd, &req, payload, payload_len) != 0) {
        fprintf(stderr, "[chidori] Failed to send func=0x%04x: %s (errno=%d)\n",
                func_id, strerror(errno), errno);
        close(g_server_fd);
        fprintf(stderr, "[chidori] SOCKET CLOSED at %s:%d\n", __FILE__, __LINE__); g_server_fd = -1;
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    /* Receive response */
    chidori_response_t resp;
    if (chidori_recv_response(g_server_fd, &resp) != 0) {
        fprintf(stderr, "[chidori] Failed to recv response for func=0x%04x: %s (errno=%d)\n",
                func_id, strerror(errno), errno);
        close(g_server_fd);
        fprintf(stderr, "[chidori] SOCKET CLOSED at %s:%d\n", __FILE__, __LINE__); g_server_fd = -1;
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    /* Read response payload if present */
    size_t resp_payload_len = resp.msg_len - CHIDORI_RESPONSE_HEADER_SIZE;
    if (resp_payload_len > 0 && out_payload != NULL) {
        size_t to_read = resp_payload_len;
        if (out_payload_len && *out_payload_len < to_read)
            to_read = *out_payload_len;

        if (chidori_read_full(g_server_fd, out_payload, to_read) != 0) {
            fprintf(stderr, "[chidori] Failed to read response payload (%zu bytes) for func=0x%04x: %s (errno=%d)\n",
                    to_read, func_id, strerror(errno), errno);
            close(g_server_fd);
            fprintf(stderr, "[chidori] SOCKET CLOSED at %s:%d\n", __FILE__, __LINE__); g_server_fd = -1;
            pthread_mutex_unlock(&g_lock);
            return CUDA_ERROR_NOT_INITIALIZED;
        }

        /* Drain any extra bytes we didn't consume */
        if (to_read < resp_payload_len) {
            char drain[4096];
            size_t remaining = resp_payload_len - to_read;
            while (remaining > 0) {
                size_t chunk = remaining < sizeof(drain) ? remaining : sizeof(drain);
                chidori_read_full(g_server_fd, drain, chunk);
                remaining -= chunk;
            }
        }

        if (out_payload_len)
            *out_payload_len = to_read;
    } else if (resp_payload_len > 0) {
        /* Caller doesn't want payload, drain it */
        char drain[4096];
        size_t remaining = resp_payload_len;
        while (remaining > 0) {
            size_t chunk = remaining < sizeof(drain) ? remaining : sizeof(drain);
            chidori_read_full(g_server_fd, drain, chunk);
            remaining -= chunk;
        }
    }

    pthread_mutex_unlock(&g_lock);
    LOG_VERBOSE("func=0x%04x result=%d resp_payload=%zu",
                func_id, resp.result, resp_payload_len);
    return (CUresult)resp.result;
}

/*
 * ============================================================================
 * CUDA Driver API Interceptions
 *
 * These are the symbols that applications (and cuBLAS, cuDNN, etc.) link against.
 * By providing them in our LD_PRELOAD library, we intercept all CUDA calls.
 *
 * Phase 0: Core functions implemented below.
 * Phase 2: Codegen will auto-generate the full set (~100+ functions).
 * ============================================================================
 */

CUresult cuInit(unsigned int flags) {
    (void)flags;
    /* Reentrance guard: when real driver's cuInit calls back into us, just succeed */
    if (g_in_real_cuda > 0) return CUDA_SUCCESS;
    init_real_cuda(); /* Initialize partial offload if CHIDORI_LOCAL_DEVICES is set */
    g_initialized = 1;
    /* For partial offload: init the REAL local driver too */
    if (g_partial_offload && g_real.cuInit) {
        g_in_real_cuda++;
        g_real.cuInit(0);
        g_in_real_cuda--;
        g_real_cuda_inited = 1;
    }
    return chidori_rpc(CHIDORI_CU_INIT, &flags, sizeof(flags), NULL, NULL);
}

static int g_driver_version_cached = 0;
static int g_driver_version = 0;
CUresult cuDriverGetVersion(int *version) {
    if (g_in_real_cuda > 0 && g_real.cuDriverGetVersion) return g_real.cuDriverGetVersion(version);
    if (g_driver_version_cached) { *version = g_driver_version; return 0; }
    size_t len = sizeof(int32_t);
    CUresult r = chidori_rpc(CHIDORI_CU_DRIVER_GET_VERSION, NULL, 0, version, &len);
    if (r == 0) { g_driver_version = *version; g_driver_version_cached = 1; }
    return r;
}

CUresult cuDeviceGetCount(int *count) {
    /* Ensure partial offload is initialized before returning count */
    init_real_cuda();
    if (g_in_real_cuda > 0 && g_real.cuDeviceGetCount)
        return g_real.cuDeviceGetCount(count); /* reentrance: return REAL device count */
    int remote_count = 0;
    size_t len = sizeof(int32_t);
    CUresult r = chidori_rpc(CHIDORI_CU_DEVICE_GET_COUNT, NULL, 0, &remote_count, &len);
    if (remote_count <= 0) remote_count = 1;
    if (g_partial_offload) {
        *count = g_local_device_count + remote_count;
        fprintf(stderr, "[chidori] cuDeviceGetCount: local=%d + remote=%d = %d (partial_offload=1)\n",
                g_local_device_count, remote_count, *count);
    } else {
        *count = remote_count;
    }
    return r;
}

/* Forward declaration — defined after cuDeviceGetAttribute */
static void ensure_real_cuda_init(void);

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (g_in_real_cuda > 0 && g_real.cuDeviceGet) return g_real.cuDeviceGet(device, ordinal);
    if (g_partial_offload && is_local_device(ordinal) && g_real.cuDeviceGet) {
        ensure_real_cuda_init();
        return g_real.cuDeviceGet(device, ordinal);
    }
    int remote_ordinal = g_partial_offload ? (ordinal - g_local_device_count) : ordinal;
    size_t len = sizeof(int32_t);
    return chidori_rpc(CHIDORI_CU_DEVICE_GET, &remote_ordinal, sizeof(remote_ordinal), device, &len);
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (g_in_real_cuda > 0 && g_real.cuDeviceGetName) return g_real.cuDeviceGetName(name, len, dev);
    if (g_partial_offload && is_local_device(dev) && g_real.cuDeviceGetName) {
        ensure_real_cuda_init();
        return g_real.cuDeviceGetName(name, len, dev);
    }
    int remote_dev = g_partial_offload ? (dev - g_local_device_count) : dev;
    struct __attribute__((packed)) { int32_t len; int32_t dev; } req = { len, remote_dev };
    size_t out_len = (size_t)len;
    return chidori_rpc(CHIDORI_CU_DEVICE_GET_NAME, &req, sizeof(req), name, &out_len);
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    if (g_in_real_cuda > 0 && g_real.cuDeviceTotalMem_v2) return g_real.cuDeviceTotalMem_v2(bytes, dev);
    if (g_partial_offload && is_local_device(dev) && g_real.cuDeviceTotalMem_v2) {
        return g_real.cuDeviceTotalMem_v2(bytes, dev);
    }
    int remote_dev = g_partial_offload ? (dev - g_local_device_count) : dev;
    size_t len = sizeof(uint64_t);
    return chidori_rpc(CHIDORI_CU_DEVICE_TOTAL_MEM, &remote_dev, sizeof(remote_dev), bytes, &len);
}

/* Device attribute cache — avoids RPC for repeated queries (PyTorch queries 100+ attributes) */
#define ATTR_CACHE_SIZE 512
static struct { int dev; int attrib; int value; int valid; } g_attr_cache[ATTR_CACHE_SIZE];
static int g_attr_cache_count = 0;

/* g_real_cuda_inited declared near g_partial_offload, used by cuInit and ensure_real_cuda_init */
static void ensure_real_cuda_init(void) {
    if (!g_real_cuda_inited && g_real.cuInit) {
        g_real.cuInit(0);
        g_real_cuda_inited = 1;
    }
}

CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev) {
    if (g_in_real_cuda > 0 && g_real.cuDeviceGetAttribute) return g_real.cuDeviceGetAttribute(pi, attrib, dev);
    /* For partial offload: route local device queries to real CUDA */
    if (g_partial_offload && is_local_device(dev) && g_real.cuDeviceGetAttribute) {
        ensure_real_cuda_init();
        return g_real.cuDeviceGetAttribute(pi, attrib, dev);
    }
    /* Remap device for remote */
    int remote_dev = g_partial_offload ? (dev - g_local_device_count) : dev;

    /* Check cache first */
    for (int i = 0; i < g_attr_cache_count; i++) {
        if (g_attr_cache[i].dev == remote_dev && g_attr_cache[i].attrib == attrib && g_attr_cache[i].valid) {
            *pi = g_attr_cache[i].value;
            return CUDA_SUCCESS;
        }
    }
    struct __attribute__((packed)) { int32_t attrib; int32_t dev; } req = { attrib, remote_dev };
    size_t len = sizeof(int32_t);
    CUresult r = chidori_rpc(CHIDORI_CU_DEVICE_GET_ATTRIBUTE, &req, sizeof(req), pi, &len);
    if (r == CUDA_SUCCESS && g_attr_cache_count < ATTR_CACHE_SIZE) {
        g_attr_cache[g_attr_cache_count].dev = remote_dev;
        g_attr_cache[g_attr_cache_count].attrib = attrib;
        g_attr_cache[g_attr_cache_count].value = *pi;
        g_attr_cache[g_attr_cache_count].valid = 1;
        g_attr_cache_count++;
    }
    return r;
}

/* Context */
static CUcontext g_cached_ctx = NULL;
static int g_cached_ctx_valid = 0;

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    if (g_in_real_cuda > 0 && g_real.cuCtxCreate_v2) return g_real.cuCtxCreate_v2(pctx, flags, dev);
    struct { uint32_t flags; int32_t dev; } req = { flags, dev };
    size_t len = sizeof(uint64_t);
    CUresult r = chidori_rpc(CHIDORI_CU_CTX_CREATE, &req, sizeof(req), pctx, &len);
    if (r == CUDA_SUCCESS) {
        g_cached_ctx = *pctx;
        g_cached_ctx_valid = 1;
    }
    return r;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
    if (g_in_real_cuda > 0 && g_real.cuCtxDestroy_v2) return g_real.cuCtxDestroy_v2(ctx);
    uint64_t handle = (uint64_t)(uintptr_t)ctx;
    return chidori_rpc(CHIDORI_CU_CTX_DESTROY, &handle, sizeof(handle), NULL, NULL);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    if (g_in_real_cuda && g_real.cuCtxSetCurrent) return g_real.cuCtxSetCurrent(ctx); /* reentrance: forward to real driver */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuCtxSetCurrent) {
        g_in_real_cuda++;
        CUresult r = g_real.cuCtxSetCurrent(ctx);
        g_in_real_cuda--;
        g_cached_ctx = ctx;
        g_cached_ctx_valid = 1;
        return r;
    }
    uint64_t handle = (uint64_t)(uintptr_t)ctx;
    g_cached_ctx = ctx;
    g_cached_ctx_valid = 1;
    return chidori_rpc_async(CHIDORI_CU_CTX_SET_CURRENT, &handle, sizeof(handle));
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
    if (g_in_real_cuda && g_real.cuCtxGetCurrent) return g_real.cuCtxGetCurrent(pctx); /* reentrance: forward to real driver */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuCtxGetCurrent) {
        g_in_real_cuda++;
        CUresult r = g_real.cuCtxGetCurrent(pctx);
        g_in_real_cuda--;
        if (r == CUDA_SUCCESS) {
            g_cached_ctx = *pctx;
            g_cached_ctx_valid = 1;
        }
        return r;
    }
    if (g_cached_ctx_valid) {
        *pctx = g_cached_ctx;
        return CUDA_SUCCESS;
    }
    size_t len = sizeof(uint64_t);
    CUresult r = chidori_rpc(CHIDORI_CU_CTX_GET_CURRENT, NULL, 0, pctx, &len);
    if (r == CUDA_SUCCESS) {
        g_cached_ctx = *pctx;
        g_cached_ctx_valid = 1;
    }
    return r;
}

CUresult cuCtxSynchronize(void) {
    if (g_in_real_cuda && g_real.cuCtxSynchronize) return g_real.cuCtxSynchronize(); /* reentrance: forward to real driver */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuCtxSynchronize) {
        g_in_real_cuda++;
        CUresult r = g_real.cuCtxSynchronize();
        g_in_real_cuda--;
        return r;
    }
    return chidori_rpc(CHIDORI_CU_CTX_SYNCHRONIZE, NULL, 0, NULL, NULL);
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    if (g_in_real_cuda && g_real.cuDevicePrimaryCtxRetain) return g_real.cuDevicePrimaryCtxRetain(pctx, dev);
    if (g_partial_offload && is_local_device(dev) && g_real.cuDevicePrimaryCtxRetain) {
        ensure_real_cuda_init();
        g_in_real_cuda++;
        CUresult r = g_real.cuDevicePrimaryCtxRetain(pctx, dev);
        g_in_real_cuda--;
        if (r == CUDA_SUCCESS) {
            g_local_ctx = *pctx;
            g_cached_ctx = *pctx;
            g_cached_ctx_valid = 1;
        }
        return r;
    }
    int remote_dev = g_partial_offload ? (dev - g_local_device_count) : dev;
    size_t len = sizeof(uint64_t);
    CUresult r = chidori_rpc(CHIDORI_CU_DEVICE_PRIMARY_CTX_RETAIN, &remote_dev, sizeof(remote_dev), pctx, &len);
    if (r == CUDA_SUCCESS) {
        g_cached_ctx = *pctx;
        g_cached_ctx_valid = 1;
    }
    return r;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    if (g_in_real_cuda && g_real.cuDevicePrimaryCtxRelease_v2) return g_real.cuDevicePrimaryCtxRelease_v2(dev);
    if (g_partial_offload && is_local_device(dev) && g_real.cuDevicePrimaryCtxRelease_v2) {
        g_in_real_cuda++;
        CUresult r = g_real.cuDevicePrimaryCtxRelease_v2(dev);
        g_in_real_cuda--;
        return r;
    }
    int remote_dev = g_partial_offload ? (dev - g_local_device_count) : dev;
    return chidori_rpc(CHIDORI_CU_DEVICE_PRIMARY_CTX_RELEASE, &remote_dev, sizeof(remote_dev), NULL, NULL);
}

/* Memory */
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    /* Reentrance guard: if inside a real CUDA call, forward to real driver */
    if (g_in_real_cuda > 0 && g_real.cuMemAlloc_v2)
        return g_real.cuMemAlloc_v2(dptr, bytesize);
    uint64_t size = (uint64_t)bytesize;
    size_t len = sizeof(uint64_t);
    return chidori_rpc(CHIDORI_CU_MEM_ALLOC, &size, sizeof(size), dptr, &len);
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    if (g_in_real_cuda > 0 && g_real.cuMemFree_v2)
        return g_real.cuMemFree_v2(dptr);
    uint64_t ptr = (uint64_t)dptr;
    return chidori_rpc_async(CHIDORI_CU_MEM_FREE, &ptr, sizeof(ptr));
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t byteCount) {
    if (g_in_real_cuda > 0 && g_real.cuMemcpyHtoD_v2) return g_real.cuMemcpyHtoD_v2(dstDevice, srcHost, byteCount); /* reentrance guard */
    /* Payload: dst_ptr(8) + size(8) + data(byteCount) */
    size_t payload_len = 16 + byteCount;
    void *payload = malloc(payload_len);
    if (!payload) return CUDA_ERROR_INVALID_VALUE;

    memcpy(payload, &dstDevice, 8);
    uint64_t size64 = (uint64_t)byteCount;
    memcpy((char *)payload + 8, &size64, 8);
    memcpy((char *)payload + 16, srcHost, byteCount);

    CUresult r = chidori_rpc_async(CHIDORI_CU_MEMCPY_H_TO_D, payload, payload_len);
    free(payload);
    return r;
}

CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t byteCount) {
    if (g_in_real_cuda > 0 && g_real.cuMemcpyDtoH_v2) return g_real.cuMemcpyDtoH_v2(dstHost, srcDevice, byteCount); /* reentrance guard */
    /* Payload: src_ptr(8) + size(8) */
    struct { uint64_t src; uint64_t size; } req;
    req.src = (uint64_t)srcDevice;
    req.size = (uint64_t)byteCount;

    size_t out_len = byteCount;
    return chidori_rpc(CHIDORI_CU_MEMCPY_D_TO_H, &req, sizeof(req), dstHost, &out_len);
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount) {
    if (g_in_real_cuda > 0 && g_real.cuMemcpyDtoD_v2) return g_real.cuMemcpyDtoD_v2(dstDevice, srcDevice, byteCount); /* reentrance: forward to real driver */
    struct { uint64_t dst; uint64_t src; uint64_t size; } req;
    req.dst = (uint64_t)dstDevice;
    req.src = (uint64_t)srcDevice;
    req.size = (uint64_t)byteCount;
    return chidori_rpc_async(CHIDORI_CU_MEMCPY_D_TO_D, &req, sizeof(req));
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    if (g_in_real_cuda > 0 && g_real.cuMemGetInfo_v2) return g_real.cuMemGetInfo_v2(free, total);
    struct { uint64_t free; uint64_t total; } info;
    size_t len = sizeof(info);
    CUresult r = chidori_rpc(CHIDORI_CU_MEM_GET_INFO, NULL, 0, &info, &len);
    if (r == CUDA_SUCCESS) {
        *free = (size_t)info.free;
        *total = (size_t)info.total;
    }
    return r;
}

/* Module */
CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (g_in_real_cuda > 0 && g_real.cuModuleLoadData) return g_real.cuModuleLoadData(module, image);
    /* Local device: forward to real driver (cuBLAS loads GEMM kernels via this) */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuModuleLoadData) {
        g_in_real_cuda++;
        CUresult r = g_real.cuModuleLoadData(module, image);
        g_in_real_cuda--;
        return r;
    }
    /* Determine image size: check for PTX (text) vs CUBIN (binary) */
    size_t image_len;
    const unsigned char *data = (const unsigned char *)image;

    if (data[0] == 0x7f && data[1] == 'E' && data[2] == 'L' && data[3] == 'F') {
        /* ELF CUBIN — read size from ELF header (e_shoff + e_shnum * e_shentsize) */
        /* For safety, send 64KB which covers most kernels */
        image_len = 64 * 1024;
    } else if (memcmp(data, "\xed\xfe\xba\xbe", 4) == 0 || memcmp(data, "\xba\xbe\xfe\xed", 4) == 0) {
        /* Fat binary magic */
        uint64_t fat_size;
        memcpy(&fat_size, data + 8, 8);
        image_len = (size_t)fat_size;
    } else {
        /* Assume PTX (text) */
        image_len = strlen((const char *)image) + 1;
    }

    size_t out_len = sizeof(uint64_t);
    return chidori_rpc(CHIDORI_CU_MODULE_LOAD_DATA, image, image_len, module, &out_len);
}

CUresult cuModuleUnload(CUmodule mod) {
    if (g_in_real_cuda > 0 && g_real.cuModuleUnload) return g_real.cuModuleUnload(mod);
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuModuleUnload) {
        g_in_real_cuda++;
        CUresult r = g_real.cuModuleUnload(mod);
        g_in_real_cuda--;
        return r;
    }
    uint64_t handle = (uint64_t)(uintptr_t)mod;
    return chidori_rpc(CHIDORI_CU_MODULE_UNLOAD, &handle, sizeof(handle), NULL, NULL);
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    if (g_in_real_cuda > 0 && g_real.cuModuleGetFunction) return g_real.cuModuleGetFunction(hfunc, hmod, name);
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuModuleGetFunction) {
        g_in_real_cuda++;
        CUresult r = g_real.cuModuleGetFunction(hfunc, hmod, name);
        g_in_real_cuda--;
        return r;
    }
    /* Payload: module_handle(8) + function_name (null-terminated) */
    uint64_t mod_handle = (uint64_t)(uintptr_t)hmod;
    size_t name_len = strlen(name) + 1;
    size_t payload_len = 8 + name_len;
    void *payload = malloc(payload_len);
    if (!payload) return CUDA_ERROR_INVALID_VALUE;

    memcpy(payload, &mod_handle, 8);
    memcpy((char *)payload + 8, name, name_len);

    size_t out_len = sizeof(uint64_t);
    CUresult r = chidori_rpc(CHIDORI_CU_MODULE_GET_FUNCTION, payload, payload_len, hfunc, &out_len);
    free(payload);
    return r;
}

/* Execution */

/*
 * CUDA extra parameter sentinel values.
 * Defined as ((void*)0x01) etc. in cuda.h; redefined here to avoid
 * requiring cuda.h on the client side.
 */
#define CHIDORI_CU_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define CHIDORI_CU_LAUNCH_PARAM_BUFFER_SIZE    ((void *)0x02)
#define CHIDORI_CU_LAUNCH_PARAM_END            ((void *)0x00)

CUresult cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc) {
    if (g_in_real_cuda > 0 && g_real.cuFuncGetAttribute) return g_real.cuFuncGetAttribute(pi, attrib, hfunc);
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuFuncGetAttribute) {
        g_in_real_cuda++;
        CUresult r = g_real.cuFuncGetAttribute(pi, attrib, hfunc);
        g_in_real_cuda--;
        return r;
    }
    struct __attribute__((packed)) { int32_t attrib; uint64_t func; } req;
    req.attrib = attrib;
    req.func = (uint64_t)(uintptr_t)hfunc;
    size_t len = sizeof(int32_t);
    return chidori_rpc(CHIDORI_CU_FUNC_GET_ATTRIBUTE, &req, sizeof(req), pi, &len);
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra) {
    if (g_in_real_cuda > 0 && g_real.cuLaunchKernel) return g_real.cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    /* Local device: forward to real driver (cuBLAS launches GEMM kernels via this) */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuLaunchKernel) {
        g_in_real_cuda++;
        CUresult r = g_real.cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
        g_in_real_cuda--;
        return r;
    }
    /*
     * Kernel launch serialization — wire format:
     *   func_handle(8) + gridX(4) + gridY(4) + gridZ(4)
     *   + blockX(4) + blockY(4) + blockZ(4)
     *   + sharedMem(4) + stream_handle(8) = 44 bytes header
     *   + has_params(1):
     *       0 = no kernel parameters
     *       1 = contiguous parameter buffer follows
     *   + [if has_params == 1]: buffer_size(4) + buffer_data(buffer_size)
     *
     * On the server side, the buffer is passed via CU_LAUNCH_PARAM_BUFFER_POINTER
     * and CU_LAUNCH_PARAM_BUFFER_SIZE through cuLaunchKernel's extra parameter.
     */

    /* --- Extract kernel parameter buffer --- */
    void *param_buffer = NULL;
    size_t param_buffer_size = 0;
    int param_buffer_allocated = 0; /* 1 if we malloc'd param_buffer */

    if (extra != NULL) {
        /*
         * The extra array contains tagged key-value pairs:
         *   CU_LAUNCH_PARAM_BUFFER_POINTER, <ptr>,
         *   CU_LAUNCH_PARAM_BUFFER_SIZE, <size_t*>,
         *   CU_LAUNCH_PARAM_END
         */
        for (int i = 0; extra[i] != CHIDORI_CU_LAUNCH_PARAM_END; i++) {
            if (extra[i] == CHIDORI_CU_LAUNCH_PARAM_BUFFER_POINTER) {
                param_buffer = extra[++i];
            } else if (extra[i] == CHIDORI_CU_LAUNCH_PARAM_BUFFER_SIZE) {
                param_buffer_size = *(size_t *)extra[++i];
            }
        }
    } else if (kernelParams != NULL) {
        /*
         * kernelParams is an array of pointers to individual kernel arguments.
         * We query CU_FUNC_ATTRIBUTE_PARAM_SIZE (enum value 74) to get the
         * total parameter buffer size, then pack arguments contiguously.
         *
         * CUDA kernel parameters are laid out in declaration order with each
         * argument aligned to its natural alignment (max 8 bytes). Since all
         * CUDA kernel arguments are scalar types (1/2/4/8 bytes) or pointers
         * (8 bytes), we iterate and copy each as an 8-byte-aligned value.
         *
         * Note: this issues an extra RPC to query param size. Applications
         * that care about launch latency should use the extra parameter approach.
         */
        int total_param_size = 0;
        CUresult attr_res = cuFuncGetAttribute(&total_param_size,
                                                74 /* CU_FUNC_ATTRIBUTE_PARAM_SIZE */, f);
        if (attr_res == CUDA_SUCCESS && total_param_size > 0) {
            param_buffer_size = (size_t)total_param_size;
            param_buffer = malloc(param_buffer_size);
            if (!param_buffer) return CUDA_ERROR_INVALID_VALUE;
            param_buffer_allocated = 1;
            memset(param_buffer, 0, param_buffer_size);

            /*
             * Pack each kernelParams[i] into the contiguous buffer.
             * We copy 8 bytes per parameter (the largest scalar/pointer size)
             * with 8-byte alignment. If the total param buffer is smaller
             * than expected, we stop early. This handles the common case
             * where all args are pointers or 64-bit values.
             *
             * For kernels with smaller argument types (int, float), the CUDA
             * driver zero-extends them in the parameter buffer, so copying
             * 8 bytes from the caller's pointer is safe as long as the
             * caller's storage is at least 8 bytes (which is typical for
             * stack variables on 64-bit systems).
             */
            size_t offset = 0;
            for (int i = 0; offset < param_buffer_size && kernelParams[i] != NULL; i++) {
                /* Align to 8-byte boundary */
                offset = (offset + 7) & ~(size_t)7;
                if (offset >= param_buffer_size) break;
                size_t remaining = param_buffer_size - offset;
                size_t copy_size = remaining >= 8 ? 8 : remaining;
                memcpy((char *)param_buffer + offset, kernelParams[i], copy_size);
                offset += copy_size;
            }
        }
    }

    /* --- Build the wire payload --- */
    uint8_t has_params = (param_buffer != NULL && param_buffer_size > 0) ? 1 : 0;
    size_t header_size = 44; /* func(8) + grid(12) + block(12) + shared(4) + stream(8) */
    size_t payload_len = header_size + 1; /* + has_params(1) */
    if (has_params) {
        payload_len += 4 + param_buffer_size; /* buffer_size(4) + buffer_data */
    }

    void *payload = malloc(payload_len);
    if (!payload) {
        if (param_buffer_allocated) free(param_buffer);
        return CUDA_ERROR_INVALID_VALUE;
    }

    char *p = (char *)payload;

    /* Header: func_handle(8) + grid(12) + block(12) + shared(4) + stream(8) */
    uint64_t func_handle = (uint64_t)(uintptr_t)f;
    uint64_t stream_handle = (uint64_t)(uintptr_t)hStream;
    memcpy(p, &func_handle, 8); p += 8;
    memcpy(p, &gridDimX, 4); p += 4;
    memcpy(p, &gridDimY, 4); p += 4;
    memcpy(p, &gridDimZ, 4); p += 4;
    memcpy(p, &blockDimX, 4); p += 4;
    memcpy(p, &blockDimY, 4); p += 4;
    memcpy(p, &blockDimZ, 4); p += 4;
    memcpy(p, &sharedMemBytes, 4); p += 4;
    memcpy(p, &stream_handle, 8); p += 8;

    /* Kernel parameter buffer */
    memcpy(p, &has_params, 1); p += 1;
    if (has_params) {
        uint32_t buf_size = (uint32_t)param_buffer_size;
        memcpy(p, &buf_size, 4); p += 4;
        memcpy(p, param_buffer, param_buffer_size);
    }

    CUresult r = chidori_rpc_async(CHIDORI_CU_LAUNCH_KERNEL, payload, payload_len);

    free(payload);
    if (param_buffer_allocated) free(param_buffer);

    return r;
}

/* Stream */
CUresult cuStreamCreate(CUstream *phStream, unsigned int flags) {
    if (g_in_real_cuda > 0 && g_real.cuStreamCreate) return g_real.cuStreamCreate(phStream, flags); /* reentrance: forward to real driver */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuStreamCreate) {
        LOG_DEBUG("cuStreamCreate -> LOCAL (dev %d)", g_current_device);
        g_in_real_cuda++;
        CUresult r = g_real.cuStreamCreate(phStream, flags);
        g_in_real_cuda--;
        return r;
    }
    size_t len = sizeof(uint64_t);
    return chidori_rpc(CHIDORI_CU_STREAM_CREATE, &flags, sizeof(flags), phStream, &len);
}

CUresult cuStreamDestroy_v2(CUstream hStream) {
    if (g_in_real_cuda > 0 && g_real.cuStreamDestroy_v2) return g_real.cuStreamDestroy_v2(hStream); /* reentrance: forward to real driver */
    uint64_t handle = (uint64_t)(uintptr_t)hStream;
    return chidori_rpc_async(CHIDORI_CU_STREAM_DESTROY, &handle, sizeof(handle));
}

CUresult cuStreamSynchronize(CUstream hStream) {
    if (g_in_real_cuda && g_real.cuStreamSynchronize) return g_real.cuStreamSynchronize(hStream); /* reentrance guard */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuStreamSynchronize) {
        g_in_real_cuda++;
        CUresult r = g_real.cuStreamSynchronize(hStream);
        g_in_real_cuda--;
        return r;
    }
    uint64_t handle = (uint64_t)(uintptr_t)hStream;
    return chidori_rpc(CHIDORI_CU_STREAM_SYNCHRONIZE, &handle, sizeof(handle), NULL, NULL);
}

/* Event — route to local driver when on local device */
CUresult cuEventCreate(CUevent *phEvent, unsigned int flags) {
    if (g_in_real_cuda > 0 && g_real.cuEventCreate) return g_real.cuEventCreate(phEvent, flags);
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuEventCreate) {
        g_in_real_cuda++;
        CUresult r = g_real.cuEventCreate(phEvent, flags);
        g_in_real_cuda--;
        return r;
    }
    size_t len = sizeof(uint64_t);
    return chidori_rpc(CHIDORI_CU_EVENT_CREATE, &flags, sizeof(flags), phEvent, &len);
}

CUresult cuEventDestroy_v2(CUevent hEvent) {
    if (g_in_real_cuda > 0 && g_real.cuEventDestroy_v2) return g_real.cuEventDestroy_v2(hEvent);
    /* Try local first, then remote — event could be from either */
    if (g_partial_offload && g_real.cuEventDestroy_v2) {
        g_in_real_cuda++;
        CUresult r = g_real.cuEventDestroy_v2(hEvent);
        g_in_real_cuda--;
        if (r == CUDA_SUCCESS) return r;
    }
    uint64_t handle = (uint64_t)(uintptr_t)hEvent;
    return chidori_rpc_async(CHIDORI_CU_EVENT_DESTROY, &handle, sizeof(handle));
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    if (g_in_real_cuda > 0 && g_real.cuEventRecord) return g_real.cuEventRecord(hEvent, hStream);
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuEventRecord) {
        g_in_real_cuda++;
        CUresult r = g_real.cuEventRecord(hEvent, hStream);
        g_in_real_cuda--;
        return r;
    }
    struct { uint64_t event; uint64_t stream; } req;
    req.event = (uint64_t)(uintptr_t)hEvent;
    req.stream = (uint64_t)(uintptr_t)hStream;
    return chidori_rpc_async(CHIDORI_CU_EVENT_RECORD, &req, sizeof(req));
}

CUresult cuEventSynchronize(CUevent hEvent) {
    if (g_in_real_cuda > 0 && g_real.cuEventSynchronize) return g_real.cuEventSynchronize(hEvent);
    uintptr_t ev = (uintptr_t)hEvent;
    /* Local events are heap-allocated (large address). Remote events are small RPC handles. */
    if (g_partial_offload && ev > 0x100000 && g_real.cuEventSynchronize && g_local_ctx) {
        g_in_real_cuda++;
        CUcontext saved = NULL;
        if (g_real.cuCtxGetCurrent) g_real.cuCtxGetCurrent(&saved);
        if (g_real.cuCtxSetCurrent) g_real.cuCtxSetCurrent(g_local_ctx);
        CUresult r = g_real.cuEventSynchronize(hEvent);
        if (g_real.cuCtxSetCurrent) g_real.cuCtxSetCurrent(saved);
        g_in_real_cuda--;
        if (r == CUDA_SUCCESS) return r;
    }
    uint64_t handle = (uint64_t)(uintptr_t)hEvent;
    return chidori_rpc(CHIDORI_CU_EVENT_SYNCHRONIZE, &handle, sizeof(handle), NULL, NULL);
}

CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    if (g_in_real_cuda > 0 && g_real.cuEventElapsedTime) return g_real.cuEventElapsedTime(pMilliseconds, hStart, hEnd);
    struct { uint64_t start; uint64_t end; } req;
    req.start = (uint64_t)(uintptr_t)hStart;
    req.end = (uint64_t)(uintptr_t)hEnd;
    size_t len = sizeof(float);
    return chidori_rpc(CHIDORI_CU_EVENT_ELAPSED_TIME, &req, sizeof(req), pMilliseconds, &len);
}

/*
 * ============================================================================
 * CUDA Runtime API Interceptions
 *
 * These intercept the cudaXxx() functions that PyTorch, TensorFlow, and other
 * frameworks call. cudaError_t is typedef'd as int, same wire format as CUresult.
 * ============================================================================
 */

typedef int cudaError_t;
typedef void *cudaStream_t;
typedef struct { unsigned int x, y, z; } dim3;
#define cudaSuccess 0
#define cudaErrorInvalidDeviceFunction 8

/* cudaMemcpyKind enum values */
#define cudaMemcpyHostToHost     0
#define cudaMemcpyHostToDevice   1
#define cudaMemcpyDeviceToHost   2
#define cudaMemcpyDeviceToDevice 3

/* Track last error locally */
static cudaError_t g_last_error = 0;

/* g_device_count_cache declared in globals section above */

cudaError_t cudaGetDeviceCount(int *count) {
    /* Ensure partial offload is initialized BEFORE returning count */
    init_real_cuda();
    if (g_device_count_cache >= 0) {
        *count = g_device_count_cache;
        return cudaSuccess;
    }
    /* Get remote device count from server */
    int remote_count = 0;
    size_t len = sizeof(int32_t);
    chidori_rpc(CHIDORI_CUDA_GET_DEVICE_COUNT, NULL, 0, &remote_count, &len);
    /* Accept remote_count even if the result code is non-zero (some servers
     * return error codes but still provide the count) */
    if (remote_count <= 0) remote_count = 1; /* fallback: assume at least 1 remote GPU */
    if (g_partial_offload) {
        *count = g_local_device_count + remote_count;
    } else {
        *count = remote_count;
    }
    g_device_count_cache = *count;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* g_current_device declared in globals section */

cudaError_t cudaGetDevice(int *device) {
    /* Return cached value — avoids RPC for this very hot path */
    *device = g_current_device;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    /* Short-circuit: skip RPC if already on this device (hot path).
     * Exception: for local devices, we must still initialize the context
     * on the first call (g_local_ctx == NULL). */
    if (device == g_current_device) {
        if (!g_partial_offload || !is_local_device(device) || g_local_ctx)
            return cudaSuccess;
        /* Fall through to initialize local context */
    }

    int was_local = g_partial_offload && is_local_device(g_current_device);

    if (g_partial_offload && is_local_device(device)) {
        /* Route to local GPU via real CUDA driver API directly.
         * We cannot use g_real.cudaSetDevice (cudart) because cudart calls
         * driver API functions internally, hitting our interceptors and
         * seeing us as a stub library (error 36). Instead, set up the
         * context directly via the real driver's cuDevicePrimaryCtxRetain
         * + cuCtxSetCurrent which we loaded via dlopen. */
        LOG_DEBUG("cudaSetDevice(%d) -> LOCAL", device);
        ensure_real_cuda_init();
        CUresult cr = CUDA_SUCCESS;
        if (g_real.cuDevicePrimaryCtxRetain && !g_local_ctx) {
            g_in_real_cuda++;
            cr = g_real.cuDevicePrimaryCtxRetain(&g_local_ctx, device);
            g_in_real_cuda--;
            if (cr != CUDA_SUCCESS) {
                LOG_DEBUG("cudaSetDevice(%d) -> cuDevicePrimaryCtxRetain failed: %d", device, cr);
                g_last_error = (cudaError_t)cr;
                return (cudaError_t)cr;
            }
        }
        if (g_local_ctx && g_real.cuCtxSetCurrent) {
            g_in_real_cuda++;
            cr = g_real.cuCtxSetCurrent(g_local_ctx);
            g_in_real_cuda--;
            LOG_DEBUG("cudaSetDevice(%d) -> activated local GPU context %p (r=%d)", device, g_local_ctx, cr);
        }
        g_current_device = device;
        g_last_error = cudaSuccess;
        return cudaSuccess;
    }

    /* Switching from local to remote — deactivate real context to avoid
     * the real driver intercepting subsequent Driver API calls */
    if (was_local && g_real.cuCtxSetCurrent) {
        g_in_real_cuda++;
        g_real.cuCtxSetCurrent(NULL);
        g_in_real_cuda--;
        LOG_DEBUG("cudaSetDevice(%d) -> deactivated local GPU context", device);
    }

    /* Route to remote GPU via Chidori RPC */
    int32_t remote_dev = g_partial_offload ? (device - g_local_device_count) : device;
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_SET_DEVICE, &remote_dev, sizeof(remote_dev), NULL, NULL);
    if (r == cudaSuccess) {
        g_current_device = device;
    }
    g_last_error = r;
    return r;
}

/* Forward declaration — defined later in this file */
cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device);

cudaError_t cudaGetDeviceProperties(void *prop, int device) {
    /*
     * cudaDeviceProp struct layout varies between CUDA versions.
     * The server may run a different CUDA version than the application.
     * We fill the struct using individual cudaDeviceGetAttribute calls
     * at CUDA 12.x offsets (the dominant PyTorch runtime version).
     *
     * CUDA 12.x layout (first ~420 bytes):
     *   0:   char name[256]
     *   288: size_t totalGlobalMem
     *   296: size_t sharedMemPerBlock
     *   304: int regsPerBlock
     *   308: int warpSize
     *   312: size_t memPitch
     *   320: int maxThreadsPerBlock
     *   324: int maxThreadsDim[3]
     *   336: int maxGridSize[3]
     *   348: int clockRate
     *   352: size_t totalConstMem
     *   360: int major
     *   364: int minor
     *   368: size_t textureAlignment
     *   376: size_t texturePitchAlignment
     *   384: int deviceOverlap
     *   388: int multiProcessorCount
     *   392: int kernelExecTimeoutEnabled
     *   396: int integrated
     *   400: int canMapHostMemory
     *   404: int computeMode
     */
    if (!prop) { g_last_error = (cudaError_t)1; return (cudaError_t)1; }
    fprintf(stderr, "[chidori] cudaGetDeviceProperties(dev=%d) CALLED\n", device);

    memset(prop, 0, 1008);

    char *p = (char *)prop;
    int val;

    /* Offset diagnostic mode */
    static int g_offset_diag_early = -1;
    if (g_offset_diag_early == -1) {
        const char *d = getenv("CHIDORI_OFFSET_DIAG");
        g_offset_diag_early = (d && d[0] == '1') ? 1 : 0;
    }
    if (g_offset_diag_early) {
        /* Fill EVERY 4-byte slot with its offset — then the macros overwrite specific slots */
        for (int _i = 256; _i < 1008; _i += 4) {
            int _ov = _i; memcpy(p + _i, &_ov, 4);
        }
        /* Return immediately — don't overwrite with real values */
        cuDeviceGetName(p, 256, device);
        g_last_error = cudaSuccess;
        return cudaSuccess;
    }

    /* name: use cuDeviceGetName (routes to local or remote) */
    cuDeviceGetName(p, 256, device);

    /* totalGlobalMem at offset 288 (size_t = 8) — use cuDeviceTotalMem */
    {
        size_t total = 0;
        cuDeviceTotalMem_v2(&total, device); /* routes to local or remote */
        memcpy(p + 288, &total, 8);
    }

    /* DIAGNOSTIC MODE: write offset markers instead of real values */
    static int g_offset_diag = -1;
    if (g_offset_diag == -1) {
        const char *d = getenv("CHIDORI_OFFSET_DIAG");
        g_offset_diag = (d && d[0] == '1') ? 1 : 0;
    }

    /* Helper macro: query a Runtime API device attribute and write to the struct */
    #define SET_INT(offset, cuda_attr) do { \
        if (g_offset_diag) { int _ov = (offset); memcpy(p + (offset), &_ov, 4); } \
        else { val = 0; cudaDeviceGetAttribute(&val, cuda_attr, device); memcpy(p + (offset), &val, 4); } \
    } while(0)

    #define SET_SIZE_T(offset, cuda_attr) do { \
        if (g_offset_diag) { size_t _ov = (size_t)(offset); memcpy(p + (offset), &_ov, 8); } \
        else { val = 0; cudaDeviceGetAttribute(&val, cuda_attr, device); size_t sv = (size_t)val; memcpy(p + (offset), &sv, 8); } \
    } while(0)

    /* cudaDevAttr enum values (stable across CUDA versions) */
    SET_SIZE_T(296, 8);   /* sharedMemPerBlock = cudaDevAttrMaxSharedMemoryPerBlock */
    SET_INT(304, 12);     /* regsPerBlock = cudaDevAttrMaxRegistersPerBlock */
    SET_INT(308, 10);     /* warpSize = cudaDevAttrWarpSize */
    SET_SIZE_T(312, 11);  /* memPitch = cudaDevAttrMaxPitch */
    SET_INT(320, 1);      /* maxThreadsPerBlock = cudaDevAttrMaxThreadsPerBlock */
    SET_INT(324, 2);      /* maxThreadsDim[0] = cudaDevAttrMaxBlockDimX */
    SET_INT(328, 3);      /* maxThreadsDim[1] = cudaDevAttrMaxBlockDimY */
    SET_INT(332, 4);      /* maxThreadsDim[2] = cudaDevAttrMaxBlockDimZ */
    SET_INT(336, 5);      /* maxGridSize[0] = cudaDevAttrMaxGridDimX */
    SET_INT(340, 6);      /* maxGridSize[1] = cudaDevAttrMaxGridDimY */
    SET_INT(344, 7);      /* maxGridSize[2] = cudaDevAttrMaxGridDimZ */
    SET_INT(348, 13);     /* clockRate = cudaDevAttrClockRate */
    SET_SIZE_T(352, 9);   /* totalConstMem = cudaDevAttrTotalConstantMemory */
    SET_INT(360, 75);     /* major = cudaDevAttrComputeCapabilityMajor */
    SET_INT(364, 76);     /* minor = cudaDevAttrComputeCapabilityMinor */
    SET_SIZE_T(368, 14);  /* textureAlignment = cudaDevAttrTextureAlignment */
    SET_SIZE_T(376, 52);  /* texturePitchAlignment = cudaDevAttrTexturePitchAlignment */
    SET_INT(384, 15);     /* deviceOverlap = cudaDevAttrGpuOverlap */
    SET_INT(388, 16);     /* multiProcessorCount = cudaDevAttrMultiProcessorCount */
    SET_INT(392, 17);     /* kernelExecTimeoutEnabled = cudaDevAttrKernelExecTimeout */
    SET_INT(396, 18);     /* integrated = cudaDevAttrIntegrated */
    SET_INT(400, 19);     /* canMapHostMemory = cudaDevAttrCanMapHostMemory */
    SET_INT(404, 20);     /* computeMode = cudaDevAttrComputeMode */
    SET_INT(408, 21);     /* maxTexture1D = cudaDevAttrMaxTexture1DWidth */
    SET_INT(424, 35);     /* concurrentKernels = cudaDevAttrConcurrentKernels */
    SET_INT(432, 36);     /* ECCEnabled = cudaDevAttrEccEnabled */
    SET_INT(436, 33);     /* pciBusID = cudaDevAttrPciBusId */
    SET_INT(440, 34);     /* pciDeviceID = cudaDevAttrPciDeviceId */
    /* Offsets below diverge between CUDA versions due to added fields.
     * These offsets are empirically determined from PyTorch 2.10+cu128
     * (CUDA 12.8) using CHIDORI_OFFSET_DIAG=1 pattern-fill technique. */
    SET_INT(612, 37);     /* memoryBusWidth = cudaDevAttrGlobalMemoryBusWidth (offset 612 in CUDA 12.8) */
    SET_INT(624, 39);     /* maxThreadsPerMultiProcessor = cudaDevAttrMaxThreadsPerMultiProcessor */
    SET_INT(648, 82);     /* regsPerMultiprocessor = cudaDevAttrMaxRegistersPerMultiprocessor */

    #undef SET_INT
    #undef SET_SIZE_T

    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device) {
    /* For partial offload: route local device queries to real CUDA */
    init_real_cuda(); /* ensure g_partial_offload is set */
    if (g_partial_offload && is_local_device(device) && g_real.cuDeviceGetAttribute) {
        LOG_DEBUG("cudaDeviceGetAttribute(attr=%d, dev=%d) → LOCAL", attr, device);
        ensure_real_cuda_init();
        g_in_real_cuda++;
        int r = (int)g_real.cuDeviceGetAttribute(value, attr, device);
        g_in_real_cuda--;
        g_last_error = (cudaError_t)r;
        return (cudaError_t)r;
    }
    /* Remap device index for remote queries */
    int remote_dev = g_partial_offload ? (device - g_local_device_count) : device;

    /* Use the driver-API attribute cache */
    for (int i = 0; i < g_attr_cache_count; i++) {
        if (g_attr_cache[i].dev == remote_dev && g_attr_cache[i].attrib == attr && g_attr_cache[i].valid) {
            *value = g_attr_cache[i].value;
            g_last_error = cudaSuccess;
            return cudaSuccess;
        }
    }
    struct __attribute__((packed)) { int32_t attr; int32_t device; } req = { attr, remote_dev };
    size_t len = sizeof(int32_t);
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_DEVICE_GET_ATTRIBUTE,
                                            &req, sizeof(req), value, &len);
    if (r == cudaSuccess && g_attr_cache_count < ATTR_CACHE_SIZE) {
        g_attr_cache[g_attr_cache_count].dev = remote_dev;
        g_attr_cache[g_attr_cache_count].attrib = attr;
        g_attr_cache[g_attr_cache_count].value = *value;
        g_attr_cache[g_attr_cache_count].valid = 1;
        g_attr_cache_count++;
    }
    g_last_error = r;
    return r;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    /* Use Driver API (cuMemAlloc) for local device to avoid Runtime API reentrance */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuMemAlloc_v2) {
        LOG_DEBUG("cudaMalloc(%zu) -> LOCAL via cuMemAlloc (dev %d)", size, g_current_device);
        ensure_real_cuda_init();
        /* Ensure local context is active (may not be if cudaSetDevice wasn't called yet) */
        if (!g_local_ctx && g_real.cuDevicePrimaryCtxRetain) {
            g_in_real_cuda++;
            g_real.cuDevicePrimaryCtxRetain(&g_local_ctx, g_current_device);
            g_in_real_cuda--;
        }
        if (g_local_ctx && g_real.cuCtxSetCurrent) {
            g_in_real_cuda++;
            g_real.cuCtxSetCurrent(g_local_ctx);
            g_in_real_cuda--;
        }
        g_in_real_cuda++;
        CUdeviceptr dptr = 0;
        CUresult r = g_real.cuMemAlloc_v2(&dptr, size);
        g_in_real_cuda--;
        if (devPtr) *devPtr = (void *)(uintptr_t)dptr;
        if (r == CUDA_SUCCESS && dptr) alloc_track_add(dptr, size, g_current_device);
        g_last_error = (cudaError_t)r;
        return (cudaError_t)r;
    }
    uint64_t size64 = (uint64_t)size;
    uint64_t ptr64 = 0;
    size_t len = sizeof(uint64_t);
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_MALLOC, &size64, sizeof(size64), &ptr64, &len);
    if (r == cudaSuccess) {
        *devPtr = (void *)(uintptr_t)ptr64;
        alloc_track_add(ptr64, size, g_current_device);
    }
    g_last_error = r;
    return r;
}

cudaError_t cudaFree(void *devPtr) {
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuMemFree_v2) {
        g_in_real_cuda++;
        CUresult r = g_real.cuMemFree_v2((CUdeviceptr)(uintptr_t)devPtr);
        g_in_real_cuda--;
        g_last_error = (cudaError_t)r;
        return (cudaError_t)r;
    }
    uint64_t ptr = (uint64_t)(uintptr_t)devPtr;
    cudaError_t r = (cudaError_t)chidori_rpc_async(CHIDORI_CUDA_FREE, &ptr, sizeof(ptr));
    g_last_error = r;
    return r;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind) {
    /* Route to local GPU if current device is local.
     * Use DRIVER API directly (cuMemcpyHtoD/DtoH) to avoid Runtime API
     * reentrance — calling g_real.cudaMemcpy would go through PLT back to us. */
    if (g_partial_offload && is_local_device(g_current_device)) {
        g_in_real_cuda++;
        CUresult r = CUDA_SUCCESS;
        switch (kind) {
        case 1: /* HtoD */
            if (g_real.cuMemcpyHtoD_v2)
                r = g_real.cuMemcpyHtoD_v2((CUdeviceptr)(uintptr_t)dst, src, count);
            break;
        case 2: /* DtoH */
            if (g_real.cuMemcpyDtoH_v2)
                r = g_real.cuMemcpyDtoH_v2(dst, (CUdeviceptr)(uintptr_t)src, count);
            break;
        case 3: /* DtoD */
            if (g_real.cuMemcpyDtoD_v2)
                r = g_real.cuMemcpyDtoD_v2((CUdeviceptr)(uintptr_t)dst, (CUdeviceptr)(uintptr_t)src, count);
            break;
        default: /* HtoH */
            memcpy(dst, src, count);
            break;
        }
        g_in_real_cuda--;
        g_last_error = (cudaError_t)r;
        return (cudaError_t)r;
    }
    cudaError_t r;

    switch (kind) {
    case cudaMemcpyHostToDevice: {
        /* Payload: dst_ptr(8) + src_ptr(8=0) + size(8) + kind(4) + data(count)
         * Same header layout as DtoH/DtoD so kind is always at offset 24. */
        size_t payload_len = 8 + 8 + 8 + 4 + count;
        void *payload = malloc(payload_len);
        if (!payload) { g_last_error = CUDA_ERROR_INVALID_VALUE; return CUDA_ERROR_INVALID_VALUE; }

        uint64_t dst64 = (uint64_t)(uintptr_t)dst;
        uint64_t src64 = 0;
        uint64_t size64 = (uint64_t)count;
        int32_t kind32 = kind;
        char *p = (char *)payload;
        memcpy(p, &dst64, 8); p += 8;
        memcpy(p, &src64, 8); p += 8;
        memcpy(p, &size64, 8); p += 8;
        memcpy(p, &kind32, 4); p += 4;
        memcpy(p, src, count);

        r = (cudaError_t)chidori_rpc_async(CHIDORI_CUDA_MEMCPY, payload, payload_len);
        free(payload);
        break;
    }
    case cudaMemcpyDeviceToHost: {
        /* Payload: dst_ptr(8) + src_ptr(8) + size(8) + kind(4) */
        struct __attribute__((packed)) {
            uint64_t dst;
            uint64_t src;
            uint64_t size;
            int32_t  kind;
        } req;
        req.dst = (uint64_t)(uintptr_t)dst;
        req.src = (uint64_t)(uintptr_t)src;
        req.size = (uint64_t)count;
        req.kind = kind;

        size_t out_len = count;
        r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_MEMCPY, &req, sizeof(req), dst, &out_len);
        break;
    }
    case cudaMemcpyDeviceToDevice: {
        /* Payload: dst_ptr(8) + src_ptr(8) + size(8) + kind(4) */
        struct __attribute__((packed)) {
            uint64_t dst;
            uint64_t src;
            uint64_t size;
            int32_t  kind;
        } req;
        req.dst = (uint64_t)(uintptr_t)dst;
        req.src = (uint64_t)(uintptr_t)src;
        req.size = (uint64_t)count;
        req.kind = kind;

        r = (cudaError_t)chidori_rpc_async(CHIDORI_CUDA_MEMCPY, &req, sizeof(req));
        break;
    }
    case cudaMemcpyHostToHost:
    default:
        /* HtoH: just memcpy locally */
        memcpy(dst, src, count);
        r = cudaSuccess;
        break;
    }

    g_last_error = r;
    return r;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, int kind, void *stream) {
    (void)stream;

    /* Cross-device DtoD: stage through CPU when src and dst are on different devices */
    if (g_partial_offload && kind == 3 /* DtoD */) {
        int src_dev = alloc_track_device((uintptr_t)src);
        int dst_dev = alloc_track_device((uintptr_t)dst);
        if (src_dev >= 0 && dst_dev >= 0 && src_dev != dst_dev) {
            /* Cross-device: DtoH from source, HtoD to destination */
            void *staging = malloc(count);
            if (!staging) return (cudaError_t)2;
            int saved = g_current_device;
            cudaSetDevice(src_dev);
            cudaError_t r = cudaMemcpy(staging, (void*)src, count, 2 /* DtoH */);
            if (r == cudaSuccess) {
                cudaSetDevice(dst_dev);
                r = cudaMemcpy(dst, staging, count, 1 /* HtoD */);
            }
            free(staging);
            cudaSetDevice(saved);
            g_last_error = r;
            return r;
        }
    }

    /* Route to local GPU via Driver API if current device is local */
    if (g_partial_offload && is_local_device(g_current_device)) {
        ensure_real_cuda_init();
        g_in_real_cuda++;
        CUresult r = CUDA_SUCCESS;
        switch (kind) {
        case 1: /* HtoD */
            if (g_real.cuMemcpyHtoD_v2)
                r = g_real.cuMemcpyHtoD_v2((CUdeviceptr)(uintptr_t)dst, src, count);
            break;
        case 2: /* DtoH */
            if (g_real.cuMemcpyDtoH_v2)
                r = g_real.cuMemcpyDtoH_v2(dst, (CUdeviceptr)(uintptr_t)src, count);
            break;
        case 3: /* DtoD */
            if (g_real.cuMemcpyDtoD_v2)
                r = g_real.cuMemcpyDtoD_v2((CUdeviceptr)(uintptr_t)dst, (CUdeviceptr)(uintptr_t)src, count);
            break;
        default:
            memcpy(dst, src, count);
            break;
        }
        g_in_real_cuda--;
        g_last_error = (cudaError_t)r;
        return (cudaError_t)r;
    }
    cudaError_t r;

    switch (kind) {
    case cudaMemcpyHostToDevice: {
        /* Payload: dst_ptr(8) + src_ptr(8=0) + size(8) + kind(4) + stream(8) + data(count)
         * Same header layout as DtoH/DtoD so kind is always at offset 24. */
        size_t payload_len = 8 + 8 + 8 + 4 + 8 + count;
        void *payload = malloc(payload_len);
        if (!payload) { g_last_error = CUDA_ERROR_INVALID_VALUE; return CUDA_ERROR_INVALID_VALUE; }

        uint64_t dst64 = (uint64_t)(uintptr_t)dst;
        uint64_t src64 = 0; /* No device source for HtoD — data is in payload */
        uint64_t size64 = (uint64_t)count;
        int32_t kind32 = kind;
        uint64_t stream64 = (uint64_t)(uintptr_t)stream;
        char *p = (char *)payload;
        memcpy(p, &dst64, 8); p += 8;
        memcpy(p, &src64, 8); p += 8;
        memcpy(p, &size64, 8); p += 8;
        memcpy(p, &kind32, 4); p += 4;
        memcpy(p, &stream64, 8); p += 8;
        memcpy(p, src, count);

        r = (cudaError_t)chidori_rpc_async(CHIDORI_CUDA_MEMCPY_ASYNC, payload, payload_len);
        free(payload);
        break;
    }
    case cudaMemcpyDeviceToHost: {
        struct __attribute__((packed)) {
            uint64_t dst;
            uint64_t src;
            uint64_t size;
            int32_t  kind;
            uint64_t stream;
        } req;
        req.dst = (uint64_t)(uintptr_t)dst;
        req.src = (uint64_t)(uintptr_t)src;
        req.size = (uint64_t)count;
        req.kind = kind;
        req.stream = (uint64_t)(uintptr_t)stream;

        size_t out_len = count;
        r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_MEMCPY_ASYNC, &req, sizeof(req), dst, &out_len);
        break;
    }
    case cudaMemcpyDeviceToDevice: {
        struct __attribute__((packed)) {
            uint64_t dst;
            uint64_t src;
            uint64_t size;
            int32_t  kind;
            uint64_t stream;
        } req;
        req.dst = (uint64_t)(uintptr_t)dst;
        req.src = (uint64_t)(uintptr_t)src;
        req.size = (uint64_t)count;
        req.kind = kind;
        req.stream = (uint64_t)(uintptr_t)stream;

        r = (cudaError_t)chidori_rpc_async(CHIDORI_CUDA_MEMCPY_ASYNC, &req, sizeof(req));
        break;
    }
    case cudaMemcpyHostToHost:
    default:
        memcpy(dst, src, count);
        r = cudaSuccess;
        break;
    }

    g_last_error = r;
    return r;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    struct __attribute__((packed)) {
        uint64_t ptr;
        int32_t  value;
        uint64_t count;
    } req;
    req.ptr = (uint64_t)(uintptr_t)devPtr;
    req.value = value;
    req.count = (uint64_t)count;
    cudaError_t r = (cudaError_t)chidori_rpc_async(CHIDORI_CUDA_MEMSET, &req, sizeof(req));
    g_last_error = r;
    return r;
}

cudaError_t cudaDeviceSynchronize(void) {
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuCtxSynchronize) {
        /* Use real driver's cuCtxSynchronize instead of cudart's cudaDeviceSynchronize
         * to avoid cudart's internal driver API calls hitting our interceptors */
        g_in_real_cuda++;
        CUresult cr = g_real.cuCtxSynchronize();
        g_in_real_cuda--;
        g_last_error = (cudaError_t)cr;
        return (cudaError_t)cr;
    }
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_DEVICE_SYNCHRONIZE, NULL, 0, NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaStreamCreate(void **pStream) {
    if (g_partial_offload && is_local_device(g_current_device)) {
        /* For local device, create stream via real CUDA driver */
        LOG_DEBUG("cudaStreamCreate -> LOCAL (dev %d)", g_current_device);
        ensure_real_cuda_init();
        g_in_real_cuda++;
        CUresult r = g_real.cuStreamCreate
            ? g_real.cuStreamCreate((CUstream *)pStream, 0)
            : CUDA_SUCCESS;
        g_in_real_cuda--;
        g_last_error = (cudaError_t)r;
        return (cudaError_t)r;
    }
    uint64_t handle = 0;
    size_t len = sizeof(uint64_t);
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_STREAM_CREATE, NULL, 0, &handle, &len);
    if (r == cudaSuccess)
        *pStream = (void *)(uintptr_t)handle;
    g_last_error = r;
    return r;
}

cudaError_t cudaStreamDestroy(void *stream) {
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    cudaError_t r = (cudaError_t)chidori_rpc_async(CHIDORI_CUDA_STREAM_DESTROY, &handle, sizeof(handle));
    g_last_error = r;
    return r;
}

cudaError_t cudaStreamSynchronize(void *stream) {
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuStreamSynchronize) {
        /* Use real driver's cuStreamSynchronize instead of cudart's version
         * to avoid cudart's internal driver API calls hitting our interceptors */
        g_in_real_cuda++;
        CUresult cr = g_real.cuStreamSynchronize((CUstream)stream);
        g_in_real_cuda--;
        g_last_error = (cudaError_t)cr;
        return (cudaError_t)cr;
    }
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_STREAM_SYNCHRONIZE, &handle, sizeof(handle), NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaEventCreate(void **pEvent) {
    /* Local device: create event on real driver */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuEventCreate) {
        g_in_real_cuda++;
        CUresult r = g_real.cuEventCreate((CUevent *)pEvent, 0x02 /* CU_EVENT_DISABLE_TIMING */);
        g_in_real_cuda--;
        g_last_error = (cudaError_t)r;
        return (cudaError_t)r;
    }
    uint64_t handle = 0;
    size_t len = sizeof(uint64_t);
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_EVENT_CREATE, NULL, 0, &handle, &len);
    if (r == cudaSuccess)
        *pEvent = (void *)(uintptr_t)handle;
    g_last_error = r;
    return r;
}

cudaError_t cudaEventRecord(void *event, void *stream) {
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuEventRecord) {
        g_in_real_cuda++;
        CUresult r = g_real.cuEventRecord((CUevent)event, (CUstream)stream);
        g_in_real_cuda--;
        g_last_error = (cudaError_t)r;
        return (cudaError_t)r;
    }
    struct { uint64_t event; uint64_t stream; } req;
    req.event = (uint64_t)(uintptr_t)event;
    req.stream = (uint64_t)(uintptr_t)stream;
    cudaError_t r = (cudaError_t)chidori_rpc_async(CHIDORI_CUDA_EVENT_RECORD, &req, sizeof(req));
    g_last_error = r;
    return r;
}

/* cudaEventRecordWithFlags — CUDA 11.1+ */
cudaError_t cudaEventRecordWithFlags(void *event, void *stream, unsigned int flags) {
    (void)flags;
    return cudaEventRecord(event, stream);
}

cudaError_t cudaEventSynchronize(void *event) {
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuEventSynchronize) {
        g_in_real_cuda++;
        CUresult r = g_real.cuEventSynchronize((CUevent)event);
        g_in_real_cuda--;
        g_last_error = (cudaError_t)r;
        return (cudaError_t)r;
    }
    uint64_t handle = (uint64_t)(uintptr_t)event;
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_EVENT_SYNCHRONIZE, &handle, sizeof(handle), NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaEventDestroy(void *event) {
    /* For simplicity, try local first then RPC — event might have been created on either */
    if (g_partial_offload && g_real.cuEventDestroy_v2) {
        g_in_real_cuda++;
        CUresult r = g_real.cuEventDestroy_v2((CUevent)event);
        g_in_real_cuda--;
        if (r == CUDA_SUCCESS) { g_last_error = cudaSuccess; return cudaSuccess; }
        /* If local destroy failed, try RPC (event might be remote) */
    }
    uint64_t handle = (uint64_t)(uintptr_t)event;
    cudaError_t r = (cudaError_t)chidori_rpc_async(CHIDORI_CUDA_EVENT_DESTROY, &handle, sizeof(handle));
    g_last_error = r;
    return r;
}

static int g_runtime_version_cached = 0;
static int g_runtime_version = 0;
cudaError_t cudaRuntimeGetVersion(int *version) {
    if (g_runtime_version_cached) { *version = g_runtime_version; return cudaSuccess; }
    size_t len = sizeof(int32_t);
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_RUNTIME_GET_VERSION, NULL, 0, version, &len);
    if (r == cudaSuccess) { g_runtime_version = *version; g_runtime_version_cached = 1; }
    g_last_error = r;
    return r;
}

static int g_cuda_driver_version_cached = 0;
static int g_cuda_driver_version = 0;
cudaError_t cudaDriverGetVersion(int *version) {
    if (g_cuda_driver_version_cached) { *version = g_cuda_driver_version; return cudaSuccess; }
    size_t len = sizeof(int32_t);
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_DRIVER_GET_VERSION, NULL, 0, version, &len);
    if (r == cudaSuccess) { g_cuda_driver_version = *version; g_cuda_driver_version_cached = 1; }
    g_last_error = r;
    return r;
}

cudaError_t cudaMemGetInfo(size_t *free_mem, size_t *total_mem) {
    struct { uint64_t free; uint64_t total; } info;
    size_t len = sizeof(info);
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_MEM_GET_INFO, NULL, 0, &info, &len);
    if (r == cudaSuccess) {
        *free_mem = (size_t)info.free;
        *total_mem = (size_t)info.total;
    }
    g_last_error = r;
    return r;
}

cudaError_t cudaDeviceReset(void) {
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_DEVICE_RESET, NULL, 0, NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaGetLastError(void) {
    cudaError_t r = g_last_error;
    if (r != 0)
        LOG_DEBUG("cudaGetLastError() returning %d", r);
    g_last_error = cudaSuccess;
    return r;
}

cudaError_t cudaPeekAtLastError(void) {
    if (g_last_error != 0)
        LOG_DEBUG("cudaPeekAtLastError() returning %d", g_last_error);
    return g_last_error;
}

cudaError_t cudaDeviceCanAccessPeer(int *canAccess, int device, int peerDevice) {
    /* Partial offload: local and remote devices can't peer-access each other.
     * Don't send RPC — the remote server doesn't know about our device numbering. */
    if (g_partial_offload) {
        if (canAccess) *canAccess = 0; /* no peer access between local+remote */
        g_last_error = cudaSuccess;
        return cudaSuccess;
    }
    struct { int32_t device; int32_t peerDevice; } req = { device, peerDevice };
    size_t len = sizeof(int32_t);
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_DEVICE_CAN_ACCESS_PEER,
                                            &req, sizeof(req), canAccess, &len);
    g_last_error = r;
    return r;
}

/* cudaFuncGetAttributes — implemented after g_registered_functions is defined (see below) */

/*
 * ============================================================================
 * cuBLAS API Interceptions
 *
 * cuBLAS uses cublasHandle_t (opaque pointer) and cublasStatus_t (int).
 * alpha/beta parameters are HOST pointers to scalars, not device pointers.
 * Matrix pointers (A, B, C) are device pointers sent as uint64_t handles.
 * ============================================================================
 */

typedef int cublasStatus_t;
typedef void *cublasHandle_t;
typedef int cublasOperation_t;
typedef int cublasMath_t;

#define CUBLAS_STATUS_SUCCESS          0
#define CUBLAS_STATUS_NOT_INITIALIZED  1
#define CUBLAS_STATUS_ALLOC_FAILED     3
#define CUBLAS_STATUS_INVALID_VALUE    7
#define CUBLAS_STATUS_NOT_SUPPORTED    15

/* Helper to determine scalar size from cudaDataType / cublasComputeType */
static size_t chidori_cublas_scalar_size(int dtype) {
    switch (dtype) {
    case 0:  /* CUDA_R_16F / CUBLAS_COMPUTE_16F */
        return 2;
    case 2:  /* CUDA_R_32F / CUBLAS_COMPUTE_32F */
    case 6:  /* CUBLAS_COMPUTE_32F_FAST_16F etc */
    case 7:
    case 8:
        return 4;
    case 1:  /* CUDA_R_64F / CUBLAS_COMPUTE_64F */
        return 8;
    default:
        return 4; /* safe default: float-sized */
    }
}

/* Local cuBLAS handle for partial offload — lazily created */
static cublasHandle_t g_local_cublas_handle = NULL;

static cublasHandle_t ensure_local_cublas_handle(void) {
    if (g_local_cublas_handle) return g_local_cublas_handle;
    void *h = ensure_real_cublas();
    if (!h) return NULL;
    typedef cublasStatus_t (*create_fn)(cublasHandle_t *);
    create_fn real_create = (create_fn)__real_dlsym(h, "cublasCreate_v2");
    if (!real_create) return NULL;
    /* The DEEPBIND cuBLAS uses the real CUDA driver via its own dependency scope.
     * Our intercepted cudaSetDevice(0) sets the local GPU context via the real driver.
     * Call it to ensure the context is current (matching the ctypes test that worked). */
    /* Ensure real CUDA context is current for the local device */
    extern int cudaSetDevice(int);
    cudaSetDevice(0);

    /* Enable dlopen redirect AND reentrance guard: cuBLAS's internal driver calls
     * (cuDeviceGetCount, cuCtxGetCurrent, cuModuleLoadData, etc.) must go to the
     * REAL driver, not through our interception layer. Without g_in_real_cuda++,
     * cuBLAS sees 2 devices (our virtual count) but the real driver only has 1,
     * causing CUBLAS_STATUS_INTERNAL_ERROR (14). */
    g_dlopen_redirect_active = 1;
    g_in_real_cuda++;
    cublasStatus_t r = real_create(&g_local_cublas_handle);
    g_in_real_cuda--;
    g_dlopen_redirect_active = 0;
    LOG_DEBUG("cublasCreate_v2 = %d, handle=%p", r, (void*)g_local_cublas_handle);
    if (r != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[chidori] ensure_local_cublas_handle: cublasCreate_v2 failed: %d\n", r);
        g_local_cublas_handle = NULL;
        return NULL;
    }
    LOG_DEBUG("Created local cuBLAS handle: %p", (void*)g_local_cublas_handle);
    return g_local_cublas_handle;
}

static uint64_t g_last_cublas_handle = 0; /* Track last cuBLAS handle for cublasLt translation */
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (local_h) {
            *handle = local_h;
            return CUBLAS_STATUS_SUCCESS;
        }
    }
    uint64_t h = 0;
    size_t len = sizeof(uint64_t);
    cublasStatus_t r = (cublasStatus_t)chidori_rpc(CHIDORI_CUBLAS_CREATE, NULL, 0, &h, &len);
    if (r == CUBLAS_STATUS_SUCCESS) {
        *handle = (cublasHandle_t)(uintptr_t)h;
        g_last_cublas_handle = h;
    }
    return r;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        /* Don't actually destroy — we reuse the local handle */
        return CUBLAS_STATUS_SUCCESS;
    }
    uint64_t h = (uint64_t)(uintptr_t)handle;
    return (cublasStatus_t)chidori_rpc(CHIDORI_CUBLAS_DESTROY, &h, sizeof(h), NULL, NULL);
}

/* Client-side cuBLAS state tracking — avoid sync RPCs for get calls */
static uint64_t g_cublas_cached_stream_handle = 0;
static CUstream g_cublas_cached_stream = NULL;
static uint64_t g_cublas_cached_mode_handle = 0;
static int32_t  g_cublas_cached_mode = 0;
static int      g_cublas_mode_valid = 0;
static int      g_cublas_stream_valid = 0;

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, CUstream stream) {
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, CUstream);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasSetStream_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, stream);
            g_in_real_cuda--;
            return r;
        }
    }
    struct { uint64_t handle; uint64_t stream; } req;
    req.handle = (uint64_t)(uintptr_t)handle;
    req.stream = (uint64_t)(uintptr_t)stream;
    cublasStatus_t r = (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_SET_STREAM, &req, sizeof(req));
    g_cublas_cached_stream_handle = req.handle;
    g_cublas_cached_stream = stream;
    g_cublas_stream_valid = 1;
    return r;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, CUstream *stream) {
    uint64_t h = (uint64_t)(uintptr_t)handle;
    if (g_cublas_stream_valid && g_cublas_cached_stream_handle == h) {
        *stream = g_cublas_cached_stream;
        return CUBLAS_STATUS_SUCCESS;
    }
    uint64_t s = 0;
    size_t len = sizeof(uint64_t);
    cublasStatus_t r = (cublasStatus_t)chidori_rpc(CHIDORI_CUBLAS_GET_STREAM, &h, sizeof(h), &s, &len);
    if (r == CUBLAS_STATUS_SUCCESS) {
        *stream = (CUstream)(uintptr_t)s;
        g_cublas_cached_stream_handle = h;
        g_cublas_cached_stream = *stream;
        g_cublas_stream_valid = 1;
    }
    return r;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasMath_t);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasSetMathMode"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, mode);
            g_in_real_cuda--;
            return r;
        }
    }
    struct __attribute__((packed)) { uint64_t handle; int32_t mode; } req;
    req.handle = (uint64_t)(uintptr_t)handle;
    req.mode = (int32_t)mode;
    cublasStatus_t r = (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_SET_MATH_MODE, &req, sizeof(req));
    g_cublas_cached_mode_handle = req.handle;
    g_cublas_cached_mode = (int32_t)mode;
    g_cublas_mode_valid = 1;
    return r;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
    uint64_t h = (uint64_t)(uintptr_t)handle;
    if (g_cublas_mode_valid && g_cublas_cached_mode_handle == h) {
        *mode = (cublasMath_t)g_cublas_cached_mode;
        return CUBLAS_STATUS_SUCCESS;
    }
    int32_t m = 0;
    size_t len = sizeof(int32_t);
    cublasStatus_t r = (cublasStatus_t)chidori_rpc(CHIDORI_CUBLAS_GET_MATH_MODE, &h, sizeof(h), &m, &len);
    if (r == CUBLAS_STATUS_SUCCESS)
        *mode = (cublasMath_t)m;
    return r;
}

static int g_cublas_version_cached = 0;
static int g_cublas_version = 0;
cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, int *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasGetVersion_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, version);
            g_in_real_cuda--;
            return r;
        }
    }
    if (g_cublas_version_cached) { *version = g_cublas_version; return CUBLAS_STATUS_SUCCESS; }
    uint64_t h = (uint64_t)(uintptr_t)handle;
    int32_t v = 0;
    size_t len = sizeof(int32_t);
    cublasStatus_t r = (cublasStatus_t)chidori_rpc(CHIDORI_CUBLAS_GET_VERSION, &h, sizeof(h), &v, &len);
    if (r == CUBLAS_STATUS_SUCCESS) { *version = (int)v; g_cublas_version = (int)v; g_cublas_version_cached = 1; }
    return r;
}

/*
 * cublasSgemm_v2 — single-precision GEMM
 * C = alpha * op(A) * op(B) + beta * C
 *
 * Wire format:
 *   handle(8) + transa(4) + transb(4) + m(4) + n(4) + k(4)
 *   + alpha(4) + A(8) + lda(4) + B(8) + ldb(4)
 *   + beta(4) + C(8) + ldc(4) = 72 bytes
 */
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const void *A, int lda,
    const void *B, int ldb,
    const float *beta,
    void *C, int ldc) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const void *, int, const void *, int, const float *, void *, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasSgemm_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        float    alpha;
        uint64_t A;
        int32_t  lda;
        uint64_t B;
        int32_t  ldb;
        float    beta;
        uint64_t C;
        int32_t  ldc;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.B = (uint64_t)(uintptr_t)B;
    req.ldb = ldb;
    req.beta = *beta;
    req.C = (uint64_t)(uintptr_t)C;
    req.ldc = ldc;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_SGEMM, &req, sizeof(req));
}

/*
 * cublasDgemm_v2 — double-precision GEMM
 * Same layout as Sgemm but with double alpha/beta.
 */
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const void *A, int lda,
    const void *B, int ldb,
    const double *beta,
    void *C, int ldc) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double *, const void *, int, const void *, int, const double *, void *, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasDgemm_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        double   alpha;
        uint64_t A;
        int32_t  lda;
        uint64_t B;
        int32_t  ldb;
        double   beta;
        uint64_t C;
        int32_t  ldc;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.B = (uint64_t)(uintptr_t)B;
    req.ldb = ldb;
    req.beta = *beta;
    req.C = (uint64_t)(uintptr_t)C;
    req.ldc = ldc;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_DGEMM, &req, sizeof(req));
}

/*
 * cublasHgemm — half-precision GEMM
 * alpha/beta are __half (2 bytes each), sent as uint16_t.
 */
cublasStatus_t cublasHgemm(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int lda,
    const void *B, int ldb,
    const void *beta,
    void *C, int ldc) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void *, int, const void *, int, const void *, void *, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasHgemm"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        uint16_t alpha;
        uint64_t A;
        int32_t  lda;
        uint64_t B;
        int32_t  ldb;
        uint16_t beta;
        uint64_t C;
        int32_t  ldc;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;
    memcpy(&req.alpha, alpha, 2);
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.B = (uint64_t)(uintptr_t)B;
    req.ldb = ldb;
    memcpy(&req.beta, beta, 2);
    req.C = (uint64_t)(uintptr_t)C;
    req.ldc = ldc;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_HGEMM, &req, sizeof(req));
}

/*
 * cublasSgemmEx — mixed-precision Sgemm with type descriptors
 */
cublasStatus_t cublasSgemmEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const float *beta,
    void *C, int Ctype, int ldc) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const void *, int, int, const void *, int, int, const float *, void *, int, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasSgemmEx"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        float    alpha;
        uint64_t A;
        int32_t  Atype, lda;
        uint64_t B;
        int32_t  Btype, ldb;
        float    beta;
        uint64_t C;
        int32_t  Ctype, ldc;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.Atype = Atype; req.lda = lda;
    req.B = (uint64_t)(uintptr_t)B;
    req.Btype = Btype; req.ldb = ldb;
    req.beta = *beta;
    req.C = (uint64_t)(uintptr_t)C;
    req.Ctype = Ctype; req.ldc = ldc;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_SGEMM_EX, &req, sizeof(req));
}

/*
 * cublasGemmEx — the critical mixed-precision GEMM used by PyTorch
 *
 * Wire format:
 *   handle(8) + transa(4) + transb(4) + m(4) + n(4) + k(4)
 *   + alpha_scalar(8) + A(8) + Atype(4) + lda(4) + B(8) + Btype(4) + ldb(4)
 *   + beta_scalar(8) + C(8) + Ctype(4) + ldc(4)
 *   + computeType(4) + algo(4)
 *
 * alpha/beta are sent as 8-byte values (max scalar size) with the actual
 * type determined by computeType.
 */
cublasStatus_t cublasGemmEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const void *beta,
    void *C, int Ctype, int ldc,
    int computeType, int algo) {
    fprintf(stderr, "[chidori] cublasGemmEx ENTERED m=%d n=%d k=%d Atype=%d compute=%d dev=%d\n",
            m, n, k, Atype, computeType, g_current_device);

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void *, int, int, const void *, int, int, const void *, void *, int, int, int, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasGemmEx"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_dlopen_redirect_active = 1;
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
            g_in_real_cuda--;
            g_dlopen_redirect_active = 0;
            if (r != 0) fprintf(stderr, "[chidori] cublasGemmEx LOCAL failed: %d (Atype=%d Ctype=%d compute=%d)\n", r, Atype, Ctype, computeType);
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        uint8_t  alpha[8]; /* up to double-precision scalar */
        uint64_t A;
        int32_t  Atype, lda;
        uint64_t B;
        int32_t  Btype, ldb;
        uint8_t  beta[8];
        uint64_t C;
        int32_t  Ctype, ldc;
        int32_t  computeType;
        int32_t  algo;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;

    size_t scalar_sz = chidori_cublas_scalar_size(computeType);
    memcpy(req.alpha, alpha, scalar_sz);
    req.A = (uint64_t)(uintptr_t)A;
    req.Atype = Atype; req.lda = lda;
    req.B = (uint64_t)(uintptr_t)B;
    req.Btype = Btype; req.ldb = ldb;
    memcpy(req.beta, beta, scalar_sz);
    req.C = (uint64_t)(uintptr_t)C;
    req.Ctype = Ctype; req.ldc = ldc;
    req.computeType = computeType;
    req.algo = algo;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_GEMM_EX, &req, sizeof(req));
}

/*
 * cublasSgemv_v2 — single-precision matrix-vector multiply
 * y = alpha * op(A) * x + beta * y
 */
cublasStatus_t cublasSgemv_v2(cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const float *alpha,
    const void *A, int lda,
    const void *x, int incx,
    const float *beta,
    void *y, int incy) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, int, int, const float *, const void *, int, const void *, int, const float *, void *, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasSgemv_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  trans;
        int32_t  m, n;
        float    alpha;
        uint64_t A;
        int32_t  lda;
        uint64_t x;
        int32_t  incx;
        float    beta;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.trans = (int32_t)trans;
    req.m = m; req.n = n;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.beta = *beta;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_SGEMV, &req, sizeof(req));
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const double *alpha,
    const void *A, int lda,
    const void *x, int incx,
    const double *beta,
    void *y, int incy) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, int, int, const double *, const void *, int, const void *, int, const double *, void *, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasDgemv_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  trans;
        int32_t  m, n;
        double   alpha;
        uint64_t A;
        int32_t  lda;
        uint64_t x;
        int32_t  incx;
        double   beta;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.trans = (int32_t)trans;
    req.m = m; req.n = n;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.beta = *beta;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_DGEMV, &req, sizeof(req));
}

/*
 * AXPY: y = alpha * x + y
 */
cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle,
    int n, const float *alpha,
    const void *x, int incx,
    void *y, int incy) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, const float *, const void *, int, void *, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasSaxpy_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, n, alpha, x, incx, y, incy);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        float    alpha;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.alpha = *alpha;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_SAXPY, &req, sizeof(req));
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle,
    int n, const double *alpha,
    const void *x, int incx,
    void *y, int incy) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, const double *, const void *, int, void *, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasDaxpy_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, n, alpha, x, incx, y, incy);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        double   alpha;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.alpha = *alpha;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_DAXPY, &req, sizeof(req));
}

/*
 * NRM2: result = ||x||_2
 * Result scalar is returned in the response payload.
 */
cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle,
    int n, const void *x, int incx, float *result) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, const void *, int, float *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasSnrm2_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, n, x, incx, result);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;

    size_t len = sizeof(float);
    return (cublasStatus_t)chidori_rpc(CHIDORI_CUBLAS_SNRM2, &req, sizeof(req), result, &len);
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle,
    int n, const void *x, int incx, double *result) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, const void *, int, double *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasDnrm2_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, n, x, incx, result);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;

    size_t len = sizeof(double);
    return (cublasStatus_t)chidori_rpc(CHIDORI_CUBLAS_DNRM2, &req, sizeof(req), result, &len);
}

/*
 * DOT: result = x . y
 */
cublasStatus_t cublasSdot_v2(cublasHandle_t handle,
    int n, const void *x, int incx,
    const void *y, int incy, float *result) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, const void *, int, const void *, int, float *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasSdot_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, n, x, incx, y, incy, result);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    size_t len = sizeof(float);
    return (cublasStatus_t)chidori_rpc(CHIDORI_CUBLAS_SDOT, &req, sizeof(req), result, &len);
}

cublasStatus_t cublasDdot_v2(cublasHandle_t handle,
    int n, const void *x, int incx,
    const void *y, int incy, double *result) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, const void *, int, const void *, int, double *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasDdot_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, n, x, incx, y, incy, result);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    size_t len = sizeof(double);
    return (cublasStatus_t)chidori_rpc(CHIDORI_CUBLAS_DDOT, &req, sizeof(req), result, &len);
}

/*
 * SCAL: x = alpha * x
 */
cublasStatus_t cublasSscal_v2(cublasHandle_t handle,
    int n, const float *alpha, void *x, int incx) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, const float *, void *, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasSscal_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, n, alpha, x, incx);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        float    alpha;
        uint64_t x;
        int32_t  incx;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.alpha = *alpha;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_SSCAL, &req, sizeof(req));
}

cublasStatus_t cublasDscal_v2(cublasHandle_t handle,
    int n, const double *alpha, void *x, int incx) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, const double *, void *, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasDscal_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, n, alpha, x, incx);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        double   alpha;
        uint64_t x;
        int32_t  incx;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.alpha = *alpha;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_DSCAL, &req, sizeof(req));
}

/*
 * COPY: y = x
 */
cublasStatus_t cublasScopy_v2(cublasHandle_t handle,
    int n, const void *x, int incx, void *y, int incy) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, const void *, int, void *, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasScopy_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, n, x, incx, y, incy);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_SCOPY, &req, sizeof(req));
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t handle,
    int n, const void *x, int incx, void *y, int incy) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, const void *, int, void *, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasDcopy_v2"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, n, x, incx, y, incy);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_DCOPY, &req, sizeof(req));
}

/*
 * cublasSgemmBatched — batched Sgemm with array of device pointers
 *
 * Aarray, Barray, Carray are HOST arrays of device pointers.
 * We send all batchCount pointers inline.
 */
cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const void * const *Aarray, int lda,
    const void * const *Barray, int ldb,
    const float *beta,
    void * const *Carray, int ldc,
    int batchCount) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const void * const *, int, const void * const *, int, const float *, void * const *, int, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasSgemmBatched"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
            g_in_real_cuda--;
            return r;
        }
    }

    /*
     * Wire format:
     *   handle(8) + transa(4) + transb(4) + m(4) + n(4) + k(4)
     *   + alpha(4) + lda(4) + ldb(4) + beta(4) + ldc(4) + batchCount(4)
     *   = 52 bytes header
     *   + Aarray(8*batchCount) + Barray(8*batchCount) + Carray(8*batchCount)
     */
    size_t ptr_array_size = (size_t)batchCount * sizeof(uint64_t);
    size_t payload_len = 52 + 3 * ptr_array_size;
    void *payload = malloc(payload_len);
    if (!payload) return CUBLAS_STATUS_ALLOC_FAILED;

    char *p = (char *)payload;
    uint64_t h = (uint64_t)(uintptr_t)handle;
    int32_t ta = (int32_t)transa, tb = (int32_t)transb;
    int32_t m32 = m, n32 = n, k32 = k;
    float a = *alpha, b = *beta;
    int32_t lda32 = lda, ldb32 = ldb, ldc32 = ldc, bc32 = batchCount;

    memcpy(p, &h, 8); p += 8;
    memcpy(p, &ta, 4); p += 4;
    memcpy(p, &tb, 4); p += 4;
    memcpy(p, &m32, 4); p += 4;
    memcpy(p, &n32, 4); p += 4;
    memcpy(p, &k32, 4); p += 4;
    memcpy(p, &a, 4); p += 4;
    memcpy(p, &lda32, 4); p += 4;
    memcpy(p, &ldb32, 4); p += 4;
    memcpy(p, &b, 4); p += 4;
    memcpy(p, &ldc32, 4); p += 4;
    memcpy(p, &bc32, 4); p += 4;

    /* Copy device pointer arrays */
    for (int i = 0; i < batchCount; i++) {
        uint64_t ptr = (uint64_t)(uintptr_t)Aarray[i];
        memcpy(p, &ptr, 8); p += 8;
    }
    for (int i = 0; i < batchCount; i++) {
        uint64_t ptr = (uint64_t)(uintptr_t)Barray[i];
        memcpy(p, &ptr, 8); p += 8;
    }
    for (int i = 0; i < batchCount; i++) {
        uint64_t ptr = (uint64_t)(uintptr_t)Carray[i];
        memcpy(p, &ptr, 8); p += 8;
    }

    cublasStatus_t r = (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_SGEMM_BATCHED, payload, payload_len);
    free(payload);
    return r;
}

/*
 * cublasSgemmStridedBatched — strided batched Sgemm
 */
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const void *A, int lda, long long int strideA,
    const void *B, int ldb, long long int strideB,
    const float *beta,
    void *C, int ldc, long long int strideC,
    int batchCount) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const void *, int, long long int, const void *, int, long long int, const float *, void *, int, long long int, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasSgemmStridedBatched"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        float    alpha;
        uint64_t A;
        int32_t  lda;
        int64_t  strideA;
        uint64_t B;
        int32_t  ldb;
        int64_t  strideB;
        float    beta;
        uint64_t C;
        int32_t  ldc;
        int64_t  strideC;
        int32_t  batchCount;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.strideA = (int64_t)strideA;
    req.B = (uint64_t)(uintptr_t)B;
    req.ldb = ldb;
    req.strideB = (int64_t)strideB;
    req.beta = *beta;
    req.C = (uint64_t)(uintptr_t)C;
    req.ldc = ldc;
    req.strideC = (int64_t)strideC;
    req.batchCount = batchCount;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_SGEMM_STRIDED_BATCHED, &req, sizeof(req));
}

/*
 * cublasGemmStridedBatchedEx — mixed-precision strided batched GEMM
 */
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long int strideA,
    const void *B, int Btype, int ldb, long long int strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long int strideC,
    int batchCount, int computeType, int algo) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void *, int, int, long long int, const void *, int, int, long long int, const void *, void *, int, int, long long int, int, int, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasGemmStridedBatchedEx"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
            g_in_real_cuda--;
            return r;
        }
    }

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        uint8_t  alpha[8];
        uint64_t A;
        int32_t  Atype, lda;
        int64_t  strideA;
        uint64_t B;
        int32_t  Btype, ldb;
        int64_t  strideB;
        uint8_t  beta[8];
        uint64_t C;
        int32_t  Ctype, ldc;
        int64_t  strideC;
        int32_t  batchCount;
        int32_t  computeType;
        int32_t  algo;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;

    size_t scalar_sz = chidori_cublas_scalar_size(computeType);
    memcpy(req.alpha, alpha, scalar_sz);
    req.A = (uint64_t)(uintptr_t)A;
    req.Atype = Atype; req.lda = lda;
    req.strideA = (int64_t)strideA;
    req.B = (uint64_t)(uintptr_t)B;
    req.Btype = Btype; req.ldb = ldb;
    req.strideB = (int64_t)strideB;
    memcpy(req.beta, beta, scalar_sz);
    req.C = (uint64_t)(uintptr_t)C;
    req.Ctype = Ctype; req.ldc = ldc;
    req.strideC = (int64_t)strideC;
    req.batchCount = batchCount;
    req.computeType = computeType;
    req.algo = algo;

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_GEMM_STRIDED_BATCHED_EX, &req, sizeof(req));
}

/*
 * ============================================================================
 * cuDNN API Interceptions
 *
 * cuDNN uses opaque descriptor handles (all transmitted as uint64_t) and
 * host-side alpha/beta scalars (same pattern as cuBLAS).
 * cudnnStatus_t is an int enum; we use int as the wire type.
 * ============================================================================
 */

typedef int cudnnStatus_t;
typedef void *cudnnHandle_t;
typedef void *cudnnTensorDescriptor_t;
typedef void *cudnnFilterDescriptor_t;
typedef void *cudnnConvolutionDescriptor_t;
typedef void *cudnnActivationDescriptor_t;
typedef void *cudnnPoolingDescriptor_t;
typedef void *cudnnDropoutDescriptor_t;
typedef void *cudnnRNNDescriptor_t;

typedef int cudnnDataType_t;
typedef int cudnnTensorFormat_t;
typedef int cudnnConvolutionMode_t;
typedef int cudnnConvolutionFwdAlgo_t;
typedef int cudnnConvolutionBwdDataAlgo_t;
typedef int cudnnConvolutionFwdAlgoPerf_t;
typedef int cudnnActivationMode_t;
typedef int cudnnNanPropagation_t;
typedef int cudnnPoolingMode_t;
typedef int cudnnSoftmaxAlgorithm_t;
typedef int cudnnSoftmaxMode_t;
typedef int cudnnBatchNormMode_t;
typedef int cudnnMathType_t;
typedef int cudnnConvolutionBwdFilterAlgo_t;

#define CUDNN_STATUS_SUCCESS             0
#define CUDNN_STATUS_NOT_INITIALIZED     1
#define CUDNN_STATUS_ALLOC_FAILED        2
#define CUDNN_STATUS_BAD_PARAM           3
#define CUDNN_STATUS_NOT_SUPPORTED       9

/* ---- Handle lifecycle ---- */

cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
    uint64_t h = 0;
    size_t len = sizeof(uint64_t);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(CHIDORI_CUDNN_CREATE, NULL, 0, &h, &len);
    if (r == CUDNN_STATUS_SUCCESS)
        *handle = (cudnnHandle_t)(uintptr_t)h;
    return r;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
    uint64_t h = (uint64_t)(uintptr_t)handle;
    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_DESTROY, &h, sizeof(h));
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, CUstream stream) {
    struct { uint64_t handle; uint64_t stream; } req;
    req.handle = (uint64_t)(uintptr_t)handle;
    req.stream = (uint64_t)(uintptr_t)stream;
    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_STREAM, &req, sizeof(req));
}

static size_t g_cudnn_version_cached = 0;
static size_t g_cudnn_version = 0;
size_t cudnnGetVersion(void) {
    if (g_cudnn_version_cached) return g_cudnn_version;
    uint64_t version = 0;
    size_t len = sizeof(uint64_t);
    chidori_rpc(CHIDORI_CUDNN_GET_VERSION, NULL, 0, &version, &len);
    g_cudnn_version = (size_t)version;
    g_cudnn_version_cached = 1;
    return g_cudnn_version;
}

/* ---- Tensor Descriptor ---- */

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
    uint64_t h = 0;
    size_t len = sizeof(uint64_t);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(CHIDORI_CUDNN_CREATE_TENSOR_DESCRIPTOR,
                                                NULL, 0, &h, &len);
    if (r == CUDNN_STATUS_SUCCESS)
        *tensorDesc = (cudnnTensorDescriptor_t)(uintptr_t)h;
    return r;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t format, cudnnDataType_t dataType,
    int n, int c, int h, int w) {

    struct __attribute__((packed)) {
        uint64_t desc;
        int32_t  format;
        int32_t  dataType;
        int32_t  n, c, h, w;
    } req;

    req.desc = (uint64_t)(uintptr_t)tensorDesc;
    req.format = (int32_t)format;
    req.dataType = (int32_t)dataType;
    req.n = n; req.c = c; req.h = h; req.w = w;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_TENSOR_4D_DESCRIPTOR,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType, int nbDims, const int *dimA, const int *strideA) {

    /*
     * Wire format:
     *   desc(8) + dataType(4) + nbDims(4) + dims(4*nbDims) + strides(4*nbDims)
     */
    size_t payload_len = 16 + (size_t)nbDims * 8;
    void *payload = malloc(payload_len);
    if (!payload) return CUDNN_STATUS_ALLOC_FAILED;

    char *p = (char *)payload;
    uint64_t desc = (uint64_t)(uintptr_t)tensorDesc;
    int32_t dt = (int32_t)dataType;
    int32_t nd = (int32_t)nbDims;
    memcpy(p, &desc, 8); p += 8;
    memcpy(p, &dt, 4); p += 4;
    memcpy(p, &nd, 4); p += 4;
    for (int i = 0; i < nbDims; i++) {
        int32_t d = dimA[i]; memcpy(p, &d, 4); p += 4;
    }
    for (int i = 0; i < nbDims; i++) {
        int32_t s = strideA[i]; memcpy(p, &s, 4); p += 4;
    }

    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_TENSOR_ND_DESCRIPTOR,
                                                      payload, payload_len);
    free(payload);
    return r;
}

cudnnStatus_t cudnnGetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t *dataType, int *n, int *c, int *h, int *w,
    int *nStride, int *cStride, int *hStride, int *wStride) {

    uint64_t desc = (uint64_t)(uintptr_t)tensorDesc;
    struct {
        int32_t dataType;
        int32_t n, c, h, w;
        int32_t nStride, cStride, hStride, wStride;
    } out;
    size_t len = sizeof(out);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(CHIDORI_CUDNN_GET_TENSOR_4D_DESCRIPTOR,
                                                &desc, sizeof(desc), &out, &len);
    if (r == CUDNN_STATUS_SUCCESS) {
        if (dataType) *dataType = (cudnnDataType_t)out.dataType;
        if (n) *n = out.n; if (c) *c = out.c;
        if (h) *h = out.h; if (w) *w = out.w;
        if (nStride) *nStride = out.nStride;
        if (cStride) *cStride = out.cStride;
        if (hStride) *hStride = out.hStride;
        if (wStride) *wStride = out.wStride;
    }
    return r;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
    uint64_t h = (uint64_t)(uintptr_t)tensorDesc;
    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_DESTROY_TENSOR_DESCRIPTOR,
                                           &h, sizeof(h));
}

/* ---- Filter Descriptor ---- */

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
    uint64_t h = 0;
    size_t len = sizeof(uint64_t);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(CHIDORI_CUDNN_CREATE_FILTER_DESCRIPTOR,
                                                NULL, 0, &h, &len);
    if (r == CUDNN_STATUS_SUCCESS)
        *filterDesc = (cudnnFilterDescriptor_t)(uintptr_t)h;
    return r;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType, cudnnTensorFormat_t format,
    int k, int c, int h, int w) {

    struct __attribute__((packed)) {
        uint64_t desc;
        int32_t  dataType;
        int32_t  format;
        int32_t  k, c, h, w;
    } req;

    req.desc = (uint64_t)(uintptr_t)filterDesc;
    req.dataType = (int32_t)dataType;
    req.format = (int32_t)format;
    req.k = k; req.c = c; req.h = h; req.w = w;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_FILTER_4D_DESCRIPTOR,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType, cudnnTensorFormat_t format,
    int nbDims, const int *filterDimA) {

    size_t payload_len = 16 + (size_t)nbDims * 4;
    void *payload = malloc(payload_len);
    if (!payload) return CUDNN_STATUS_ALLOC_FAILED;

    char *p = (char *)payload;
    uint64_t desc = (uint64_t)(uintptr_t)filterDesc;
    int32_t dt = (int32_t)dataType;
    int32_t fmt = (int32_t)format;
    int32_t nd = (int32_t)nbDims;
    memcpy(p, &desc, 8); p += 8;
    memcpy(p, &dt, 4); p += 4;
    memcpy(p, &fmt, 4); p += 4;
    /* nbDims is encoded implicitly by payload size, but send it explicitly */
    /* We need to adjust: desc(8) + dataType(4) + format(4) + nbDims(4) + dims(4*n) */
    /* Re-do: */
    free(payload);
    payload_len = 20 + (size_t)nbDims * 4;
    payload = malloc(payload_len);
    if (!payload) return CUDNN_STATUS_ALLOC_FAILED;
    p = (char *)payload;
    memcpy(p, &desc, 8); p += 8;
    memcpy(p, &dt, 4); p += 4;
    memcpy(p, &fmt, 4); p += 4;
    memcpy(p, &nd, 4); p += 4;
    for (int i = 0; i < nbDims; i++) {
        int32_t d = filterDimA[i]; memcpy(p, &d, 4); p += 4;
    }

    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_FILTER_ND_DESCRIPTOR,
                                                      payload, payload_len);
    free(payload);
    return r;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
    uint64_t h = (uint64_t)(uintptr_t)filterDesc;
    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_DESTROY_FILTER_DESCRIPTOR,
                                           &h, sizeof(h));
}

/* ---- Convolution Descriptor ---- */

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
    uint64_t h = 0;
    size_t len = sizeof(uint64_t);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(CHIDORI_CUDNN_CREATE_CONVOLUTION_DESCRIPTOR,
                                                NULL, 0, &h, &len);
    if (r == CUDNN_STATUS_SUCCESS)
        *convDesc = (cudnnConvolutionDescriptor_t)(uintptr_t)h;
    return r;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
    int pad_h, int pad_w, int u, int v, int dilation_h, int dilation_w,
    cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {

    struct __attribute__((packed)) {
        uint64_t desc;
        int32_t  pad_h, pad_w;
        int32_t  u, v;
        int32_t  dilation_h, dilation_w;
        int32_t  mode;
        int32_t  computeType;
    } req;

    req.desc = (uint64_t)(uintptr_t)convDesc;
    req.pad_h = pad_h; req.pad_w = pad_w;
    req.u = u; req.v = v;
    req.dilation_h = dilation_h; req.dilation_w = dilation_w;
    req.mode = (int32_t)mode;
    req.computeType = (int32_t)computeType;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
    int arrayLength, const int *padA, const int *filterStrideA,
    const int *dilationA, cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType) {

    /* desc(8) + arrayLength(4) + mode(4) + computeType(4) + pads(4*n) + strides(4*n) + dilations(4*n) */
    size_t payload_len = 20 + (size_t)arrayLength * 12;
    void *payload = malloc(payload_len);
    if (!payload) return CUDNN_STATUS_ALLOC_FAILED;

    char *p = (char *)payload;
    uint64_t desc = (uint64_t)(uintptr_t)convDesc;
    int32_t al = (int32_t)arrayLength;
    int32_t m = (int32_t)mode;
    int32_t ct = (int32_t)computeType;
    memcpy(p, &desc, 8); p += 8;
    memcpy(p, &al, 4); p += 4;
    memcpy(p, &m, 4); p += 4;
    memcpy(p, &ct, 4); p += 4;
    for (int i = 0; i < arrayLength; i++) {
        int32_t v = padA[i]; memcpy(p, &v, 4); p += 4;
    }
    for (int i = 0; i < arrayLength; i++) {
        int32_t v = filterStrideA[i]; memcpy(p, &v, 4); p += 4;
    }
    for (int i = 0; i < arrayLength; i++) {
        int32_t v = dilationA[i]; memcpy(p, &v, 4); p += 4;
    }

    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR,
                                                      payload, payload_len);
    free(payload);
    return r;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
    uint64_t h = (uint64_t)(uintptr_t)convDesc;
    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR,
                                           &h, sizeof(h));
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
    cudnnMathType_t mathType) {
    struct __attribute__((packed)) { uint64_t desc; int32_t mathType; } req;
    req.desc = (uint64_t)(uintptr_t)convDesc;
    req.mathType = (int32_t)mathType;
    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_CONVOLUTION_MATH_TYPE,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
    int groupCount) {
    struct __attribute__((packed)) { uint64_t desc; int32_t groupCount; } req;
    req.desc = (uint64_t)(uintptr_t)convDesc;
    req.groupCount = groupCount;
    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_CONVOLUTION_GROUP_COUNT,
                                           &req, sizeof(req));
}

/* ---- Convolution Forward Algorithm ---- */

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc,
    int requestedAlgoCount, int *returnedAlgoCount,
    void *perfResults) {

    struct __attribute__((packed)) {
        uint64_t handle;
        uint64_t xDesc;
        uint64_t wDesc;
        uint64_t convDesc;
        uint64_t yDesc;
        int32_t  requestedAlgoCount;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.wDesc = (uint64_t)(uintptr_t)wDesc;
    req.convDesc = (uint64_t)(uintptr_t)convDesc;
    req.yDesc = (uint64_t)(uintptr_t)yDesc;
    req.requestedAlgoCount = requestedAlgoCount;

    /*
     * Response: returnedAlgoCount(4) + perfResults array.
     * cudnnConvolutionFwdAlgoPerf_t is ~56 bytes each. We receive up to
     * requestedAlgoCount entries.
     */
    size_t out_len = 4 + (size_t)requestedAlgoCount * 56;
    void *out = malloc(out_len);
    if (!out) return CUDNN_STATUS_ALLOC_FAILED;

    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(CHIDORI_CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM,
                                                &req, sizeof(req), out, &out_len);
    if (r == CUDNN_STATUS_SUCCESS && out_len >= 4) {
        int32_t count;
        memcpy(&count, out, 4);
        if (returnedAlgoCount) *returnedAlgoCount = count;
        if (perfResults && out_len > 4)
            memcpy(perfResults, (char *)out + 4, out_len - 4);
    }
    free(out);
    return r;
}

/* ---- Convolution Workspace Size ---- */

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdAlgo_t algo, size_t *sizeInBytes) {

    struct __attribute__((packed)) {
        uint64_t handle;
        uint64_t xDesc;
        uint64_t wDesc;
        uint64_t convDesc;
        uint64_t yDesc;
        int32_t  algo;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.wDesc = (uint64_t)(uintptr_t)wDesc;
    req.convDesc = (uint64_t)(uintptr_t)convDesc;
    req.yDesc = (uint64_t)(uintptr_t)yDesc;
    req.algo = (int32_t)algo;

    uint64_t sz = 0;
    size_t len = sizeof(uint64_t);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(
        CHIDORI_CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE,
        &req, sizeof(req), &sz, &len);
    if (r == CUDNN_STATUS_SUCCESS)
        *sizeInBytes = (size_t)sz;
    return r;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle,
    cudnnFilterDescriptor_t wDesc, cudnnTensorDescriptor_t dyDesc,
    cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataAlgo_t algo, size_t *sizeInBytes) {

    struct __attribute__((packed)) {
        uint64_t handle;
        uint64_t wDesc;
        uint64_t dyDesc;
        uint64_t convDesc;
        uint64_t dxDesc;
        int32_t  algo;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.wDesc = (uint64_t)(uintptr_t)wDesc;
    req.dyDesc = (uint64_t)(uintptr_t)dyDesc;
    req.convDesc = (uint64_t)(uintptr_t)convDesc;
    req.dxDesc = (uint64_t)(uintptr_t)dxDesc;
    req.algo = (int32_t)algo;

    uint64_t sz = 0;
    size_t len = sizeof(uint64_t);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(
        CHIDORI_CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE,
        &req, sizeof(req), &sz, &len);
    if (r == CUDNN_STATUS_SUCCESS)
        *sizeInBytes = (size_t)sz;
    return r;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc, cudnnTensorDescriptor_t dyDesc,
    cudnnConvolutionDescriptor_t convDesc, cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {

    struct __attribute__((packed)) {
        uint64_t handle;
        uint64_t xDesc;
        uint64_t dyDesc;
        uint64_t convDesc;
        uint64_t gradDesc;
        int32_t  algo;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.dyDesc = (uint64_t)(uintptr_t)dyDesc;
    req.convDesc = (uint64_t)(uintptr_t)convDesc;
    req.gradDesc = (uint64_t)(uintptr_t)gradDesc;
    req.algo = (int32_t)algo;

    uint64_t sz = 0;
    size_t len = sizeof(uint64_t);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(
        CHIDORI_CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE,
        &req, sizeof(req), &sz, &len);
    if (r == CUDNN_STATUS_SUCCESS)
        *sizeInBytes = (size_t)sz;
    return r;
}

/* ---- Convolution Forward ---- */

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle,
    const void *alpha,
    cudnnTensorDescriptor_t xDesc, const void *x,
    cudnnFilterDescriptor_t wDesc, const void *w,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta,
    cudnnTensorDescriptor_t yDesc, void *y) {

    /*
     * Wire format:
     *   handle(8) + alpha(8) + xDesc(8) + x(8) + wDesc(8) + w(8)
     *   + convDesc(8) + algo(4) + workspace(8) + workspaceSize(8)
     *   + beta(8) + yDesc(8) + y(8) = 100 bytes
     *
     * alpha/beta sent as 8 bytes (double-sized buffer, covers float and double).
     */
    struct __attribute__((packed)) {
        uint64_t handle;
        uint8_t  alpha[8];
        uint64_t xDesc;
        uint64_t x;
        uint64_t wDesc;
        uint64_t w;
        uint64_t convDesc;
        int32_t  algo;
        uint64_t workSpace;
        uint64_t workSpaceSizeInBytes;
        uint8_t  beta[8];
        uint64_t yDesc;
        uint64_t y;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    memcpy(req.alpha, alpha, 8);
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.x = (uint64_t)(uintptr_t)x;
    req.wDesc = (uint64_t)(uintptr_t)wDesc;
    req.w = (uint64_t)(uintptr_t)w;
    req.convDesc = (uint64_t)(uintptr_t)convDesc;
    req.algo = (int32_t)algo;
    req.workSpace = (uint64_t)(uintptr_t)workSpace;
    req.workSpaceSizeInBytes = (uint64_t)workSpaceSizeInBytes;
    memcpy(req.beta, beta, 8);
    req.yDesc = (uint64_t)(uintptr_t)yDesc;
    req.y = (uint64_t)(uintptr_t)y;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_CONVOLUTION_FORWARD,
                                           &req, sizeof(req));
}

/* ---- Convolution Backward Data ---- */

cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t handle,
    const void *alpha,
    cudnnFilterDescriptor_t wDesc, const void *w,
    cudnnTensorDescriptor_t dyDesc, const void *dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta,
    cudnnTensorDescriptor_t dxDesc, void *dx) {

    struct __attribute__((packed)) {
        uint64_t handle;
        uint8_t  alpha[8];
        uint64_t wDesc;
        uint64_t w;
        uint64_t dyDesc;
        uint64_t dy;
        uint64_t convDesc;
        int32_t  algo;
        uint64_t workSpace;
        uint64_t workSpaceSizeInBytes;
        uint8_t  beta[8];
        uint64_t dxDesc;
        uint64_t dx;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    memcpy(req.alpha, alpha, 8);
    req.wDesc = (uint64_t)(uintptr_t)wDesc;
    req.w = (uint64_t)(uintptr_t)w;
    req.dyDesc = (uint64_t)(uintptr_t)dyDesc;
    req.dy = (uint64_t)(uintptr_t)dy;
    req.convDesc = (uint64_t)(uintptr_t)convDesc;
    req.algo = (int32_t)algo;
    req.workSpace = (uint64_t)(uintptr_t)workSpace;
    req.workSpaceSizeInBytes = (uint64_t)workSpaceSizeInBytes;
    memcpy(req.beta, beta, 8);
    req.dxDesc = (uint64_t)(uintptr_t)dxDesc;
    req.dx = (uint64_t)(uintptr_t)dx;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_CONVOLUTION_BACKWARD_DATA,
                                           &req, sizeof(req));
}

/* ---- Convolution Backward Filter ---- */

cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t handle,
    const void *alpha,
    cudnnTensorDescriptor_t xDesc, const void *x,
    cudnnTensorDescriptor_t dyDesc, const void *dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes,
    const void *beta,
    cudnnFilterDescriptor_t dwDesc, void *dw) {

    struct __attribute__((packed)) {
        uint64_t handle;
        uint8_t  alpha[8];
        uint64_t xDesc;
        uint64_t x;
        uint64_t dyDesc;
        uint64_t dy;
        uint64_t convDesc;
        int32_t  algo;
        uint64_t workSpace;
        uint64_t workSpaceSizeInBytes;
        uint8_t  beta[8];
        uint64_t dwDesc;
        uint64_t dw;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    memcpy(req.alpha, alpha, 8);
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.x = (uint64_t)(uintptr_t)x;
    req.dyDesc = (uint64_t)(uintptr_t)dyDesc;
    req.dy = (uint64_t)(uintptr_t)dy;
    req.convDesc = (uint64_t)(uintptr_t)convDesc;
    req.algo = (int32_t)algo;
    req.workSpace = (uint64_t)(uintptr_t)workSpace;
    req.workSpaceSizeInBytes = (uint64_t)workSpaceSizeInBytes;
    memcpy(req.beta, beta, 8);
    req.dwDesc = (uint64_t)(uintptr_t)dwDesc;
    req.dw = (uint64_t)(uintptr_t)dw;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_CONVOLUTION_BACKWARD_FILTER,
                                           &req, sizeof(req));
}

/* ---- Activation Descriptor ---- */

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
    uint64_t h = 0;
    size_t len = sizeof(uint64_t);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(CHIDORI_CUDNN_CREATE_ACTIVATION_DESCRIPTOR,
                                                NULL, 0, &h, &len);
    if (r == CUDNN_STATUS_SUCCESS)
        *activationDesc = (cudnnActivationDescriptor_t)(uintptr_t)h;
    return r;
}

cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
    cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt,
    double coef) {

    struct __attribute__((packed)) {
        uint64_t desc;
        int32_t  mode;
        int32_t  reluNanOpt;
        double   coef;
    } req;

    req.desc = (uint64_t)(uintptr_t)activationDesc;
    req.mode = (int32_t)mode;
    req.reluNanOpt = (int32_t)reluNanOpt;
    req.coef = coef;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_ACTIVATION_DESCRIPTOR,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
    uint64_t h = (uint64_t)(uintptr_t)activationDesc;
    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_DESTROY_ACTIVATION_DESCRIPTOR,
                                           &h, sizeof(h));
}

cudnnStatus_t cudnnActivationForward(cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta,
    cudnnTensorDescriptor_t yDesc, void *y) {

    struct __attribute__((packed)) {
        uint64_t handle;
        uint64_t activationDesc;
        uint8_t  alpha[8];
        uint64_t xDesc;
        uint64_t x;
        uint8_t  beta[8];
        uint64_t yDesc;
        uint64_t y;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.activationDesc = (uint64_t)(uintptr_t)activationDesc;
    memcpy(req.alpha, alpha, 8);
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.x = (uint64_t)(uintptr_t)x;
    memcpy(req.beta, beta, 8);
    req.yDesc = (uint64_t)(uintptr_t)yDesc;
    req.y = (uint64_t)(uintptr_t)y;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_ACTIVATION_FORWARD,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnActivationBackward(cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    cudnnTensorDescriptor_t yDesc, const void *y,
    cudnnTensorDescriptor_t dyDesc, const void *dy,
    cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta,
    cudnnTensorDescriptor_t dxDesc, void *dx) {

    struct __attribute__((packed)) {
        uint64_t handle;
        uint64_t activationDesc;
        uint8_t  alpha[8];
        uint64_t yDesc;
        uint64_t y;
        uint64_t dyDesc;
        uint64_t dy;
        uint64_t xDesc;
        uint64_t x;
        uint8_t  beta[8];
        uint64_t dxDesc;
        uint64_t dx;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.activationDesc = (uint64_t)(uintptr_t)activationDesc;
    memcpy(req.alpha, alpha, 8);
    req.yDesc = (uint64_t)(uintptr_t)yDesc;
    req.y = (uint64_t)(uintptr_t)y;
    req.dyDesc = (uint64_t)(uintptr_t)dyDesc;
    req.dy = (uint64_t)(uintptr_t)dy;
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.x = (uint64_t)(uintptr_t)x;
    memcpy(req.beta, beta, 8);
    req.dxDesc = (uint64_t)(uintptr_t)dxDesc;
    req.dx = (uint64_t)(uintptr_t)dx;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_ACTIVATION_BACKWARD,
                                           &req, sizeof(req));
}

/* ---- Pooling Descriptor ---- */

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
    uint64_t h = 0;
    size_t len = sizeof(uint64_t);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(CHIDORI_CUDNN_CREATE_POOLING_DESCRIPTOR,
                                                NULL, 0, &h, &len);
    if (r == CUDNN_STATUS_SUCCESS)
        *poolingDesc = (cudnnPoolingDescriptor_t)(uintptr_t)h;
    return r;
}

cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
    cudnnPoolingMode_t mode, cudnnNanPropagation_t maxpoolingNanOpt,
    int windowHeight, int windowWidth,
    int verticalPadding, int horizontalPadding,
    int verticalStride, int horizontalStride) {

    struct __attribute__((packed)) {
        uint64_t desc;
        int32_t  mode;
        int32_t  maxpoolingNanOpt;
        int32_t  windowHeight, windowWidth;
        int32_t  verticalPadding, horizontalPadding;
        int32_t  verticalStride, horizontalStride;
    } req;

    req.desc = (uint64_t)(uintptr_t)poolingDesc;
    req.mode = (int32_t)mode;
    req.maxpoolingNanOpt = (int32_t)maxpoolingNanOpt;
    req.windowHeight = windowHeight;
    req.windowWidth = windowWidth;
    req.verticalPadding = verticalPadding;
    req.horizontalPadding = horizontalPadding;
    req.verticalStride = verticalStride;
    req.horizontalStride = horizontalStride;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_POOLING_2D_DESCRIPTOR,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
    uint64_t h = (uint64_t)(uintptr_t)poolingDesc;
    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_DESTROY_POOLING_DESCRIPTOR,
                                           &h, sizeof(h));
}

cudnnStatus_t cudnnPoolingForward(cudnnHandle_t handle,
    cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha,
    cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta,
    cudnnTensorDescriptor_t yDesc, void *y) {

    struct __attribute__((packed)) {
        uint64_t handle;
        uint64_t poolingDesc;
        uint8_t  alpha[8];
        uint64_t xDesc;
        uint64_t x;
        uint8_t  beta[8];
        uint64_t yDesc;
        uint64_t y;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.poolingDesc = (uint64_t)(uintptr_t)poolingDesc;
    memcpy(req.alpha, alpha, 8);
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.x = (uint64_t)(uintptr_t)x;
    memcpy(req.beta, beta, 8);
    req.yDesc = (uint64_t)(uintptr_t)yDesc;
    req.y = (uint64_t)(uintptr_t)y;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_POOLING_FORWARD,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t handle,
    cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha,
    cudnnTensorDescriptor_t yDesc, const void *y,
    cudnnTensorDescriptor_t dyDesc, const void *dy,
    cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta,
    cudnnTensorDescriptor_t dxDesc, void *dx) {

    struct __attribute__((packed)) {
        uint64_t handle;
        uint64_t poolingDesc;
        uint8_t  alpha[8];
        uint64_t yDesc;
        uint64_t y;
        uint64_t dyDesc;
        uint64_t dy;
        uint64_t xDesc;
        uint64_t x;
        uint8_t  beta[8];
        uint64_t dxDesc;
        uint64_t dx;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.poolingDesc = (uint64_t)(uintptr_t)poolingDesc;
    memcpy(req.alpha, alpha, 8);
    req.yDesc = (uint64_t)(uintptr_t)yDesc;
    req.y = (uint64_t)(uintptr_t)y;
    req.dyDesc = (uint64_t)(uintptr_t)dyDesc;
    req.dy = (uint64_t)(uintptr_t)dy;
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.x = (uint64_t)(uintptr_t)x;
    memcpy(req.beta, beta, 8);
    req.dxDesc = (uint64_t)(uintptr_t)dxDesc;
    req.dx = (uint64_t)(uintptr_t)dx;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_POOLING_BACKWARD,
                                           &req, sizeof(req));
}

/* ---- Batch Normalization ---- */

cudnnStatus_t cudnnBatchNormalizationForwardTraining(cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alpha, const void *beta,
    cudnnTensorDescriptor_t xDesc, const void *x,
    cudnnTensorDescriptor_t yDesc, void *y,
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale, const void *bnBias,
    double exponentialAverageFactor,
    void *resultRunningMean, void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  mode;
        uint8_t  alpha[8];
        uint8_t  beta[8];
        uint64_t xDesc;
        uint64_t x;
        uint64_t yDesc;
        uint64_t y;
        uint64_t bnScaleBiasMeanVarDesc;
        uint64_t bnScale;
        uint64_t bnBias;
        double   exponentialAverageFactor;
        uint64_t resultRunningMean;
        uint64_t resultRunningVariance;
        double   epsilon;
        uint64_t resultSaveMean;
        uint64_t resultSaveInvVariance;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.mode = (int32_t)mode;
    memcpy(req.alpha, alpha, 8);
    memcpy(req.beta, beta, 8);
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.x = (uint64_t)(uintptr_t)x;
    req.yDesc = (uint64_t)(uintptr_t)yDesc;
    req.y = (uint64_t)(uintptr_t)y;
    req.bnScaleBiasMeanVarDesc = (uint64_t)(uintptr_t)bnScaleBiasMeanVarDesc;
    req.bnScale = (uint64_t)(uintptr_t)bnScale;
    req.bnBias = (uint64_t)(uintptr_t)bnBias;
    req.exponentialAverageFactor = exponentialAverageFactor;
    req.resultRunningMean = (uint64_t)(uintptr_t)resultRunningMean;
    req.resultRunningVariance = (uint64_t)(uintptr_t)resultRunningVariance;
    req.epsilon = epsilon;
    req.resultSaveMean = (uint64_t)(uintptr_t)resultSaveMean;
    req.resultSaveInvVariance = (uint64_t)(uintptr_t)resultSaveInvVariance;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_BATCH_NORM_FORWARD_TRAINING,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alpha, const void *beta,
    cudnnTensorDescriptor_t xDesc, const void *x,
    cudnnTensorDescriptor_t yDesc, void *y,
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale, const void *bnBias,
    const void *estimatedMean, const void *estimatedVariance,
    double epsilon) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  mode;
        uint8_t  alpha[8];
        uint8_t  beta[8];
        uint64_t xDesc;
        uint64_t x;
        uint64_t yDesc;
        uint64_t y;
        uint64_t bnScaleBiasMeanVarDesc;
        uint64_t bnScale;
        uint64_t bnBias;
        uint64_t estimatedMean;
        uint64_t estimatedVariance;
        double   epsilon;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.mode = (int32_t)mode;
    memcpy(req.alpha, alpha, 8);
    memcpy(req.beta, beta, 8);
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.x = (uint64_t)(uintptr_t)x;
    req.yDesc = (uint64_t)(uintptr_t)yDesc;
    req.y = (uint64_t)(uintptr_t)y;
    req.bnScaleBiasMeanVarDesc = (uint64_t)(uintptr_t)bnScaleBiasMeanVarDesc;
    req.bnScale = (uint64_t)(uintptr_t)bnScale;
    req.bnBias = (uint64_t)(uintptr_t)bnBias;
    req.estimatedMean = (uint64_t)(uintptr_t)estimatedMean;
    req.estimatedVariance = (uint64_t)(uintptr_t)estimatedVariance;
    req.epsilon = epsilon;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_BATCH_NORM_FORWARD_INFERENCE,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnBatchNormalizationBackward(cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    cudnnTensorDescriptor_t xDesc, const void *x,
    cudnnTensorDescriptor_t dyDesc, const void *dy,
    cudnnTensorDescriptor_t dxDesc, void *dx,
    cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScale,
    void *dBnScaleResult, void *dBnBiasResult,
    double epsilon,
    const void *savedMean, const void *savedInvVariance) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  mode;
        uint8_t  alphaDataDiff[8];
        uint8_t  betaDataDiff[8];
        uint8_t  alphaParamDiff[8];
        uint8_t  betaParamDiff[8];
        uint64_t xDesc;
        uint64_t x;
        uint64_t dyDesc;
        uint64_t dy;
        uint64_t dxDesc;
        uint64_t dx;
        uint64_t dBnScaleBiasDesc;
        uint64_t bnScale;
        uint64_t dBnScaleResult;
        uint64_t dBnBiasResult;
        double   epsilon;
        uint64_t savedMean;
        uint64_t savedInvVariance;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.mode = (int32_t)mode;
    memcpy(req.alphaDataDiff, alphaDataDiff, 8);
    memcpy(req.betaDataDiff, betaDataDiff, 8);
    memcpy(req.alphaParamDiff, alphaParamDiff, 8);
    memcpy(req.betaParamDiff, betaParamDiff, 8);
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.x = (uint64_t)(uintptr_t)x;
    req.dyDesc = (uint64_t)(uintptr_t)dyDesc;
    req.dy = (uint64_t)(uintptr_t)dy;
    req.dxDesc = (uint64_t)(uintptr_t)dxDesc;
    req.dx = (uint64_t)(uintptr_t)dx;
    req.dBnScaleBiasDesc = (uint64_t)(uintptr_t)dBnScaleBiasDesc;
    req.bnScale = (uint64_t)(uintptr_t)bnScale;
    req.dBnScaleResult = (uint64_t)(uintptr_t)dBnScaleResult;
    req.dBnBiasResult = (uint64_t)(uintptr_t)dBnBiasResult;
    req.epsilon = epsilon;
    req.savedMean = (uint64_t)(uintptr_t)savedMean;
    req.savedInvVariance = (uint64_t)(uintptr_t)savedInvVariance;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_BATCH_NORM_BACKWARD,
                                           &req, sizeof(req));
}

/* ---- Softmax ---- */

cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha,
    cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta,
    cudnnTensorDescriptor_t yDesc, void *y) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  algo;
        int32_t  mode;
        uint8_t  alpha[8];
        uint64_t xDesc;
        uint64_t x;
        uint8_t  beta[8];
        uint64_t yDesc;
        uint64_t y;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.algo = (int32_t)algo;
    req.mode = (int32_t)mode;
    memcpy(req.alpha, alpha, 8);
    req.xDesc = (uint64_t)(uintptr_t)xDesc;
    req.x = (uint64_t)(uintptr_t)x;
    memcpy(req.beta, beta, 8);
    req.yDesc = (uint64_t)(uintptr_t)yDesc;
    req.y = (uint64_t)(uintptr_t)y;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SOFTMAX_FORWARD,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha,
    cudnnTensorDescriptor_t yDesc, const void *y,
    cudnnTensorDescriptor_t dyDesc, const void *dy,
    const void *beta,
    cudnnTensorDescriptor_t dxDesc, void *dx) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  algo;
        int32_t  mode;
        uint8_t  alpha[8];
        uint64_t yDesc;
        uint64_t y;
        uint64_t dyDesc;
        uint64_t dy;
        uint8_t  beta[8];
        uint64_t dxDesc;
        uint64_t dx;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.algo = (int32_t)algo;
    req.mode = (int32_t)mode;
    memcpy(req.alpha, alpha, 8);
    req.yDesc = (uint64_t)(uintptr_t)yDesc;
    req.y = (uint64_t)(uintptr_t)y;
    req.dyDesc = (uint64_t)(uintptr_t)dyDesc;
    req.dy = (uint64_t)(uintptr_t)dy;
    memcpy(req.beta, beta, 8);
    req.dxDesc = (uint64_t)(uintptr_t)dxDesc;
    req.dx = (uint64_t)(uintptr_t)dx;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SOFTMAX_BACKWARD,
                                           &req, sizeof(req));
}

/* ---- Dropout ---- */

cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) {
    uint64_t h = 0;
    size_t len = sizeof(uint64_t);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(CHIDORI_CUDNN_CREATE_DROPOUT_DESCRIPTOR,
                                                NULL, 0, &h, &len);
    if (r == CUDNN_STATUS_SUCCESS)
        *dropoutDesc = (cudnnDropoutDescriptor_t)(uintptr_t)h;
    return r;
}

cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle, float dropout,
    void *states, size_t stateSizeInBytes, unsigned long long seed) {

    struct __attribute__((packed)) {
        uint64_t desc;
        uint64_t handle;
        float    dropout;
        uint64_t states;
        uint64_t stateSizeInBytes;
        uint64_t seed;
    } req;

    req.desc = (uint64_t)(uintptr_t)dropoutDesc;
    req.handle = (uint64_t)(uintptr_t)handle;
    req.dropout = dropout;
    req.states = (uint64_t)(uintptr_t)states;
    req.stateSizeInBytes = (uint64_t)stateSizeInBytes;
    req.seed = (uint64_t)seed;

    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_SET_DROPOUT_DESCRIPTOR,
                                           &req, sizeof(req));
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
    uint64_t h = (uint64_t)(uintptr_t)dropoutDesc;
    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_DESTROY_DROPOUT_DESCRIPTOR,
                                           &h, sizeof(h));
}

/* ---- RNN Descriptor (stub lifecycle) ---- */

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
    uint64_t h = 0;
    size_t len = sizeof(uint64_t);
    cudnnStatus_t r = (cudnnStatus_t)chidori_rpc(CHIDORI_CUDNN_CREATE_RNN_DESCRIPTOR,
                                                NULL, 0, &h, &len);
    if (r == CUDNN_STATUS_SUCCESS)
        *rnnDesc = (cudnnRNNDescriptor_t)(uintptr_t)h;
    return r;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
    uint64_t h = (uint64_t)(uintptr_t)rnnDesc;
    return (cudnnStatus_t)chidori_rpc_async(CHIDORI_CUDNN_DESTROY_RNN_DESCRIPTOR,
                                           &h, sizeof(h));
}

/*
 * ============================================================================
 * PyTorch Compatibility Stubs — CUDA Runtime API
 *
 * These stubs provide symbols that PyTorch (and its dependencies) reference
 * at load time. On a GPU-less client with LD_PRELOAD, these must exist or
 * the process will crash with "undefined symbol" errors.
 *
 * Most are trivial stubs returning success. Critical ones (pinned memory,
 * streams, events) do local work or forward to the server.
 * ============================================================================
 */

/* ---- Error Handling ---- */

const char *cudaGetErrorString(cudaError_t error) {
    switch (error) {
    case 0:  return "no error";
    case 1:  return "invalid argument";
    case 2:  return "out of memory";
    case 3:  return "not initialized";
    case 4:  return "deinitialized";
    case 35: return "insufficient driver";
    case 46: return "device not found";
    case 71: return "launch failed";
    case 77: return "peer access not enabled";
    case 100: return "no CUDA-capable device";
    case 101: return "invalid device";
    case 700: return "illegal address";
    case 999: return "unknown error";
    default: return "unrecognized error code";
    }
}

const char *cudaGetErrorName(cudaError_t error) {
    switch (error) {
    case 0:  return "cudaSuccess";
    case 1:  return "cudaErrorInvalidValue";
    case 2:  return "cudaErrorMemoryAllocation";
    case 3:  return "cudaErrorInitializationError";
    case 4:  return "cudaErrorCudartUnloading";
    case 35: return "cudaErrorInsufficientDriver";
    case 46: return "cudaErrorDevicesUnavailable";
    case 71: return "cudaErrorLaunchFailure";
    case 77: return "cudaErrorPeerAccessNotEnabled";
    case 100: return "cudaErrorNoDevice";
    case 101: return "cudaErrorInvalidDevice";
    case 700: return "cudaErrorIllegalAddress";
    case 999: return "cudaErrorUnknown";
    default: return "cudaErrorUnknown";
    }
}

/* ---- Pinned (Host) Memory ---- */

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    *ptr = malloc(size);
    g_last_error = *ptr ? cudaSuccess : 2; /* cudaErrorMemoryAllocation */
    return g_last_error;
}

cudaError_t cudaFreeHost(void *ptr) {
    free(ptr);
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
    (void)flags;
    *pHost = malloc(size);
    g_last_error = *pHost ? cudaSuccess : 2;
    return g_last_error;
}

cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
    (void)flags;
    *pDevice = pHost;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    (void)ptr; (void)size; (void)flags;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaHostUnregister(void *ptr) {
    (void)ptr;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* ---- Async Memory (CUDA 11.2+) ---- */

cudaError_t cudaMallocAsync(void **devPtr, size_t size, void *stream) {
    (void)stream;
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaFreeAsync(void *devPtr, void *stream) {
    (void)stream;
    return cudaFree(devPtr);
}

/* ---- 2D Memory ---- */

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch,
                         const void *src, size_t spitch,
                         size_t width, size_t height, int kind) {
    for (size_t row = 0; row < height; row++) {
        cudaError_t r = cudaMemcpy(
            (char *)dst + row * dpitch,
            (const char *)src + row * spitch,
            width, kind);
        if (r != cudaSuccess) return r;
    }
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch,
                              const void *src, size_t spitch,
                              size_t width, size_t height, int kind,
                              void *stream) {
    for (size_t row = 0; row < height; row++) {
        cudaError_t r = cudaMemcpyAsync(
            (char *)dst + row * dpitch,
            (const char *)src + row * spitch,
            width, kind, stream);
        if (r != cudaSuccess) return r;
    }
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* ---- Peer Memory ---- */

cudaError_t cudaMemcpyPeer(void *dst, int dstDevice,
                           const void *src, int srcDevice, size_t count) {
    /* Same-device: direct DtoD */
    if (srcDevice == dstDevice) {
        return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
    }
    /* Cross-device: stage through CPU buffer.
     * One device is local (real GPU), the other is remote (RPC).
     * They can't access each other's memory directly. */
    void *staging = malloc(count);
    if (!staging) return (cudaError_t)2; /* cudaErrorMemoryAllocation */
    int saved_dev = g_current_device;

    /* Step 1: DtoH from source device */
    cudaSetDevice(srcDevice);
    cudaError_t r = cudaMemcpy(staging, src, count, cudaMemcpyDeviceToHost);
    if (r != cudaSuccess) { free(staging); cudaSetDevice(saved_dev); return r; }

    /* Step 2: HtoD to destination device */
    cudaSetDevice(dstDevice);
    r = cudaMemcpy(dst, staging, count, cudaMemcpyHostToDevice);
    free(staging);
    cudaSetDevice(saved_dev);
    return r;
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice,
                                const void *src, int srcDevice,
                                size_t count, void *stream) {
    /* Async cross-device: fall back to sync staging for now */
    (void)stream;
    return cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
}

/* ---- Async Memset ---- */

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, void *stream) {
    (void)stream;
    return cudaMemset(devPtr, value, count);
}

/* ---- Device Management (additional) ---- */

cudaError_t cudaGetDeviceProperties_v2(void *prop, int device) {
    return cudaGetDeviceProperties(prop, device);
}

cudaError_t cudaDeviceGetDefaultMemPool(void **memPool, int device) {
    (void)device;
    static int fake_pool;
    *memPool = (void *)&fake_pool;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceSetMemPool(int device, void *memPool) {
    (void)device; (void)memPool;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetMemPool(void **memPool, int device) {
    return cudaDeviceGetDefaultMemPool(memPool, device);
}

/* cudaLimit enum values */
#define cudaLimitStackSize           0x00
#define cudaLimitPrintfFifoSize      0x01
#define cudaLimitMallocHeapSize      0x02

cudaError_t cudaDeviceSetLimit(int limit, size_t value) {
    (void)limit; (void)value;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, int limit) {
    switch (limit) {
    case cudaLimitStackSize:       *pValue = 1024; break;
    case cudaLimitPrintfFifoSize:  *pValue = 1048576; break;
    case cudaLimitMallocHeapSize:  *pValue = 8388608; break;
    default:                       *pValue = 0; break;
    }
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = -1;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetP2PAttribute(int *value, int attr, int srcDevice, int dstDevice) {
    (void)attr; (void)srcDevice; (void)dstDevice;
    if (value) *value = 0;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
    (void)pciBusId;
    if (device) *device = 0;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    (void)device;
    if (pciBusId && len > 0) snprintf(pciBusId, (size_t)len, "0000:00:00.0");
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    (void)peerDevice; (void)flags;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    (void)peerDevice;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* ---- IPC ---- */

cudaError_t cudaIpcGetMemHandle(void *handle, void *devPtr) {
    (void)devPtr;
    if (handle) memset(handle, 0, 64);
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaIpcOpenMemHandle(void **devPtr, void *handle, unsigned int flags) {
    (void)handle; (void)flags;
    *devPtr = NULL;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaIpcCloseMemHandle(void *devPtr) {
    (void)devPtr;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaIpcGetEventHandle(void *handle, void *event) {
    (void)event;
    if (handle) memset(handle, 0, 64);
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaIpcOpenEventHandle(void **event, void *handle) {
    (void)handle;
    *event = NULL;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* ---- Stream Management (additional) ---- */

cudaError_t cudaStreamCreateWithFlags(void **pStream, unsigned int flags) {
    (void)flags;
    return cudaStreamCreate(pStream);
}

cudaError_t cudaStreamCreateWithPriority(void **pStream, unsigned int flags, int priority) {
    (void)flags; (void)priority;
    return cudaStreamCreate(pStream);
}

cudaError_t cudaStreamGetFlags(void *stream, unsigned int *flags) {
    (void)stream;
    if (flags) *flags = 0;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamGetPriority(void *stream, int *priority) {
    (void)stream;
    if (priority) *priority = 0;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(void *stream) {
    (void)stream;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(void *stream, void *event, unsigned int flags) {
    (void)flags; (void)stream;
    /* For cross-device copy: if event is local, synchronize it directly.
     * Must temporarily activate local context — it may have been deactivated
     * when we switched to the remote device. */
    if (g_partial_offload && event) {
        /* Remote events are small RPC handles (< 0x100000).
         * Local events are heap-allocated pointers (large addresses).
         * Only pass local events to the real driver. For remote events,
         * just synchronize the device instead (ensures ordering). */
        uintptr_t ev = (uintptr_t)event;
        if (ev > 0x100000 && g_real.cuEventSynchronize && g_local_ctx) {
            /* Local event: sync via real driver with local context active */
            g_in_real_cuda++;
            CUcontext saved_ctx = NULL;
            if (g_real.cuCtxGetCurrent) g_real.cuCtxGetCurrent(&saved_ctx);
            if (g_real.cuCtxSetCurrent) g_real.cuCtxSetCurrent(g_local_ctx);
            CUresult r = g_real.cuEventSynchronize((CUevent)event);
            if (g_real.cuCtxSetCurrent) g_real.cuCtxSetCurrent(saved_ctx);
            g_in_real_cuda--;
            if (r == CUDA_SUCCESS) { g_last_error = cudaSuccess; return cudaSuccess; }
        } else if (ev <= 0x100000) {
            /* Remote event: sync by synchronizing the remote device */
            if (!is_local_device(g_current_device)) {
                cudaError_t r = cudaDeviceSynchronize();
                g_last_error = r;
                return r;
            }
        }
    }
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(void *stream, void *callback, void *userData, unsigned int flags) {
    (void)stream; (void)callback; (void)userData; (void)flags;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(void *stream, int *pCaptureStatus) {
    (void)stream;
    if (pCaptureStatus) *pCaptureStatus = 0; /* cudaStreamCaptureStatusNone */
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo(void *stream, int *captureStatus, unsigned long long *id) {
    (void)stream;
    if (captureStatus) *captureStatus = 0;
    if (id) *id = 0;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaThreadExchangeStreamCaptureMode(int *mode) {
    (void)mode;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* ---- Event Management (additional) ---- */

cudaError_t cudaEventCreateWithFlags(void **pEvent, unsigned int flags) {
    /* Local device: create with real driver using actual flags */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuEventCreate) {
        g_in_real_cuda++;
        CUresult r = g_real.cuEventCreate((CUevent *)pEvent, flags);
        g_in_real_cuda--;
        g_last_error = (cudaError_t)r;
        return (cudaError_t)r;
    }
    return cudaEventCreate(pEvent);
}

cudaError_t cudaEventElapsedTime(float *ms, void *start, void *end) {
    struct { uint64_t start; uint64_t end; } req;
    req.start = (uint64_t)(uintptr_t)start;
    req.end = (uint64_t)(uintptr_t)end;
    size_t len = sizeof(float);
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_EVENT_ELAPSED_TIME, &req, sizeof(req), ms, &len);
    g_last_error = r;
    return r;
}

cudaError_t cudaEventQuery(void *event) {
    (void)event;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* ---- Execution (Runtime API) ---- */

/*
 * Lazy kernel registration: store fat binaries and function registrations
 * locally, only send to server on first cudaLaunchKernel call per function.
 * This avoids 1400+ synchronous RPCs during PyTorch init.
 */

/* Stored fat binary data (not yet sent to server) */
#define MAX_FAT_BINARIES 1024
static struct {
    const void *data;       /* raw fat binary data pointer (lives in process memory) */
    uint64_t size;          /* fat binary size */
    uint64_t server_handle; /* 0 = not yet registered on server */
    void **real_handle;     /* handle from real CUDA runtime (for local execution) */
} g_fat_binaries[MAX_FAT_BINARIES];
static int g_fat_binary_count = 0;

/* Stored function registrations (not yet sent to server) */
/* (forward-declared before, defined here after g_fat_binaries is available) */
#define MAX_REGISTERED_FUNCTIONS 65536
#define MAX_KERNEL_PARAMS 64
static struct {
    const void *hostFun;        /* host function pointer (PyTorch's kernel ID) */
    int fat_binary_idx;         /* index into g_fat_binaries */
    char *deviceName;           /* kernel name on device (strdup'd, PyTorch names can be 300+ chars) */
    uint64_t server_handle;     /* 0 = not yet registered on server */
    int32_t total_param_size;   /* total parameter buffer size */
    int32_t num_params;         /* number of kernel parameters */
    struct __attribute__((packed)) { int32_t offset; int32_t size; } params[MAX_KERNEL_PARAMS];
} g_registered_functions[MAX_REGISTERED_FUNCTIONS];
static int g_registered_function_count = 0;

/*
 * Replay deferred fat binary and function registrations.
 * Called from cuInit AFTER init_real_cuda() and g_real.cuInit(0) have completed.
 * This ensures the real CUDA driver is fully initialized before we register
 * any fat binaries or functions with it.
 */
static void replay_deferred_registrations(void) {
    if (g_deferred_replay_done) return;
    g_deferred_replay_done = 1;

    if (!g_partial_offload || !g_real.__cudaRegisterFatBinary) {
        LOG_DEBUG("deferred replay: no partial offload or no real __cudaRegisterFatBinary, skipping");
        return;
    }

    fprintf(stderr, "[chidori] deferred replay: replaying %d fat binaries, %d fatbin ends, %d functions\n",
            g_pending_fatbin_count, g_pending_fatbin_end_count, g_pending_func_count);

    /* Phase 1: Replay __cudaRegisterFatBinary calls */
    for (int i = 0; i < g_pending_fatbin_count; i++) {
        int idx = g_pending_fatbins[i].idx;
        void *fatCubin = g_pending_fatbins[i].fatCubin;
        g_in_real_cuda++;
        g_fat_binaries[idx].real_handle = g_real.__cudaRegisterFatBinary(fatCubin);
        g_in_real_cuda--;
        LOG_DEBUG("deferred replay: __cudaRegisterFatBinary[%d] -> real_handle=%p", idx,
                  (void *)g_fat_binaries[idx].real_handle);
    }

    /* Phase 2: Replay __cudaRegisterFatBinaryEnd calls */
    if (g_real.__cudaRegisterFatBinaryEnd) {
        for (int i = 0; i < g_pending_fatbin_end_count; i++) {
            int idx = g_pending_fatbin_ends[i];
            if (idx >= 0 && idx < g_fat_binary_count && g_fat_binaries[idx].real_handle) {
                g_in_real_cuda++;
                g_real.__cudaRegisterFatBinaryEnd(g_fat_binaries[idx].real_handle);
                g_in_real_cuda--;
            }
        }
    }

    /* Phase 3: Replay __cudaRegisterFunction calls */
    if (g_real.__cudaRegisterFunction) {
        for (int i = 0; i < g_pending_func_count; i++) {
            int fat_idx = g_pending_funcs[i].fat_idx;
            void **real_handle = (fat_idx >= 0 && fat_idx < g_fat_binary_count)
                                 ? g_fat_binaries[fat_idx].real_handle : NULL;
            if (real_handle) {
                g_in_real_cuda++;
                g_real.__cudaRegisterFunction(real_handle,
                    g_pending_funcs[i].hostFun,
                    g_pending_funcs[i].deviceFun,
                    g_pending_funcs[i].deviceName,
                    g_pending_funcs[i].thread_limit,
                    g_pending_funcs[i].tid,
                    g_pending_funcs[i].bid,
                    g_pending_funcs[i].bDim,
                    g_pending_funcs[i].gDim,
                    g_pending_funcs[i].wSize);
                g_in_real_cuda--;
            }
        }
    }

    fprintf(stderr, "[chidori] deferred replay: complete\n");
}

/* Lazily register a fat binary on the server, returns server module handle */
static uint64_t lazy_register_fat_binary(int idx) {
    if (g_fat_binaries[idx].server_handle != 0)
        return g_fat_binaries[idx].server_handle;

    uint64_t handle = 0;
    size_t out_len = sizeof(handle);
    CUresult r = chidori_rpc(CHIDORI_CUDA_REGISTER_FAT_BINARY,
                             g_fat_binaries[idx].data,
                             (size_t)g_fat_binaries[idx].size,
                             &handle, &out_len);
    if (r != 0 || handle == 0) {
        const unsigned char *d = (const unsigned char *)g_fat_binaries[idx].data;
        uint32_t mag = 0;
        memcpy(&mag, d, 4);
        fprintf(stderr, "[chidori] fat_binary[%d] FAILED: size=%zu r=%d handle=%llu magic=0x%08x\n",
                idx, (size_t)g_fat_binaries[idx].size, r, (unsigned long long)handle, mag);
    }
    if (r == 0 && handle != 0)
        g_fat_binaries[idx].server_handle = handle;
    return handle;
}

/* Lazily register a function on the server, returns server CUfunction handle */
static uint64_t lazy_register_function(int idx) {
    if (g_registered_functions[idx].server_handle != 0)
        return g_registered_functions[idx].server_handle;

    /* First ensure the fat binary is registered */
    int fb_idx = g_registered_functions[idx].fat_binary_idx;
    uint64_t mod_handle = lazy_register_fat_binary(fb_idx);
    if (mod_handle == 0) return 0;

    /* Register the function */
    const char *name = g_registered_functions[idx].deviceName;
    size_t name_len = strlen(name) + 1;
    size_t payload_len = 8 + name_len;
    void *payload = malloc(payload_len);
    memcpy(payload, &mod_handle, 8);
    memcpy((char *)payload + 8, name, name_len);

    /*
     * Server returns: func_handle(8) + total_param_size(4) + num_params(4)
     *                 + [offset(4) + size(4)] × num_params
     */
    uint8_t resp_buf[8 + 4 + 4 + MAX_KERNEL_PARAMS * 8];
    memset(resp_buf, 0, sizeof(resp_buf));
    size_t out_len = sizeof(resp_buf);
    chidori_rpc(CHIDORI_CUDA_REGISTER_FUNCTION, payload, payload_len, resp_buf, &out_len);
    free(payload);

    uint64_t func_h = 0;
    memcpy(&func_h, resp_buf, 8);
    if (func_h != 0) {
        g_registered_functions[idx].server_handle = func_h;
        memcpy(&g_registered_functions[idx].total_param_size, resp_buf + 8, 4);
        memcpy(&g_registered_functions[idx].num_params, resp_buf + 12, 4);
        int np = g_registered_functions[idx].num_params;
        if (np > MAX_KERNEL_PARAMS) np = MAX_KERNEL_PARAMS;
        for (int i = 0; i < np; i++) {
            memcpy(&g_registered_functions[idx].params[i].offset, resp_buf + 16 + i * 8, 4);
            memcpy(&g_registered_functions[idx].params[i].size, resp_buf + 16 + i * 8 + 4, 4);
        }
        if (g_verbose) {
            LOG_DEBUG("Registered '%s' handle=0x%llx params=%d total=%d",
                    g_registered_functions[idx].deviceName,
                    (unsigned long long)func_h, np,
                    g_registered_functions[idx].total_param_size);
            for (int i = 0; i < np; i++)
                LOG_DEBUG("  param[%d] offset=%d size=%d",
                        i, g_registered_functions[idx].params[i].offset,
                        g_registered_functions[idx].params[i].size);
        }
    }
    return func_h;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream) {
    /* Route to local GPU if current device is local */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cudaLaunchKernel_ptr) {
        LOG_DEBUG("cudaLaunchKernel → LOCAL (dev %d)", g_current_device);
        typedef int (*real_cudaLaunchKernel_t)(const void *, dim3, dim3, void **, size_t, void *);
        real_cudaLaunchKernel_t real_fn = (real_cudaLaunchKernel_t)g_real.cudaLaunchKernel_ptr;
        g_in_real_cuda++;
        cudaError_t r = (cudaError_t)real_fn(func, gridDim, blockDim, args, sharedMem, stream);
        g_in_real_cuda--;
        g_last_error = r;
        return r;
    }

    /* Look up function (lazy register on first use) */
    uint64_t func_handle = 0;
    int func_idx = -1;
    for (int i = 0; i < g_registered_function_count; i++) {
        if (g_registered_functions[i].hostFun == func) {
            func_handle = lazy_register_function(i);
            func_idx = i;
            break;
        }
    }

    if (func_handle == 0 || func_idx < 0) {
        /* Diagnostic: find closest registered function to understand the mismatch */
        fprintf(stderr, "[chidori-launch] UNREGISTERED func=%p (%d functions registered)\n",
                func, g_registered_function_count);
        uintptr_t target = (uintptr_t)func;
        uintptr_t closest_dist = (uintptr_t)-1;
        int closest_idx = -1;
        for (int i = 0; i < g_registered_function_count; i++) {
            uintptr_t addr = (uintptr_t)g_registered_functions[i].hostFun;
            uintptr_t dist = (target > addr) ? (target - addr) : (addr - target);
            if (dist < closest_dist) {
                closest_dist = dist;
                closest_idx = i;
            }
        }
        if (closest_idx >= 0) {
            fprintf(stderr, "[chidori-launch]   closest: %p '%s' (delta=%ld)\n",
                    g_registered_functions[closest_idx].hostFun,
                    g_registered_functions[closest_idx].deviceName,
                    (long)(target - (uintptr_t)g_registered_functions[closest_idx].hostFun));
        }
        g_last_error = cudaErrorInvalidDeviceFunction;
        return cudaErrorInvalidDeviceFunction;
    }

    int32_t total_ps = g_registered_functions[func_idx].total_param_size;
    int32_t num_params = g_registered_functions[func_idx].num_params;
    if (args == NULL) { total_ps = 0; num_params = 0; }

    /*
     * Pack kernel parameters using EXACT layout from cuFuncGetParamInfo.
     * Each param has a known offset and size in the contiguous buffer.
     * args[i] points to the value of parameter i.
     * We copy args[i] (size bytes) into param_buf at the correct offset.
     */
    uint8_t *param_buf = NULL;
    if (total_ps > 0 && num_params > 0 && args != NULL) {
        param_buf = (uint8_t *)calloc(1, total_ps);
        for (int i = 0; i < num_params; i++) {
            int32_t off = g_registered_functions[func_idx].params[i].offset;
            int32_t sz = g_registered_functions[func_idx].params[i].size;
            if (off + sz <= total_ps) {
                memcpy(param_buf + off, args[i], sz);
            }
        }
    }

    if (g_verbose)
        fprintf(stderr, "[chidori-launch] func=0x%llx grid=(%u,%u,%u) block=(%u,%u,%u) "
                "params=%d total_size=%d\n",
                (unsigned long long)func_handle,
                gridDim.x, gridDim.y, gridDim.z,
                blockDim.x, blockDim.y, blockDim.z,
                num_params, total_ps);

    size_t payload_len = 44 + 1;
    if (total_ps > 0)
        payload_len += 4 + total_ps;

    void *payload = malloc(payload_len);
    char *p = (char *)payload;

    memcpy(p, &func_handle, 8); p += 8;
    uint32_t gx = gridDim.x, gy = gridDim.y, gz = gridDim.z;
    uint32_t bx = blockDim.x, by = blockDim.y, bz = blockDim.z;
    uint32_t shared = (uint32_t)sharedMem;
    uint64_t stream_handle = (uint64_t)(uintptr_t)stream;
    memcpy(p, &gx, 4); p += 4;
    memcpy(p, &gy, 4); p += 4;
    memcpy(p, &gz, 4); p += 4;
    memcpy(p, &bx, 4); p += 4;
    memcpy(p, &by, 4); p += 4;
    memcpy(p, &bz, 4); p += 4;
    memcpy(p, &shared, 4); p += 4;
    memcpy(p, &stream_handle, 8); p += 8;

    if (total_ps > 0 && param_buf) {
        uint8_t has_params = 1;
        memcpy(p, &has_params, 1); p += 1;
        uint32_t buf_size = (uint32_t)total_ps;
        memcpy(p, &buf_size, 4); p += 4;
        memcpy(p, param_buf, total_ps);
        free(param_buf);
    } else {
        uint8_t has_params = 0;
        memcpy(p, &has_params, 1);
    }

    cudaError_t r = (cudaError_t)chidori_rpc_async(CHIDORI_CUDA_LAUNCH_KERNEL, payload, payload_len);
    free(payload);
    g_last_error = r;
    return r;
}

/*
 * __cudaLaunchKernel — internal CUDA Runtime launch entry point.
 * Source-built PyTorch (CUDA 13.x) calls this instead of cudaLaunchKernel.
 * Same signature, just delegate.
 */
cudaError_t __cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                               void **args, size_t sharedMem, cudaStream_t stream) {
    return cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

/*
 * cudaLaunchKernelExC — CUDA 12.x extended launch (cluster dims, etc.)
 * Extract grid/block from the config struct and delegate to cudaLaunchKernel.
 *
 * cudaLaunchConfig_t layout (CUDA 12.x):
 *   dim3 gridDim;       // offset 0, 12 bytes
 *   dim3 blockDim;      // offset 12, 12 bytes
 *   size_t dynamicSmemBytes;  // offset 24, 8 bytes
 *   cudaStream_t stream;     // offset 32, 8 bytes
 *   ...attrs after offset 40
 */
typedef struct {
    dim3 gridDim;
    dim3 blockDim;
    size_t dynamicSmemBytes;
    void *stream;
    /* remaining fields ignored */
} chidori_launch_config_t;

cudaError_t cudaLaunchKernelExC(const chidori_launch_config_t *config,
                                 const void *func, void **args) {
    return cudaLaunchKernel(func, config->gridDim, config->blockDim,
                            args, config->dynamicSmemBytes, config->stream);
}

/*
 * __cudaPushCallConfiguration / __cudaPopCallConfiguration
 * Used by nvcc-compiled code for <<<>>> kernel launch syntax.
 */
static __thread dim3 s_push_grid, s_push_block;
static __thread size_t s_push_shared;
static __thread void *s_push_stream;

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem, void *stream) {
    s_push_grid = gridDim;
    s_push_block = blockDim;
    s_push_shared = sharedMem;
    s_push_stream = stream;
    if (g_verbose)
        fprintf(stderr, "[chidori-push] grid=(%u,%u,%u) block=(%u,%u,%u) shared=%zu stream=%p\n",
                gridDim.x, gridDim.y, gridDim.z,
                blockDim.x, blockDim.y, blockDim.z,
                sharedMem, stream);
    return 0; /* success */
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                        size_t *sharedMem, void **stream) {
    if (gridDim) *gridDim = s_push_grid;
    if (blockDim) *blockDim = s_push_block;
    if (sharedMem) *sharedMem = s_push_shared;
    if (stream) *stream = s_push_stream;
    return cudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void *func, int attr, int value) {
    (void)func; (void)attr; (void)value;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* Cache for cudaFuncGetAttributes — attributes are immutable per function */
#define FUNC_ATTR_CACHE_SIZE 256
static struct { const void *func; char attr[64]; int valid; } g_func_attr_cache[FUNC_ATTR_CACHE_SIZE];
static int g_func_attr_cache_count = 0;

cudaError_t cudaFuncGetAttributes(void *attr, const void *func) {
    if (!attr) { g_last_error = (cudaError_t)1; return (cudaError_t)1; }

    /* Check cache first */
    for (int i = 0; i < g_func_attr_cache_count; i++) {
        if (g_func_attr_cache[i].func == func && g_func_attr_cache[i].valid) {
            memcpy(attr, g_func_attr_cache[i].attr, 64);
            g_last_error = cudaSuccess;
            return cudaSuccess;
        }
    }

    memset(attr, 0, 64);

    /* Find and lazy-register the function on the server */
    uint64_t func_handle = 0;
    for (int i = 0; i < g_registered_function_count; i++) {
        if (g_registered_functions[i].hostFun == func) {
            func_handle = lazy_register_function(i);
            break;
        }
    }

    if (func_handle == 0) {
        /* Can't find function — return reasonable defaults so grid calc doesn't get 0 */
        int mtp = 1024;
        memcpy((char *)attr + 24, &mtp, 4);  /* maxThreadsPerBlock */
        g_last_error = cudaSuccess;
        return cudaSuccess;
    }

    /* RPC to server: send func_handle(8), receive cudaFuncAttributes blob */
    size_t out_len = 64;
    cudaError_t r = (cudaError_t)chidori_rpc(CHIDORI_CUDA_FUNC_GET_ATTRIBUTES,
                                              &func_handle, 8, attr, &out_len);
    /* Cache the result */
    if (r == cudaSuccess && g_func_attr_cache_count < FUNC_ATTR_CACHE_SIZE) {
        g_func_attr_cache[g_func_attr_cache_count].func = func;
        memcpy(g_func_attr_cache[g_func_attr_cache_count].attr, attr, 64);
        g_func_attr_cache[g_func_attr_cache_count].valid = 1;
        g_func_attr_cache_count++;
    }
    g_last_error = r;
    return r;
}

cudaError_t cudaFuncSetCacheConfig(const void *func, int cacheConfig) {
    (void)func; (void)cacheConfig;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaFuncSetSharedMemConfig(const void *func, int config) {
    (void)func; (void)config;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
    const void *func, int blockSize, size_t dynamicSMemSize) {
    (void)func; (void)blockSize; (void)dynamicSMemSize;
    if (numBlocks) *numBlocks = 16;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize,
    size_t dynamicSMemSize, unsigned int flags) {
    (void)flags;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

/* ---- Cache / Shared Memory Config ---- */

cudaError_t cudaDeviceSetCacheConfig(int cacheConfig) {
    (void)cacheConfig;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetCacheConfig(int *pCacheConfig) {
    if (pCacheConfig) *pCacheConfig = 0;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceSetSharedMemConfig(int config) {
    (void)config;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetSharedMemConfig(int *pConfig) {
    if (pConfig) *pConfig = 0;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* ---- Pointer Attributes ---- */

cudaError_t cudaPointerGetAttributes(void *attributes, const void *ptr) {
    (void)ptr;
    if (attributes) memset(attributes, 0, 64);
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* ---- Managed Memory ---- */

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    (void)flags;
    return cudaMalloc(devPtr, size);
}

/* ---- Memory Pool Operations (CUDA 11.2+) ---- */

cudaError_t cudaMemPoolTrimTo(void *memPool, size_t minBytesToKeep) {
    (void)memPool; (void)minBytesToKeep;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaMemPoolSetAttribute(void *memPool, int attr, void *value) {
    (void)memPool; (void)attr; (void)value;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaMemPoolGetAttribute(void *memPool, int attr, void *value) {
    (void)memPool; (void)attr;
    if (value) memset(value, 0, 8);
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaMemPoolSetAccess(void *memPool, const void *descList, size_t count) {
    (void)memPool; (void)descList; (void)count;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaMemPoolGetAccess(int *flags, void *memPool, void *location) {
    (void)memPool; (void)location;
    if (flags) *flags = 0;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* ---- CUDA Graphs (stubs) ---- */

cudaError_t cudaGraphCreate(void **pGraph, unsigned int flags) {
    (void)flags;
    static int fake_graph;
    *pGraph = (void *)&fake_graph;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphDestroy(void *graph) {
    (void)graph;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphExecDestroy(void *graphExec) {
    (void)graphExec;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphInstantiate(void **pGraphExec, void *graph,
                                  void *pErrorNode, char *pLogBuffer,
                                  size_t bufferSize) {
    (void)graph; (void)pErrorNode; (void)pLogBuffer; (void)bufferSize;
    static int fake_exec;
    *pGraphExec = (void *)&fake_exec;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGraphLaunch(void *graphExec, void *stream) {
    (void)graphExec; (void)stream;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamBeginCapture(void *stream, int mode) {
    (void)stream; (void)mode;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaStreamEndCapture(void *stream, void **pGraph) {
    (void)stream;
    static int fake_graph;
    if (pGraph) *pGraph = (void *)&fake_graph;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* ---- Texture / Surface (stubs PyTorch may reference) ---- */

cudaError_t cudaCreateTextureObject(unsigned long long *pTexObject,
                                     const void *pResDesc, const void *pTexDesc,
                                     const void *pResViewDesc) {
    (void)pResDesc; (void)pTexDesc; (void)pResViewDesc;
    if (pTexObject) *pTexObject = 0;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaDestroyTextureObject(unsigned long long texObject) {
    (void)texObject;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/* ---- Misc Runtime API ---- */

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    (void)flags;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceFlags(unsigned int *flags) {
    if (flags) *flags = 0;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/*
 * __cudaRegisterFatBinary / __cudaUnregisterFatBinary / __cudaRegisterFunction
 *
 * Called by CUDA Runtime code generated by nvcc. These send the fat binary
 * and function registration data to the remote GPU server.
 */
void **__cudaRegisterFatBinary(void *fatCubin) {
    /*
     * PASSTHROUGH MODE: When CHIDORI_FATBIN_PASSTHROUGH=1, skip all our fat binary
     * tracking (size parsing, index storage, fake handles) and just forward to
     * real __cudaRegisterFatBinary. Tests whether our tracking machinery breaks cuBLAS.
     */
    if (is_fatbin_passthrough()) {
        init_real_cuda();
        if (g_real.__cudaRegisterFatBinary) {
            void **handle;
            g_in_real_cuda++;
            handle = g_real.__cudaRegisterFatBinary(fatCubin);
            g_in_real_cuda--;
            LOG_DEBUG("__cudaRegisterFatBinary PASSTHROUGH → real handle=%p", (void *)handle);
            return handle;
        }
        LOG_DEBUG("__cudaRegisterFatBinary PASSTHROUGH: no real function, returning NULL");
        return NULL;
    }
    /*
     * LAZY REGISTRATION: Store the fat binary data locally.
     * Only send to server when a kernel from this module is actually launched.
     * This avoids 50+ RPCs during PyTorch init (each ~166ms over internet).
     */
    struct __fatBinC_Wrapper {
        int magic;
        int version;
        const void *data;
        void *filename;
    };

    struct __fatBinC_Wrapper *wrapper = (struct __fatBinC_Wrapper *)fatCubin;
    const unsigned char *data = (const unsigned char *)wrapper->data;

    /* Determine fat binary size from the header.
     *
     * NVIDIA fat binary formats:
     *   Magic 0xba55ed50 (__cudaFatBinaryHeader):
     *     offset 0: magic(4) version(4) headerSize(8) fatSize(8)
     *     fatSize at offset 16 (8 bytes) is the total size
     *   Magic 0x466243b1 (old FATBIN):
     *     offset 0: magic(4) version(4) size(8)
     *     size at offset 8 is the total
     *   ELF (0x7f454c46):
     *     Read e_shoff + e_shnum * e_shentsize for actual size
     */
    uint32_t magic = 0;
    memcpy(&magic, data, 4);

    uint64_t fat_size = 0;
    if (magic == 0xba55ed50) {
        /* __cudaFatBinaryHeader: magic(4) version(2) headerSize(2) fatSize(8)
         * The fatSize includes the header itself. */
        uint16_t version = 0, headerSize = 0;
        memcpy(&version, data + 4, 2);
        memcpy(&headerSize, data + 6, 2);
        memcpy(&fat_size, data + 8, 8);

        /* Sanity: if fatSize is 0 or absurd, use headerSize as minimum */
        if (fat_size == 0 || fat_size > 256*1024*1024) {
            fat_size = headerSize > 0 ? headerSize : 4096;
            fprintf(stderr, "[chidori] fat_binary: bad fatSize=%llu, using %llu (v=%u hdr=%u)\n",
                    (unsigned long long)fat_size, (unsigned long long)fat_size, version, headerSize);
        }
    } else if (magic == 0x466243b1) {
        /* Old FATBIN — size at offset 8 */
        memcpy(&fat_size, data + 8, 8);
    } else if (magic == 0x464c457f || (data[0] == 0x7f && data[1] == 'E' && data[2] == 'L' && data[3] == 'F')) {
        /* ELF — parse header for actual size */
        uint64_t e_shoff = 0;
        uint16_t e_shnum = 0, e_shentsize = 0;
        memcpy(&e_shoff, data + 40, 8);    /* e_shoff at offset 40 in ELF64 */
        memcpy(&e_shentsize, data + 58, 2); /* e_shentsize at offset 58 */
        memcpy(&e_shnum, data + 60, 2);     /* e_shnum at offset 60 */
        fat_size = e_shoff + (uint64_t)e_shnum * e_shentsize;
    } else {
        /* Unknown format — try offset 8 as generic size field */
        memcpy(&fat_size, data + 8, 8);
    }

    if (fat_size == 0 || fat_size > 256*1024*1024) {
        /* Last resort fallback */
        fat_size = 256 * 1024;
        fprintf(stderr, "[chidori] fat_binary: unknown format magic=0x%08x, using fallback size=%llu\n",
                magic, (unsigned long long)fat_size);
    } else if (g_verbose) {
        fprintf(stderr, "[chidori] fat_binary: magic=0x%08x size=%llu\n",
                magic, (unsigned long long)fat_size);
    }

    /* Store pointer to fat binary data for lazy registration.
     * The data lives in the .nv_fatbin ELF section of the loaded .so — it
     * remains valid for the lifetime of the process. */
    static int fat_idx_storage[MAX_FAT_BINARIES];
    if (g_fat_binary_count < MAX_FAT_BINARIES) {
        int idx = g_fat_binary_count++;
        g_fat_binaries[idx].data = data;
        g_fat_binaries[idx].size = fat_size;
        g_fat_binaries[idx].server_handle = 0;
        g_fat_binaries[idx].real_handle = NULL;

        /* For partial offload: register with real CUDA for local execution.
         * In deferred init mode, skip init_real_cuda() here (it will be called
         * from cuInit) and store this fatbin for later replay. */
        if (is_deferred_init() && !g_real_cuda_initialized) {
            /* Deferred: store for replay after cuInit */
            if (g_pending_fatbin_count < MAX_PENDING_FATBINS) {
                g_pending_fatbins[g_pending_fatbin_count].fatCubin = fatCubin;
                g_pending_fatbins[g_pending_fatbin_count].idx = idx;
                g_pending_fatbin_count++;
                LOG_DEBUG("__cudaRegisterFatBinary[%d] DEFERRED (size=%llu, pending=%d)",
                          idx, (unsigned long long)fat_size, g_pending_fatbin_count);
            }
        } else {
            init_real_cuda();
            if (g_partial_offload && g_real.__cudaRegisterFatBinary) {
                LOG_DEBUG("__cudaRegisterFatBinary[%d] → LOCAL + REMOTE (size=%llu)", idx, (unsigned long long)fat_size);
                g_in_real_cuda++;
                g_fat_binaries[idx].real_handle = g_real.__cudaRegisterFatBinary(fatCubin);
                g_in_real_cuda--;
            }
        }

        fat_idx_storage[idx] = idx;
        return (void **)&fat_idx_storage[idx];
    }

    static int fallback_idx = 0;
    return (void **)&fallback_idx;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    if (is_fatbin_passthrough()) {
        if (g_real.__cudaRegisterFatBinaryEnd) {
            g_in_real_cuda++;
            g_real.__cudaRegisterFatBinaryEnd(fatCubinHandle);
            g_in_real_cuda--;
        }
        return;
    }
    /* Deferred init: store for later replay if real CUDA not yet initialized */
    if (is_deferred_init() && !g_deferred_replay_done) {
        int idx = fatCubinHandle ? *(int *)fatCubinHandle : -1;
        if (idx >= 0 && g_pending_fatbin_end_count < MAX_PENDING_FATBINS) {
            g_pending_fatbin_ends[g_pending_fatbin_end_count++] = idx;
        }
        return;
    }
    if (g_partial_offload && g_real.__cudaRegisterFatBinaryEnd) {
        int idx = fatCubinHandle ? *(int *)fatCubinHandle : -1;
        if (idx >= 0 && idx < g_fat_binary_count && g_fat_binaries[idx].real_handle) {
            g_in_real_cuda++;
            g_real.__cudaRegisterFatBinaryEnd(g_fat_binaries[idx].real_handle);
            g_in_real_cuda--;
        }
    }
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    if (is_fatbin_passthrough()) {
        if (g_real.__cudaUnregisterFatBinary) {
            g_in_real_cuda++;
            g_real.__cudaUnregisterFatBinary(fatCubinHandle);
            g_in_real_cuda--;
        }
        return;
    }
    if (g_partial_offload && g_real.__cudaUnregisterFatBinary) {
        int idx = fatCubinHandle ? *(int *)fatCubinHandle : -1;
        if (idx >= 0 && idx < g_fat_binary_count && g_fat_binaries[idx].real_handle) {
            g_in_real_cuda++;
            g_real.__cudaUnregisterFatBinary(g_fat_binaries[idx].real_handle);
            g_in_real_cuda--;
        }
    }
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, void *tid, void *bid,
                            void *bDim, void *gDim, int *wSize) {
    /* PASSTHROUGH: forward directly to real __cudaRegisterFunction, skip our tracking */
    if (is_fatbin_passthrough()) {
        if (g_real.__cudaRegisterFunction) {
            g_in_real_cuda++;
            g_real.__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                                          thread_limit, tid, bid, bDim, gDim, wSize);
            g_in_real_cuda--;
        }
        return;
    }
    (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;

    /*
     * LAZY REGISTRATION: Store function info locally.
     * Only send to server when cudaLaunchKernel is called with this function.
     * This avoids 1400+ RPCs during PyTorch init.
     */
    int fat_idx = *(int *)fatCubinHandle;
    const char *name = deviceName ? deviceName : deviceFun;

    if (g_registered_function_count < MAX_REGISTERED_FUNCTIONS) {
        int idx = g_registered_function_count++;
        g_registered_functions[idx].hostFun = (const void *)hostFun;
        g_registered_functions[idx].fat_binary_idx = fat_idx;
        g_registered_functions[idx].deviceName = strdup(name);
        g_registered_functions[idx].server_handle = 0; /* not yet registered */
    }

    /* For partial offload: also register with real CUDA for local kernel execution.
     * In deferred init mode, store for replay if real CUDA not yet initialized. */
    if (is_deferred_init() && !g_deferred_replay_done) {
        /* Deferred: store for replay after cuInit */
        if (g_pending_func_count < MAX_PENDING_FUNCS) {
            g_pending_funcs[g_pending_func_count].fat_idx = fat_idx;
            g_pending_funcs[g_pending_func_count].hostFun = hostFun;
            g_pending_funcs[g_pending_func_count].deviceFun = deviceFun;
            g_pending_funcs[g_pending_func_count].deviceName = deviceName;
            g_pending_funcs[g_pending_func_count].thread_limit = thread_limit;
            g_pending_funcs[g_pending_func_count].tid = tid;
            g_pending_funcs[g_pending_func_count].bid = bid;
            g_pending_funcs[g_pending_func_count].bDim = bDim;
            g_pending_funcs[g_pending_func_count].gDim = gDim;
            g_pending_funcs[g_pending_func_count].wSize = wSize;
            g_pending_func_count++;
        }
    } else if (g_partial_offload && g_real.__cudaRegisterFunction) {
        void **real_handle = (fat_idx >= 0 && fat_idx < g_fat_binary_count)
                             ? g_fat_binaries[fat_idx].real_handle : NULL;
        if (real_handle) {
            g_in_real_cuda++;
            g_real.__cudaRegisterFunction(real_handle, hostFun, deviceFun, deviceName,
                                          thread_limit, tid, bid, bDim, gDim, wSize);
            g_in_real_cuda--;
        }
    }
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName,
                       int ext, size_t size, int constant, int global) {
    (void)fatCubinHandle; (void)hostVar; (void)deviceAddress; (void)deviceName;
    (void)ext; (void)size; (void)constant; (void)global;
}

void __cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress,
                              char *deviceAddress, const char *deviceName,
                              int ext, size_t size, int constant, int global) {
    (void)fatCubinHandle; (void)hostVarPtrAddress; (void)deviceAddress;
    (void)deviceName; (void)ext; (void)size; (void)constant; (void)global;
}

void __cudaRegisterTexture(void **fatCubinHandle, const void *hostVar,
                           const void **deviceAddress, const char *deviceName,
                           int dim, int norm, int ext) {
    (void)fatCubinHandle; (void)hostVar; (void)deviceAddress; (void)deviceName;
    (void)dim; (void)norm; (void)ext;
}

void __cudaRegisterSurface(void **fatCubinHandle, const void *hostVar,
                           const void **deviceAddress, const char *deviceName,
                           int dim, int ext) {
    (void)fatCubinHandle; (void)hostVar; (void)deviceAddress; (void)deviceName;
    (void)dim; (void)ext;
}

char __cudaInitModule(void **fatCubinHandle) {
    (void)fatCubinHandle;
    return 0;
}

/*
 * ============================================================================
 * PyTorch Compatibility Stubs — CUDA Driver API (additional)
 * ============================================================================
 */

/* ---- Error Handling (Driver API) ---- */

CUresult cuGetErrorString(CUresult error, const char **pStr) {
    if (pStr) *pStr = cudaGetErrorString((cudaError_t)error);
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char **pStr) {
    if (pStr) *pStr = cudaGetErrorName((cudaError_t)error);
    return CUDA_SUCCESS;
}

/* ---- Module (additional) ---- */

CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    return cuModuleLoadData(module, fatCubin);
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, void *options, void **optionValues) {
    (void)numOptions; (void)options; (void)optionValues;
    return cuModuleLoadData(module, image);
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes,
                               CUmodule hmod, const char *name) {
    (void)hmod; (void)name;
    if (dptr) *dptr = 0;
    if (bytes) *bytes = 0;
    return CUDA_SUCCESS;
}

/* ---- Pointer Attributes (Driver API) ---- */

CUresult cuPointerGetAttribute(void *data, int attribute, CUdeviceptr ptr) {
    if (!data) return CUDA_ERROR_INVALID_VALUE;
    memset(data, 0, 8);

    int dev = alloc_track_device((uintptr_t)ptr);
    if (dev < 0) dev = g_current_device; /* fallback: assume current device */

    switch (attribute) {
    case 1: /* CU_POINTER_ATTRIBUTE_CONTEXT */
        /* Return local context for local device, NULL for remote */
        if (is_local_device(dev)) {
            memcpy(data, &g_local_ctx, sizeof(void*));
        }
        break;
    case 2: /* CU_POINTER_ATTRIBUTE_MEMORY_TYPE */
        { int type = 2; /* CU_MEMORYTYPE_DEVICE */ memcpy(data, &type, sizeof(int)); }
        break;
    case 3: /* CU_POINTER_ATTRIBUTE_DEVICE_POINTER */
        { CUdeviceptr dp = ptr; memcpy(data, &dp, sizeof(CUdeviceptr)); }
        break;
    case 8: /* CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL */
        memcpy(data, &dev, sizeof(int));
        break;
    default:
        break;
    }
    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttributes(unsigned int numAttributes, int *attributes,
                                 void **data, CUdeviceptr ptr) {
    (void)ptr;
    for (unsigned int i = 0; i < numAttributes; i++) {
        if (data[i]) memset(data[i], 0, 8);
    }
    return CUDA_SUCCESS;
}

/* ---- Virtual Memory (CUDA 10.2+) ---- */

CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                              CUdeviceptr addr, unsigned long long flags) {
    (void)alignment; (void)addr; (void)flags; (void)size;
    if (ptr) *ptr = 0;
    return CUDA_SUCCESS;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    (void)ptr; (void)size;
    return CUDA_SUCCESS;
}

CUresult cuMemCreate(void **handle, size_t size, void *prop, unsigned long long flags) {
    (void)size; (void)prop; (void)flags;
    static int fake_alloc;
    if (handle) *handle = (void *)&fake_alloc;
    return CUDA_SUCCESS;
}

CUresult cuMemRelease(void *handle) {
    (void)handle;
    return CUDA_SUCCESS;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                   void *handle, unsigned long long flags) {
    (void)ptr; (void)size; (void)offset; (void)handle; (void)flags;
    return CUDA_SUCCESS;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    (void)ptr; (void)size;
    return CUDA_SUCCESS;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const void *desc, size_t count) {
    (void)ptr; (void)size; (void)desc; (void)count;
    return CUDA_SUCCESS;
}

CUresult cuMemGetAccess(unsigned long long *flags, const void *location, CUdeviceptr ptr) {
    (void)location; (void)ptr;
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuMemGetAllocationGranularity(size_t *granularity, const void *prop, int option) {
    (void)prop; (void)option;
    if (granularity) *granularity = 65536;
    return CUDA_SUCCESS;
}

CUresult cuMemGetAllocationPropertiesFromHandle(void *prop, void *handle) {
    (void)handle;
    if (prop) memset(prop, 0, 64);
    return CUDA_SUCCESS;
}

/* ---- Stream Capture / CUDA Graphs (Driver API) ---- */

CUresult cuStreamBeginCapture_v2(CUstream hStream, int mode) {
    (void)hStream; (void)mode;
    return CUDA_SUCCESS;
}

CUresult cuStreamEndCapture(CUstream hStream, void **phGraph) {
    (void)hStream;
    static int fake_graph;
    if (phGraph) *phGraph = (void *)&fake_graph;
    return CUDA_SUCCESS;
}

CUresult cuStreamIsCapturing(CUstream hStream, int *captureStatus) {
    (void)hStream;
    if (captureStatus) *captureStatus = 0;
    return CUDA_SUCCESS;
}

CUresult cuGraphLaunch(void *hGraphExec, CUstream hStream) {
    (void)hGraphExec; (void)hStream;
    return CUDA_SUCCESS;
}

CUresult cuGraphInstantiate_v2(void **phGraphExec, void *hGraph,
                                void **phErrorNode, char *logBuffer,
                                size_t bufferSize) {
    (void)hGraph; (void)phErrorNode; (void)logBuffer; (void)bufferSize;
    static int fake_exec;
    if (phGraphExec) *phGraphExec = (void *)&fake_exec;
    return CUDA_SUCCESS;
}

CUresult cuGraphExecDestroy(void *hGraphExec) {
    (void)hGraphExec;
    return CUDA_SUCCESS;
}

CUresult cuGraphDestroy(void *hGraph) {
    (void)hGraph;
    return CUDA_SUCCESS;
}

/* ---- Stream (Driver API additional) ---- */

CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    (void)flags; (void)priority;
    return cuStreamCreate(phStream, 0);
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    (void)hStream;
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
    (void)hStream;
    if (priority) *priority = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags) {
    (void)hStream; (void)hEvent; (void)flags;
    return CUDA_SUCCESS;
}

CUresult cuStreamQuery(CUstream hStream) {
    (void)hStream;
    return CUDA_SUCCESS;
}

CUresult cuStreamAddCallback(CUstream hStream, void *callback, void *userData, unsigned int flags) {
    (void)hStream; (void)callback; (void)userData; (void)flags;
    return CUDA_SUCCESS;
}

/* ---- Event (Driver API additional) ---- */

CUresult cuEventQuery(CUevent hEvent) {
    (void)hEvent;
    return CUDA_SUCCESS;
}

/* ---- Context (additional) ---- */

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    return cuCtxSetCurrent(ctx);
}

CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
    return cuCtxGetCurrent(pctx);
}

CUresult cuCtxGetDevice(CUdevice *device) {
    size_t len = sizeof(int32_t);
    return chidori_rpc(CHIDORI_CU_CTX_GET_DEVICE, NULL, 0, device, &len);
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    (void)dev;
    if (flags) *flags = 0;
    if (active) *active = 1;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    (void)dev; (void)flags;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    (void)dev;
    return CUDA_SUCCESS;
}

/*
 * cuGetExportTable -- Internal CUDA driver function used by cudart for
 * initialization handshake. Without this, the real cudart reports
 * cudaErrorCallRequiresNewerDriver (36) because it can't negotiate
 * internal function tables with the driver.
 *
 * Forward directly to the real driver (dlopen'd handle).
 */
CUresult cuGetExportTable(const void **ppExportTable, const void *pExportTableId) {
    if (g_real.cuGetExportTable)
        return g_real.cuGetExportTable(ppExportTable, pExportTableId);
    return CUDA_ERROR_NOT_INITIALIZED;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    (void)ctx;
    if (version) *version = 3020;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = -1;
    return CUDA_SUCCESS;
}

/* ---- Device (Driver API additional) ---- */

CUresult cuDeviceGetUuid(void *uuid, CUdevice dev) {
    (void)dev;
    if (uuid) memset(uuid, 0, 16);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid_v2(void *uuid, CUdevice dev) {
    return cuDeviceGetUuid(uuid, dev);
}

CUresult cuDeviceGetDefaultMemPool(void **pool, CUdevice dev) {
    (void)dev;
    static int fake_pool;
    if (pool) *pool = (void *)&fake_pool;
    return CUDA_SUCCESS;
}

CUresult cuDeviceSetMemPool(CUdevice dev, void *pool) {
    (void)dev; (void)pool;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetMemPool(void **pool, CUdevice dev) {
    return cuDeviceGetDefaultMemPool(pool, dev);
}

CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
    (void)dev; (void)peerDev;
    if (canAccessPeer) *canAccessPeer = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetP2PAttribute(int *value, int attrib, CUdevice srcDevice, CUdevice dstDevice) {
    (void)attrib; (void)srcDevice; (void)dstDevice;
    if (value) *value = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    (void)pciBusId;
    if (dev) *dev = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    (void)dev;
    if (pciBusId && len > 0) snprintf(pciBusId, (size_t)len, "0000:00:00.0");
    return CUDA_SUCCESS;
}

/* ---- Occupancy (Driver API) ---- */

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
    CUfunction func, int blockSize, size_t dynamicSMemSize) {
    (void)func; (void)blockSize; (void)dynamicSMemSize;
    if (numBlocks) *numBlocks = 16;
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize,
    size_t dynamicSMemSize, unsigned int flags) {
    (void)flags;
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
    CUfunction func, void *blockSizeToDynSMemSize,
    size_t dynamicSMemSize, int blockSizeLimit) {
    (void)func; (void)blockSizeToDynSMemSize; (void)dynamicSMemSize; (void)blockSizeLimit;
    if (minGridSize) *minGridSize = 1;
    if (blockSize) *blockSize = 256;
    return CUDA_SUCCESS;
}

/* ---- Func (Driver API additional) ---- */

CUresult cuFuncSetCacheConfig(CUfunction hfunc, int config) {
    (void)hfunc; (void)config;
    return CUDA_SUCCESS;
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, int config) {
    (void)hfunc; (void)config;
    return CUDA_SUCCESS;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, int attrib, int value) {
    (void)hfunc; (void)attrib; (void)value;
    return CUDA_SUCCESS;
}

/* ---- Peer Context (Driver API) ---- */

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags) {
    (void)peerContext; (void)flags;
    return CUDA_SUCCESS;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    (void)peerContext;
    return CUDA_SUCCESS;
}

/* ---- Memory (Driver API additional) ---- */

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    (void)flags;
    return cuMemAlloc_v2(dptr, bytesize);
}

CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int flags) {
    (void)flags;
    *pp = malloc(bytesize);
    return *pp ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemFreeHost(void *p) {
    free(p);
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int flags) {
    (void)flags;
    if (pdptr) *pdptr = (CUdeviceptr)(uintptr_t)p;
    return CUDA_SUCCESS;
}

CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int flags) {
    (void)p; (void)bytesize; (void)flags;
    return CUDA_SUCCESS;
}

CUresult cuMemHostUnregister(void *p) {
    (void)p;
    return CUDA_SUCCESS;
}

CUresult cuMemcpy_v2(CUdeviceptr dst, CUdeviceptr src, size_t byteCount) {
    struct { uint64_t dst; uint64_t src; uint64_t size; } req;
    req.dst = (uint64_t)dst;
    req.src = (uint64_t)src;
    req.size = (uint64_t)byteCount;
    return chidori_rpc(CHIDORI_CU_MEMCPY, &req, sizeof(req), NULL, NULL);
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream) {
    (void)hStream;
    return cuMemcpy_v2(dst, src, byteCount);
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                       CUdeviceptr srcDevice, CUcontext srcContext,
                       size_t byteCount) {
    (void)dstContext; (void)srcContext;
    /* Cross-context: stage through CPU */
    void *staging = malloc(byteCount);
    if (!staging) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = cuMemcpyDtoH_v2(staging, srcDevice, byteCount);
    if (r == CUDA_SUCCESS)
        r = cuMemcpyHtoD_v2(dstDevice, staging, byteCount);
    free(staging);
    return r;
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                            CUdeviceptr srcDevice, CUcontext srcContext,
                            size_t byteCount, CUstream hStream) {
    (void)hStream;
    return cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, byteCount);
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    if (g_partial_offload && is_local_device(g_current_device) && g_real_cuda_handle) {
        typedef CUresult (*fn_t)(CUdeviceptr, unsigned char, size_t);
        static fn_t real_fn = NULL;
        if (!real_fn) real_fn = (fn_t)__real_dlsym(g_real_cuda_handle, "cuMemsetD8_v2");
        if (real_fn) { g_in_real_cuda++; CUresult r = real_fn(dstDevice, uc, N); g_in_real_cuda--; return r; }
    }
    struct __attribute__((packed)) {
        uint64_t ptr; int32_t value; uint64_t count;
    } req;
    req.ptr = (uint64_t)dstDevice; req.value = (int32_t)uc; req.count = N;
    return chidori_rpc_async(CHIDORI_CU_MEMSET_D8, &req, sizeof(req));
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    (void)hStream;
    return cuMemsetD8_v2(dstDevice, uc, N);
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    if (g_partial_offload && is_local_device(g_current_device) && g_real_cuda_handle) {
        typedef CUresult (*fn_t)(CUdeviceptr, unsigned int, size_t);
        static fn_t real_fn = NULL;
        if (!real_fn) real_fn = (fn_t)__real_dlsym(g_real_cuda_handle, "cuMemsetD32_v2");
        if (real_fn) { g_in_real_cuda++; CUresult r = real_fn(dstDevice, ui, N); g_in_real_cuda--; return r; }
    }
    struct __attribute__((packed)) {
        uint64_t ptr; int32_t value; uint64_t count;
    } req;
    req.ptr = (uint64_t)dstDevice; req.value = (int32_t)ui; req.count = N;
    return chidori_rpc_async(CHIDORI_CU_MEMSET_D32, &req, sizeof(req));
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
    (void)hStream;
    return cuMemsetD32_v2(dstDevice, ui, N);
}

/* ---- IPC (Driver API) ---- */

CUresult cuIpcGetMemHandle(void *pHandle, CUdeviceptr dptr) {
    (void)dptr;
    if (pHandle) memset(pHandle, 0, 64);
    return CUDA_SUCCESS;
}

CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, void *handle, unsigned int flags) {
    (void)handle; (void)flags;
    if (pdptr) *pdptr = 0;
    return CUDA_SUCCESS;
}

CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
    (void)dptr;
    return CUDA_SUCCESS;
}

CUresult cuIpcGetEventHandle(void *pHandle, CUevent event) {
    (void)event;
    if (pHandle) memset(pHandle, 0, 64);
    return CUDA_SUCCESS;
}

CUresult cuIpcOpenEventHandle(CUevent *phEvent, void *handle) {
    (void)handle;
    if (phEvent) *phEvent = NULL;
    return CUDA_SUCCESS;
}

/*
 * ============================================================================
 * cuBLAS Compatibility Stubs (additional)
 * ============================================================================
 */

const char *cublasGetStatusString(cublasStatus_t status) {
    switch (status) {
    case 0:  return "CUBLAS_STATUS_SUCCESS";
    case 1:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case 3:  return "CUBLAS_STATUS_ALLOC_FAILED";
    case 7:  return "CUBLAS_STATUS_INVALID_VALUE";
    case 8:  return "CUBLAS_STATUS_ARCH_MISMATCH";
    case 11: return "CUBLAS_STATUS_MAPPING_ERROR";
    case 13: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case 14: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case 15: return "CUBLAS_STATUS_NOT_SUPPORTED";
    default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

cublasStatus_t cublasGetProperty(int type, int *value) {
    /* ALWAYS forward — cuBLAS may call during load-time version checks */
    void *h = ensure_real_cublas();
    if (h) {
        typedef cublasStatus_t (*fn_t)(int, int *);
        fn_t real_fn = (fn_t)__real_dlsym(h, "cublasGetProperty");
        if (real_fn) return real_fn(type, value);
    }
    (void)type;
    if (value) *value = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes) {
    (void)handle; (void)workspace; (void)workspaceSizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, int mode) {
    (void)handle; (void)mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, int *mode) {
    (void)handle;
    if (mode) *mode = 0; /* CUBLAS_POINTER_MODE_HOST */
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, int mode) {
    (void)handle; (void)mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, int *mode) {
    (void)handle;
    if (mode) *mode = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut,
                                      int logToStdErr, const char *logFileName) {
    (void)logIsOn; (void)logToStdOut; (void)logToStdErr; (void)logFileName;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void * const *Aarray, int Atype, int lda,
    const void * const *Barray, int Btype, int ldb,
    const void *beta,
    void * const *Carray, int Ctype, int ldc,
    int batchCount, int computeType, int algo) {

    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void * const *, int, int, const void * const *, int, int, const void *, void * const *, int, int, int, int, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublas(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasGemmBatchedEx"); }
        cublasHandle_t local_h = ensure_local_cublas_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo);
            g_in_real_cuda--;
            return r;
        }
    }

    for (int i = 0; i < batchCount; i++) {
        cublasStatus_t r = cublasGemmEx(handle, transa, transb, m, n, k,
                                         alpha,
                                         Aarray[i], Atype, lda,
                                         Barray[i], Btype, ldb,
                                         beta,
                                         Carray[i], Ctype, ldc,
                                         computeType, algo);
        if (r != CUBLAS_STATUS_SUCCESS) return r;
    }
    return CUBLAS_STATUS_SUCCESS;
}

/*
 * cuBLASLt stubs — PyTorch links against cuBLASLt for optimized GEMM scheduling.
 */
typedef void *cublasLtHandle_t;

/* Lazy-load real cuBLASLt for local device routing */
static void *g_real_cublaslt_handle = NULL;
static int g_cublaslt_load_attempted = 0;
static inline void *ensure_real_cublaslt(void) {
    if (g_real_cublaslt_handle) return g_real_cublaslt_handle;
    if (g_cublaslt_load_attempted) return NULL;
    g_cublaslt_load_attempted = 1;
    /* Same as cuBLAS: use PyTorch's already-loaded version to avoid version mismatch */
    g_real_cublaslt_handle = dlopen("libcublasLt.so.12", RTLD_LAZY | RTLD_NOLOAD);
    if (!g_real_cublaslt_handle)
        g_real_cublaslt_handle = dlopen("libcublasLt.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!g_real_cublaslt_handle) {
        static const char *lt_paths[] = {
            "/opt/cuda/lib64/libcublasLt.so",
            "/usr/local/cuda/lib64/libcublasLt.so",
            "/usr/lib/x86_64-linux-gnu/libcublasLt.so",
            "/usr/lib64/libcublasLt.so",
            NULL
        };
        for (int i = 0; lt_paths[i] && !g_real_cublaslt_handle; i++) {
            g_real_cublaslt_handle = dlopen(lt_paths[i], RTLD_LAZY);
        }
    }
    if (g_real_cublaslt_handle)
        LOG_DEBUG("Loaded real cuBLASLt: %p", g_real_cublaslt_handle);
    else
        LOG_DEBUG("WARNING: Could not load real cuBLASLt — local device cublasLt calls will use stubs");
    return g_real_cublaslt_handle;
}

/* Local cuBLASLt handle for partial offload — lazily created */
static cublasLtHandle_t g_local_cublaslt_handle = NULL;

static cublasLtHandle_t ensure_local_cublaslt_handle(void) {
    if (g_local_cublaslt_handle) return g_local_cublaslt_handle;
    void *h = ensure_real_cublaslt();
    if (!h) return NULL;
    typedef cublasStatus_t (*create_fn)(cublasLtHandle_t *);
    create_fn real_create = (create_fn)__real_dlsym(h, "cublasLtCreate");
    if (!real_create) return NULL;
    ensure_real_cuda_init();
    /* Use real cudart's cudaSetDevice for full device initialization */
    g_in_real_cuda++;
    if (g_real.cudaSetDevice) {
        g_real.cudaSetDevice(0);
    }
    cublasStatus_t r = real_create(&g_local_cublaslt_handle);
    g_in_real_cuda--;
    if (r != 0) {
        fprintf(stderr, "[chidori] ensure_local_cublaslt_handle: cublasLtCreate failed: %d\n", r);
        g_local_cublaslt_handle = NULL;
        return NULL;
    }
    LOG_DEBUG("Created local cuBLASLt handle: %p", (void*)g_local_cublaslt_handle);
    return g_local_cublaslt_handle;
}

static int g_lt_handle_storage;
cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(cublasLtHandle_t *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtCreate");
            if (real_fn) return real_fn(lightHandle);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        cublasLtHandle_t local_h = ensure_local_cublaslt_handle();
        if (local_h) {
            if (lightHandle) *lightHandle = local_h;
            return CUBLAS_STATUS_SUCCESS;
        }
    }
    if (lightHandle) *lightHandle = (cublasLtHandle_t)&g_lt_handle_storage;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(cublasLtHandle_t);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtDestroy");
            if (real_fn) return real_fn(lightHandle);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        /* Don't actually destroy — we reuse the local handle */
        return CUBLAS_STATUS_SUCCESS;
    }
    (void)lightHandle;
    return CUBLAS_STATUS_SUCCESS;
}

/*
 * ============================================================================
 * cublasLt — Lightweight descriptors that track matrix params so we can
 *            translate cublasLtMatmul into cublasGemmEx RPCs.
 * ============================================================================
 */

/* cublasLt attribute IDs (CUDA 12.x actual enum values) */
#define CUBLASLT_MATMUL_DESC_TRANSA      3
#define CUBLASLT_MATMUL_DESC_TRANSB      4
#define CUBLASLT_MATMUL_DESC_EPILOGUE    7
#define CUBLASLT_MATMUL_DESC_BIAS_POINTER 8
#define CUBLASLT_MATRIX_LAYOUT_TYPE  0
#define CUBLASLT_MATRIX_LAYOUT_ROWS  2
#define CUBLASLT_MATRIX_LAYOUT_COLS  3
#define CUBLASLT_MATRIX_LAYOUT_LD    4
#define CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT 6
#define CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET 7

typedef struct {
    int computeType;
    int scaleType;
    int transa;  /* 0=N, 1=T */
    int transb;
    int epilogue; /* CUBLASLT_EPILOGUE_* */
    uint64_t bias_pointer;
} lt_matmul_desc_t;

typedef struct {
    int dtype;       /* CUDA_R_32F=0, CUDA_R_16F=2, CUDA_R_16BF=14 */
    uint64_t rows;
    uint64_t cols;
    int64_t ld;
    int64_t batch_count;
    int64_t strided_batch_offset;
} lt_matrix_layout_t;

#define LT_DESC_POOL 1024
static lt_matmul_desc_t g_lt_descs[LT_DESC_POOL];
static int g_lt_desc_count = 0;
static lt_matrix_layout_t g_lt_layouts[LT_DESC_POOL * 4];
static int g_lt_layout_count = 0;

cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle,
    void *computeDesc, const void *alpha,
    const void *A, void *Adesc, const void *B, void *Bdesc,
    const void *beta, const void *C, void *Cdesc,
    void *D, void *Ddesc, const void *algo,
    void *workspace, size_t workspaceSizeInBytes, void *stream) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(cublasLtHandle_t, void *, const void *, const void *, void *, const void *, void *, const void *, const void *, void *, void *, void *, const void *, void *, size_t, void *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmul");
            if (real_fn) return real_fn(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasLtHandle_t, void *, const void *, const void *, void *, const void *, void *, const void *, const void *, void *, void *, void *, const void *, void *, size_t, void *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmul"); }
        cublasLtHandle_t local_h = ensure_local_cublaslt_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
            g_in_real_cuda--;
            return r;
        }
    }
    (void)lightHandle; (void)workspace; (void)workspaceSizeInBytes; (void)C;

    lt_matmul_desc_t *desc = (lt_matmul_desc_t *)computeDesc;
    lt_matrix_layout_t *layA = (lt_matrix_layout_t *)Adesc;
    lt_matrix_layout_t *layB = (lt_matrix_layout_t *)Bdesc;
    lt_matrix_layout_t *layC = (lt_matrix_layout_t *)Cdesc;
    lt_matrix_layout_t *layD = (lt_matrix_layout_t *)Ddesc;

    if (!layA || !layB || !layD) {
        LOG_DEBUG("cublasLtMatmul: NULL layout descriptor");
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    /* Build payload: 188 (header+descriptors) + 16 (alpha+beta) + 40 (5 ptrs) + 64 (algo) = 308 */
    uint8_t payload[308];
    char *p = (char *)payload;

    /* 28 bytes: transa, transb, computeType, scaleType, epilogue, bias_ptr */
    int32_t transa_v = desc ? desc->transa : 0;
    int32_t transb_v = desc ? desc->transb : 0;
    int32_t computeType_v = desc ? desc->computeType : 0;
    int32_t scaleType_v = desc ? desc->scaleType : 0;
    int32_t epilogue_v = desc ? desc->epilogue : 0;
    uint64_t bias_ptr_v = desc ? desc->bias_pointer : 0;
    memcpy(p, &transa_v, 4); p += 4;
    memcpy(p, &transb_v, 4); p += 4;
    memcpy(p, &computeType_v, 4); p += 4;
    memcpy(p, &scaleType_v, 4); p += 4;
    memcpy(p, &epilogue_v, 4); p += 4;
    memcpy(p, &bias_ptr_v, 8); p += 8;

    /* 160 bytes: 4 matrix descriptors (A, B, C, D) */
    lt_matrix_layout_t *mats[4] = {layA, layB, layC, layD};
    for (int i = 0; i < 4; i++) {
        lt_matrix_layout_t *l = mats[i] ? mats[i] : layD;
        int32_t dtype = l->dtype;
        uint64_t rows = l->rows, cols = l->cols;
        int64_t ld = l->ld;
        int32_t batch = (int32_t)l->batch_count;
        int64_t stride = l->strided_batch_offset;
        memcpy(p, &dtype, 4); p += 4;
        memcpy(p, &rows, 8); p += 8;
        memcpy(p, &cols, 8); p += 8;
        memcpy(p, &ld, 8); p += 8;
        memcpy(p, &batch, 4); p += 4;
        memcpy(p, &stride, 8); p += 8;
    }
    /* 16 bytes: alpha(8) + beta(8) */
    memset(p, 0, 8);
    if (alpha) memcpy(p, alpha, 4);
    else { float one = 1.0f; memcpy(p, &one, 4); }
    p += 8;
    memset(p, 0, 8);
    if (beta) memcpy(p, beta, 4);
    p += 8;
    /* 40 bytes: A_ptr B_ptr C_ptr D_ptr stream */
    uint64_t ptrs[5];
    ptrs[0] = (uint64_t)(uintptr_t)A;
    ptrs[1] = (uint64_t)(uintptr_t)B;
    ptrs[2] = (uint64_t)(uintptr_t)(layC ? C : D);
    ptrs[3] = (uint64_t)(uintptr_t)D;
    ptrs[4] = (uint64_t)(uintptr_t)stream;
    memcpy(p, ptrs, 40); p += 40;
    /* 64 bytes: algo */
    if (algo) memcpy(p, algo, 64);
    else memset(p, 0, 64);

    return (cublasStatus_t)chidori_rpc_async(CHIDORI_CUBLAS_LT_MATMUL, payload, 308);
}

cublasStatus_t cublasLtMatmulDescCreate(void **matmulDesc, int computeType, int scaleType) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void **, int, int);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulDescCreate");
            if (real_fn) return real_fn(matmulDesc, computeType, scaleType);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(void **, int, int);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulDescCreate"); }
        if (real_fn) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(matmulDesc, computeType, scaleType);
            g_in_real_cuda--;
            return r;
        }
    }
    if (g_lt_desc_count >= LT_DESC_POOL) { if (matmulDesc) *matmulDesc = NULL; return CUBLAS_STATUS_ALLOC_FAILED; }
    lt_matmul_desc_t *d = &g_lt_descs[g_lt_desc_count++];
    memset(d, 0, sizeof(*d));
    d->computeType = computeType;
    d->scaleType = scaleType;
    if (matmulDesc) *matmulDesc = (void *)d;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(void *matmulDesc) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulDescDestroy");
            if (real_fn) return real_fn(matmulDesc);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(void *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulDescDestroy"); }
        if (real_fn) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(matmulDesc);
            g_in_real_cuda--;
            return r;
        }
    }
    (void)matmulDesc;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(void *matmulDesc, int attr,
                                                const void *buf, size_t sizeInBytes) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void *, int, const void *, size_t);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulDescSetAttribute");
            if (real_fn) return real_fn(matmulDesc, attr, buf, sizeInBytes);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(void *, int, const void *, size_t);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulDescSetAttribute"); }
        if (real_fn) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(matmulDesc, attr, buf, sizeInBytes);
            g_in_real_cuda--;
            return r;
        }
    }
    lt_matmul_desc_t *d = (lt_matmul_desc_t *)matmulDesc;
    if (!d || !buf) return CUBLAS_STATUS_SUCCESS;
    if (attr == CUBLASLT_MATMUL_DESC_TRANSA && sizeInBytes >= 4) d->transa = *(const int *)buf;
    if (attr == CUBLASLT_MATMUL_DESC_TRANSB && sizeInBytes >= 4) d->transb = *(const int *)buf;
    if (attr == CUBLASLT_MATMUL_DESC_EPILOGUE && sizeInBytes >= 4) d->epilogue = *(const int *)buf;
    if (attr == CUBLASLT_MATMUL_DESC_BIAS_POINTER && sizeInBytes >= 8) d->bias_pointer = *(const uint64_t *)buf;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescGetAttribute(void *matmulDesc, int attr,
                                                void *buf, size_t sizeInBytes,
                                                size_t *sizeWritten) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void *, int, void *, size_t, size_t *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulDescGetAttribute");
            if (real_fn) return real_fn(matmulDesc, attr, buf, sizeInBytes, sizeWritten);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(void *, int, void *, size_t, size_t *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulDescGetAttribute"); }
        if (real_fn) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(matmulDesc, attr, buf, sizeInBytes, sizeWritten);
            g_in_real_cuda--;
            return r;
        }
    }
    lt_matmul_desc_t *d = (lt_matmul_desc_t *)matmulDesc;
    if (buf) memset(buf, 0, sizeInBytes);
    if (d && buf) {
        if (attr == CUBLASLT_MATMUL_DESC_TRANSA && sizeInBytes >= 4) *(int *)buf = d->transa;
        if (attr == CUBLASLT_MATMUL_DESC_TRANSB && sizeInBytes >= 4) *(int *)buf = d->transb;
    }
    if (sizeWritten) *sizeWritten = sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutCreate(void **matLayout, int type,
                                            uint64_t rows, uint64_t cols, int64_t ld) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void **, int, uint64_t, uint64_t, int64_t);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatrixLayoutCreate");
            if (real_fn) return real_fn(matLayout, type, rows, cols, ld);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(void **, int, uint64_t, uint64_t, int64_t);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatrixLayoutCreate"); }
        if (real_fn) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(matLayout, type, rows, cols, ld);
            g_in_real_cuda--;
            return r;
        }
    }
    if (g_lt_layout_count >= LT_DESC_POOL * 4) { if (matLayout) *matLayout = NULL; return CUBLAS_STATUS_ALLOC_FAILED; }
    lt_matrix_layout_t *l = &g_lt_layouts[g_lt_layout_count++];
    memset(l, 0, sizeof(*l));
    l->dtype = type;
    l->rows = rows;
    l->cols = cols;
    l->ld = ld;
    l->batch_count = 1;
    if (matLayout) *matLayout = (void *)l;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(void *matLayout) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatrixLayoutDestroy");
            if (real_fn) return real_fn(matLayout);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(void *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatrixLayoutDestroy"); }
        if (real_fn) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(matLayout);
            g_in_real_cuda--;
            return r;
        }
    }
    (void)matLayout;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(void *matLayout, int attr,
                                                  const void *buf, size_t sizeInBytes) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void *, int, const void *, size_t);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatrixLayoutSetAttribute");
            if (real_fn) return real_fn(matLayout, attr, buf, sizeInBytes);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(void *, int, const void *, size_t);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatrixLayoutSetAttribute"); }
        if (real_fn) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(matLayout, attr, buf, sizeInBytes);
            g_in_real_cuda--;
            return r;
        }
    }
    lt_matrix_layout_t *l = (lt_matrix_layout_t *)matLayout;
    if (!l || !buf) return CUBLAS_STATUS_SUCCESS;
    if (attr == CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT && sizeInBytes >= 4)
        l->batch_count = *(const int32_t *)buf;
    if (attr == CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET && sizeInBytes >= 8)
        l->strided_batch_offset = *(const int64_t *)buf;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(void *matLayout, int attr,
                                                  void *buf, size_t sizeInBytes,
                                                  size_t *sizeWritten) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void *, int, void *, size_t, size_t *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatrixLayoutGetAttribute");
            if (real_fn) return real_fn(matLayout, attr, buf, sizeInBytes, sizeWritten);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(void *, int, void *, size_t, size_t *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatrixLayoutGetAttribute"); }
        if (real_fn) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(matLayout, attr, buf, sizeInBytes, sizeWritten);
            g_in_real_cuda--;
            return r;
        }
    }
    lt_matrix_layout_t *l = (lt_matrix_layout_t *)matLayout;
    if (buf) memset(buf, 0, sizeInBytes);
    if (l && buf) {
        if (attr == CUBLASLT_MATRIX_LAYOUT_ROWS && sizeInBytes >= 8) *(uint64_t *)buf = l->rows;
        if (attr == CUBLASLT_MATRIX_LAYOUT_COLS && sizeInBytes >= 8) *(uint64_t *)buf = l->cols;
        if (attr == CUBLASLT_MATRIX_LAYOUT_LD && sizeInBytes >= 8) *(int64_t *)buf = l->ld;
        if (attr == CUBLASLT_MATRIX_LAYOUT_TYPE && sizeInBytes >= 4) *(int *)buf = l->dtype;
    }
    if (sizeWritten) *sizeWritten = sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceCreate(void **pref) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void **);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulPreferenceCreate");
            if (real_fn) return real_fn(pref);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(void **);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulPreferenceCreate"); }
        if (real_fn) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(pref);
            g_in_real_cuda--;
            return r;
        }
    }
    static int fake_pref;
    if (pref) *pref = (void *)&fake_pref;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(void *pref) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulPreferenceDestroy");
            if (real_fn) return real_fn(pref);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(void *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulPreferenceDestroy"); }
        if (real_fn) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(pref);
            g_in_real_cuda--;
            return r;
        }
    }
    (void)pref;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(void *pref, int attr,
                                                      const void *buf, size_t sizeInBytes) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void *, int, const void *, size_t);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulPreferenceSetAttribute");
            if (real_fn) return real_fn(pref, attr, buf, sizeInBytes);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(void *, int, const void *, size_t);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulPreferenceSetAttribute"); }
        if (real_fn) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(pref, attr, buf, sizeInBytes);
            g_in_real_cuda--;
            return r;
        }
    }
    (void)pref; (void)attr; (void)buf; (void)sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
    void *operationDesc, void *Adesc, void *Bdesc, void *Cdesc, void *Ddesc,
    void *preference, int requestedAlgoCount, void *heuristicResultsArray,
    int *returnAlgoCount) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(cublasLtHandle_t, void *, void *, void *, void *, void *, void *, int, void *, int *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulAlgoGetHeuristic");
            if (real_fn) return real_fn(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResultsArray, returnAlgoCount);
        }
    }
    /* LOCAL DEVICE ROUTING */
    if (g_partial_offload && is_local_device(g_current_device)) {
        typedef cublasStatus_t (*fn_t)(cublasLtHandle_t, void *, void *, void *, void *, void *, void *, int, void *, int *);
        static fn_t real_fn = NULL;
        if (!real_fn) { void *h = ensure_real_cublaslt(); if (h) real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulAlgoGetHeuristic"); }
        cublasLtHandle_t local_h = ensure_local_cublaslt_handle();
        if (real_fn && local_h) {
            g_in_real_cuda++;
            cublasStatus_t r = real_fn(local_h, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResultsArray, returnAlgoCount);
            g_in_real_cuda--;
            return r;
        }
    }
    (void)lightHandle; (void)preference; (void)requestedAlgoCount;

    lt_matmul_desc_t *desc = (lt_matmul_desc_t *)operationDesc;
    lt_matrix_layout_t *layA = (lt_matrix_layout_t *)Adesc;
    lt_matrix_layout_t *layB = (lt_matrix_layout_t *)Bdesc;
    lt_matrix_layout_t *layC = (lt_matrix_layout_t *)Cdesc;
    lt_matrix_layout_t *layD = (lt_matrix_layout_t *)Ddesc;

    if (!desc || !layA || !layB || !layD) {
        if (returnAlgoCount) *returnAlgoCount = 0;
        return CUBLAS_STATUS_SUCCESS;
    }

    /* Build payload with raw descriptor fields (188 bytes):
     * transa(4) transb(4) computeType(4) scaleType(4) epilogue(4) bias_ptr(8) = 28
     * For each A,B,C,D: dtype(4) rows(8) cols(8) ld(8) batch(4) stride(8) = 40 x 4 = 160 */
    uint8_t payload[188];
    char *p = (char *)payload;
    int32_t transa_v = desc->transa, transb_v = desc->transb;
    int32_t computeType_v = desc->computeType, scaleType_v = desc->scaleType;
    int32_t epilogue_v = desc->epilogue;
    uint64_t bias_ptr_v = desc->bias_pointer;
    memcpy(p, &transa_v, 4); p += 4;
    memcpy(p, &transb_v, 4); p += 4;
    memcpy(p, &computeType_v, 4); p += 4;
    memcpy(p, &scaleType_v, 4); p += 4;
    memcpy(p, &epilogue_v, 4); p += 4;
    memcpy(p, &bias_ptr_v, 8); p += 8;

    lt_matrix_layout_t *mats[4] = {layA, layB, layC, layD};
    for (int i = 0; i < 4; i++) {
        lt_matrix_layout_t *l = mats[i] ? mats[i] : layD; /* fallback to D if C is NULL */
        int32_t dtype = l->dtype;
        uint64_t rows = l->rows, cols = l->cols;
        int64_t ld = l->ld;
        int32_t batch = (int32_t)l->batch_count;
        int64_t stride = l->strided_batch_offset;
        memcpy(p, &dtype, 4); p += 4;
        memcpy(p, &rows, 8); p += 8;
        memcpy(p, &cols, 8); p += 8;
        memcpy(p, &ld, 8); p += 8;
        memcpy(p, &batch, 4); p += 4;
        memcpy(p, &stride, 8); p += 8;
    }

    /* RPC to server */
    uint8_t resp[4 + 96]; /* algoCount(4) + cublasLtMatmulHeuristicResult_t (96) */
    size_t out_len = sizeof(resp);
    cublasStatus_t st = (cublasStatus_t)chidori_rpc(
        CHIDORI_CUBLAS_LT_MATMUL_ALGO_HEURISTIC, payload, 188, resp, &out_len);

    int32_t algoCount = 0;
    if (out_len >= 4) memcpy(&algoCount, resp, 4);
    if (returnAlgoCount) *returnAlgoCount = algoCount;
    if (algoCount > 0 && heuristicResultsArray && out_len > 4)
        memcpy(heuristicResultsArray, resp + 4, out_len - 4);

    return st;
}

/* Additional cublasLt stubs that PyTorch may call after getting heuristic results */
cublasStatus_t cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle,
    int computeType, int scaleType, int Atype, int Btype, int Ctype, int Dtype,
    int algoId, void *algo) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(cublasLtHandle_t, int, int, int, int, int, int, int, void *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulAlgoInit");
            if (real_fn) return real_fn(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, algoId, algo);
        }
    }
    (void)lightHandle; (void)computeType; (void)scaleType;
    (void)Atype; (void)Btype; (void)Ctype; (void)Dtype; (void)algoId;
    if (algo) memset(algo, 0, 64);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(void *algo, int attr,
    const void *buf, size_t sizeInBytes) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void *, int, const void *, size_t);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulAlgoConfigSetAttribute");
            if (real_fn) return real_fn(algo, attr, buf, sizeInBytes);
        }
    }
    (void)algo; (void)attr; (void)buf; (void)sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const void *algo, int attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(const void *, int, void *, size_t, size_t *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulAlgoConfigGetAttribute");
            if (real_fn) return real_fn(algo, attr, buf, sizeInBytes, sizeWritten);
        }
    }
    (void)algo; (void)attr;
    if (buf) memset(buf, 0, sizeInBytes);
    if (sizeWritten) *sizeWritten = sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoCheck(cublasLtHandle_t lightHandle,
    void *operationDesc, void *Adesc, void *Bdesc, void *Cdesc, void *Ddesc,
    const void *algo, void *result) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(cublasLtHandle_t, void *, void *, void *, void *, void *, const void *, void *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulAlgoCheck");
            if (real_fn) return real_fn(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, algo, result);
        }
    }
    (void)lightHandle; (void)operationDesc; (void)Adesc; (void)Bdesc;
    (void)Cdesc; (void)Ddesc; (void)algo;
    if (result) memset(result, 0, 280);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoGetIds(cublasLtHandle_t lightHandle,
    int computeType, int scaleType, int Atype, int Btype, int Ctype, int Dtype,
    int requestedAlgoCount, int *algoIdsArray, int *returnAlgoCount) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(cublasLtHandle_t, int, int, int, int, int, int, int, int *, int *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtMatmulAlgoGetIds");
            if (real_fn) return real_fn(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, requestedAlgoCount, algoIdsArray, returnAlgoCount);
        }
    }
    (void)lightHandle; (void)computeType; (void)scaleType;
    (void)Atype; (void)Btype; (void)Ctype; (void)Dtype;
    (void)requestedAlgoCount; (void)algoIdsArray;
    if (returnAlgoCount) *returnAlgoCount = 0;
    return CUBLAS_STATUS_SUCCESS;
}

size_t cublasLtGetCudartVersion(void) {
    /* ALWAYS forward via RTLD_NEXT — cuBLAS calls at library load time.
     * Can't use dlsym(handle) because our function may be in the handle's scope. */
    typedef size_t (*fn_t)(void);
    static fn_t real_fn = NULL;
    if (!real_fn) real_fn = (fn_t)__real_dlsym(RTLD_NEXT, "cublasLtGetCudartVersion");
    if (real_fn) return real_fn();
    return 12040; /* fallback */
}
cublasStatus_t cublasLtGetStatusName(cublasStatus_t status, const char **str) {
    typedef cublasStatus_t (*fn_t)(cublasStatus_t, const char **);
    static fn_t real_fn = NULL;
    if (!real_fn) real_fn = (fn_t)__real_dlsym(RTLD_NEXT, "cublasLtGetStatusName");
    if (real_fn) return real_fn(status, str);
    (void)status; if (str) *str = "CUBLAS_STATUS_SUCCESS"; return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtGetStatusString(cublasStatus_t status, const char **str) {
    typedef cublasStatus_t (*fn_t)(cublasStatus_t, const char **);
    static fn_t real_fn = NULL;
    if (!real_fn) real_fn = (fn_t)__real_dlsym(RTLD_NEXT, "cublasLtGetStatusString");
    if (real_fn) return real_fn(status, str);
    (void)status; if (str) *str = "success"; return CUBLAS_STATUS_SUCCESS;
}
size_t cublasLtGetVersion(void) {
    typedef size_t (*fn_t)(void);
    static fn_t real_fn = NULL;
    if (!real_fn) real_fn = (fn_t)__real_dlsym(RTLD_NEXT, "cublasLtGetVersion");
    if (real_fn) return real_fn();
    return 120405; /* fallback */
}
cublasStatus_t cublasLtGetProperty(int type, int *value) {
    typedef cublasStatus_t (*fn_t)(int, int *);
    static fn_t real_fn = NULL;
    if (!real_fn) real_fn = (fn_t)__real_dlsym(RTLD_NEXT, "cublasLtGetProperty");
    if (real_fn) return real_fn(type, value);
    void *h = ensure_real_cublaslt();
    if (h) {
        typedef cublasStatus_t (*fn_t)(int, int *);
        fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtGetProperty");
        if (real_fn) return real_fn(type, value);
    }
    if (value) *value = (type == 0) ? 12 : (type == 1) ? 4 : 5;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtLoggerSetCallback(void *callback) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtLoggerSetCallback");
            if (real_fn) return real_fn(callback);
        }
    }
    (void)callback; return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtLoggerSetLevel(int level) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(int);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtLoggerSetLevel");
            if (real_fn) return real_fn(level);
        }
    }
    (void)level; return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtLoggerSetMask(int mask) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(int);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtLoggerSetMask");
            if (real_fn) return real_fn(mask);
        }
    }
    (void)mask; return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtLoggerSetFile(void *file) {
    /* REENTRANCE GUARD: forward to real cublasLt when inside cuBLAS init */
    if (g_in_real_cuda > 0) {
        void *h = ensure_real_cublaslt();
        if (h) {
            typedef cublasStatus_t (*fn_t)(void *);
            fn_t real_fn = (fn_t)__real_dlsym(h, "cublasLtLoggerSetFile");
            if (real_fn) return real_fn(file);
        }
    }
    (void)file; return CUBLAS_STATUS_SUCCESS;
}

/*
 * ============================================================================
 * cuDNN Compatibility Stubs (additional — beyond the RPC implementations above)
 * ============================================================================
 */

const char *cudnnGetErrorString(cudnnStatus_t status) {
    switch (status) {
    case 0: return "CUDNN_STATUS_SUCCESS";
    case 1: return "CUDNN_STATUS_NOT_INITIALIZED";
    case 2: return "CUDNN_STATUS_ALLOC_FAILED";
    case 3: return "CUDNN_STATUS_BAD_PARAM";
    case 9: return "CUDNN_STATUS_NOT_SUPPORTED";
    default: return "CUDNN_STATUS_INTERNAL_ERROR";
    }
}

size_t cudnnGetCudartVersion(void) {
    return 12000;
}

cudnnStatus_t cudnnGetProperty(int type, int *value) {
    switch (type) {
    case 0: if (value) *value = 8; break;  /* major */
    case 1: if (value) *value = 9; break;  /* minor */
    case 2: if (value) *value = 4; break;  /* patch */
    default: if (value) *value = 0; break;
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, void **streamId) {
    (void)handle;
    if (streamId) *streamId = NULL;
    return CUDNN_STATUS_SUCCESS;
}

/*
 * ============================================================================
 * NVRTC Stubs — PyTorch uses NVRTC for runtime compilation
 * ============================================================================
 */

typedef int nvrtcResult;
#define NVRTC_SUCCESS 0

const char *nvrtcGetErrorString(nvrtcResult result) {
    (void)result;
    return "NVRTC success";
}

nvrtcResult nvrtcVersion(int *major, int *minor) {
    if (major) *major = 12;
    if (minor) *minor = 0;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCreateProgram(void **prog, const char *src,
                                const char *name, int numHeaders,
                                const char * const *headers,
                                const char * const *includeNames) {
    (void)src; (void)name; (void)numHeaders; (void)headers; (void)includeNames;
    static int fake_prog;
    if (prog) *prog = (void *)&fake_prog;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcDestroyProgram(void **prog) {
    (void)prog;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCompileProgram(void *prog, int numOptions, const char * const *options) {
    (void)prog; (void)numOptions; (void)options;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetPTXSize(void *prog, size_t *ptxSizeRet) {
    (void)prog;
    if (ptxSizeRet) *ptxSizeRet = 1;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetPTX(void *prog, char *ptx) {
    (void)prog;
    if (ptx) ptx[0] = '\0';
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetCUBINSize(void *prog, size_t *cubinSizeRet) {
    (void)prog;
    if (cubinSizeRet) *cubinSizeRet = 0;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetCUBIN(void *prog, char *cubin) {
    (void)prog; (void)cubin;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetProgramLogSize(void *prog, size_t *logSizeRet) {
    (void)prog;
    if (logSizeRet) *logSizeRet = 1;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetProgramLog(void *prog, char *log) {
    (void)prog;
    if (log) log[0] = '\0';
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcAddNameExpression(void *prog, const char *name_expression) {
    (void)prog; (void)name_expression;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetLoweredName(void *prog, const char *name_expression,
                                 const char **lowered_name) {
    (void)prog; (void)name_expression;
    if (lowered_name) *lowered_name = "";
    return NVRTC_SUCCESS;
}

/* ====================================================================
 * NVML Interception for Partial GPU Offloading
 *
 * PyTorch calls nvmlDeviceGetCount_v2 (via libnvidia-ml.so.1) FIRST
 * to get the device count, BEFORE calling cudaGetDeviceCount. If NVML
 * returns a count, PyTorch uses it and never calls our cudaGetDeviceCount.
 * We must intercept NVML to report the merged device count.
 * ==================================================================== */

#define NVML_SUCCESS 0

int nvmlInit(void) { return NVML_SUCCESS; }
int nvmlInit_v2(void) { return NVML_SUCCESS; }

int nvmlDeviceGetCount(unsigned int *deviceCount) {
    init_real_cuda();
    if (g_partial_offload && deviceCount) {
        /* Report local + remote device count */
        int remote_count = 0;
        size_t len = sizeof(int32_t);
        chidori_rpc(CHIDORI_CUDA_GET_DEVICE_COUNT, NULL, 0, &remote_count, &len);
        if (remote_count <= 0) remote_count = 1;
        *deviceCount = (unsigned int)(g_local_device_count + remote_count);
        fprintf(stderr, "[chidori] nvmlDeviceGetCount: local=%d + remote=%d = %u\n",
                g_local_device_count, remote_count, *deviceCount);
        return NVML_SUCCESS;
    }
    /* Non-partial: query real NVML */
    typedef int (*nvml_fn)(unsigned int *);
    static nvml_fn real_fn = NULL;
    if (!real_fn) {
        void *nvml = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (nvml) real_fn = (nvml_fn)__real_dlsym(nvml, "nvmlDeviceGetCount");
    }
    if (real_fn) return real_fn(deviceCount);
    if (deviceCount) *deviceCount = 1;
    return NVML_SUCCESS;
}

int nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
    return nvmlDeviceGetCount(deviceCount);
}

int nvmlShutdown(void) { return NVML_SUCCESS; }

/*
 * cudaGetDriverEntryPoint / cudaGetDriverEntryPointByVersion interception.
 *
 * PyTorch resolves CUDA Driver API symbols via cudaGetDriverEntryPoint()
 * (from libcudart), not via direct function calls or dlsym. Without
 * intercepting this, PyTorch gets the real driver's cuDeviceGetCount
 * (returning 1 GPU) instead of our merged count (2 GPUs for partial offload).
 *
 * We maintain a lookup table of symbol name → our function pointer.
 * When PyTorch requests a symbol we intercept, we return ours.
 */

/* Names of CUDA Driver API symbols we intercept. When PyTorch requests
 * any of these via cudaGetDriverEntryPoint, we return our own function
 * pointer (found via dlsym(RTLD_DEFAULT) on ourselves). */
static const char *g_intercepted_symbols[] = {
    "cuInit", "cuDriverGetVersion",
    "cuDeviceGet", "cuDeviceGetCount", "cuDeviceGetName",
    "cuDeviceTotalMem", "cuDeviceTotalMem_v2",
    "cuDeviceGetAttribute", "cuDeviceGetUuid",
    "cuCtxGetCurrent", "cuCtxSetCurrent", "cuCtxSynchronize",
    "cuModuleLoadData", "cuModuleGetFunction", "cuModuleUnload",
    "cuMemAlloc", "cuMemAlloc_v2", "cuMemFree", "cuMemFree_v2",
    "cuMemcpyHtoD", "cuMemcpyHtoD_v2", "cuMemcpyDtoH", "cuMemcpyDtoH_v2",
    "cuMemcpyDtoD", "cuMemcpyDtoD_v2",
    "cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2",
    "cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2",
    "cuMemsetD8", "cuMemsetD8_v2", "cuMemsetD32", "cuMemsetD32_v2",
    "cuLaunchKernel",
    "cuStreamCreate", "cuStreamSynchronize", "cuStreamDestroy", "cuStreamDestroy_v2",
    "cuEventCreate", "cuEventRecord", "cuEventSynchronize",
    "cuEventDestroy", "cuEventDestroy_v2", "cuEventElapsedTime",
    "cuFuncGetAttribute",
    "cuDevicePrimaryCtxRetain", "cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2",
    "cuDevicePrimaryCtxGetState", "cuDevicePrimaryCtxSetFlags", "cuDevicePrimaryCtxSetFlags_v2",
    "cuDevicePrimaryCtxReset", "cuDevicePrimaryCtxReset_v2",
    "cuDeviceGetPCIBusId",
    "cuGetErrorString", "cuGetErrorName",
    "cuPointerGetAttribute",
    "cuGetExportTable",
    NULL
};

static void *lookup_intercepted_symbol(const char *symbol) {
    for (int i = 0; g_intercepted_symbols[i] != NULL; i++) {
        if (strcmp(symbol, g_intercepted_symbols[i]) == 0)
            return __real_dlsym(RTLD_DEFAULT, symbol);
    }
    return NULL;
}

/*
 * cuGetProcAddress / cuGetProcAddress_v2 — CUDA 11.3+ function pointer resolver.
 *
 * This is the CUDA Driver API equivalent of cudaGetDriverEntryPoint.
 * PyTorch's libcudart uses this internally to resolve driver symbols.
 * By intercepting it, we ensure ALL driver symbol lookups return our
 * function pointers for partial GPU offloading.
 */
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, uint64_t flags) {
    (void)cudaVersion; (void)flags;
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;

    /* ALWAYS return OUR intercepted functions — NEVER forward to real driver.
     * Forwarding during reentrance causes cudart to cache real driver pointers,
     * which then bypass our routing for ALL subsequent calls (including remote
     * device operations). cuBLAS isolation is handled by RTLD_DEEPBIND instead.
     *
     * Return OUR intercepted functions for ALL modes (including partial offload).
     * Our functions handle the local/remote routing via g_in_real_cuda guard.
     * If we returned real driver pointers here, cudart would bypass our
     * interception for ALL operations including remote device kernel launches.
     *
     * Exception: cuGetProcAddress itself — prevent infinite recursion. */
    if (strcmp(symbol, "cuGetProcAddress") != 0 &&
        strcmp(symbol, "cuGetProcAddress_v2") != 0) {
        void *ours = lookup_intercepted_symbol(symbol);
        if (ours) {
            *pfn = ours;
            return CUDA_SUCCESS;
        }
    }

    /* Symbol not in our table — forward to real driver if available */
    if (g_real_cuda_handle) {
        typedef CUresult (*fn_t)(const char *, void **, int, uint64_t);
        static fn_t real_fn = NULL;
        if (!real_fn) real_fn = (fn_t)__real_dlsym(g_real_cuda_handle, "cuGetProcAddress");
        if (real_fn) return real_fn(symbol, pfn, cudaVersion, flags);
    }

    *pfn = NULL;
    return CUDA_ERROR_NOT_FOUND;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                              uint64_t flags, void *symbolStatus) {
    CUresult r = cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    if (symbolStatus) *(int *)symbolStatus = (r == CUDA_SUCCESS) ? 1 : 0;
    return r;
}

/*
 * cudaGetDriverEntryPoint — CUDA Runtime API function that PyTorch uses
 * to resolve Driver API symbols. We intercept it to return our function
 * pointers instead of the real driver's.
 */
cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr,
                                     unsigned long long flags) {
    if (!symbol || !funcPtr) return (cudaError_t)1;

    void *ours = lookup_intercepted_symbol(symbol);
    if (ours) {
        *funcPtr = ours;
        LOG_DEBUG("cudaGetDriverEntryPoint('%s') → OUR interception", symbol);
        return cudaSuccess;
    }

    /* Not intercepted — forward to real implementation */
    LOG_DEBUG("cudaGetDriverEntryPoint('%s') → PASSTHROUGH", symbol);
    /* The real cudaGetDriverEntryPoint is in libcudart. Since we're
     * libcuda.so.1 (not libcudart), the real function should be
     * resolvable via the normal symbol chain. */
    typedef cudaError_t (*real_fn_t)(const char *, void **, unsigned long long);
    static real_fn_t real_fn = NULL;
    if (!real_fn) {
        void *cudart = dlopen("libcudart.so", RTLD_LAZY | RTLD_NOLOAD);
        if (cudart)
            real_fn = (real_fn_t)__real_dlsym(cudart, "cudaGetDriverEntryPoint");
    }
    if (real_fn)
        return real_fn(symbol, funcPtr, flags);

    *funcPtr = NULL;
    return (cudaError_t)500; /* cudaErrorNotFound */
}

cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr,
                                              unsigned int cudaVersion,
                                              unsigned long long flags,
                                              int *queryResult) {
    if (!symbol || !funcPtr) return (cudaError_t)1;

    void *ours = lookup_intercepted_symbol(symbol);
    if (ours) {
        *funcPtr = ours;
        if (queryResult) *queryResult = 1; /* cudaDriverEntryPointSuccess */
        LOG_DEBUG("cudaGetDriverEntryPointByVersion('%s', v%u) → OUR interception", symbol, cudaVersion);
        return cudaSuccess;
    }

    LOG_DEBUG("cudaGetDriverEntryPointByVersion('%s', v%u) → PASSTHROUGH", symbol, cudaVersion);
    typedef cudaError_t (*real_fn_t)(const char *, void **, unsigned int, unsigned long long, int *);
    static real_fn_t real_fn = NULL;
    if (!real_fn) {
        void *cudart = dlopen("libcudart.so", RTLD_LAZY | RTLD_NOLOAD);
        if (cudart)
            real_fn = (real_fn_t)__real_dlsym(cudart, "cudaGetDriverEntryPointByVersion");
    }
    if (real_fn)
        return real_fn(symbol, funcPtr, cudaVersion, flags, queryResult);

    *funcPtr = NULL;
    if (queryResult) *queryResult = 0;
    return (cudaError_t)500;
}

/*
 * Cleanup on library unload
 */
__attribute__((destructor))
static void chidori_cleanup(void) {
    if (g_server_fd >= 0) {
        close(g_server_fd);
        fprintf(stderr, "[chidori] SOCKET CLOSED at %s:%d\n", __FILE__, __LINE__); g_server_fd = -1;
    }
}
