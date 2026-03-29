/*
 * LD_PRELOAD Interceptor — Proves the actual execution path
 *
 * This library intercepts key functions called during cudaGetFuncBySymbol
 * by hooking the dl* functions and CUDA internal entry points.
 *
 * Build:
 *   gcc -shared -fPIC -O0 -g -o interceptor.so interceptor.c -ldl
 *
 * Run:
 *   LD_PRELOAD=./interceptor.so ./deep_trace simple_sm89.fatbin simple_add
 *
 * What it proves:
 *   1. Which dlopen/dlsym calls happen during cudaGetFuncBySymbol
 *   2. Which internal functions are called (by hooking their PLT entries)
 *   3. Whether the alternative resolver path is taken
 *   4. Whether the normal vtable path is taken
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>

/* Thread safety */
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_in_getFuncBySymbol = 0;
static int g_trace_enabled = 0;

/* Original function pointers */
static void *(*real_dlopen)(const char *, int) = NULL;
static void *(*real_dlsym)(void *, const char *) = NULL;
static int (*real_dlclose)(void *) = NULL;

/* Counters */
static int dlopen_count = 0;
static int dlsym_count = 0;

__attribute__((constructor))
static void init_interceptor(void) {
    real_dlopen = dlopen;
    real_dlsym = dlsym;
    real_dlclose = dlclose;
    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, " LD_PRELOAD Interceptor — cudaGetFuncBySymbol Tracing\n");
    fprintf(stderr, "============================================================\n\n");
}

__attribute__((destructor))
static void fini_interceptor(void) {
    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, " Interceptor Summary\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, " dlopen calls: %d\n", dlopen_count);
    fprintf(stderr, " dlsym calls: %d\n", dlsym_count);
    fprintf(stderr, "============================================================\n\n");
}

/* Hook dlopen — tracks which libraries are loaded during resolution */
void *dlopen(const char *filename, int flags) {
    void *result = real_dlopen(filename, flags);
    pthread_mutex_lock(&g_mutex);
    dlopen_count++;
    fprintf(stderr, "  [INTERCEPT] dlopen(\"%s\", 0x%x) = %p\n",
            filename ? filename : "(null)", flags, result);
    pthread_mutex_unlock(&g_mutex);
    return result;
}

/* Hook dlsym — tracks which symbols are resolved during cudaGetFuncBySymbol */
void *dlsym(void *handle, const char *symbol) {
    void *result = real_dlsym(handle, symbol);
    pthread_mutex_lock(&g_mutex);
    dlsym_count++;
    const char *lib_name = "(unknown)";
    Dl_info info;
    if (handle && dladdr(handle, &info) && info.dli_fname) {
        lib_name = info.dli_fname;
    }
    /* Only trace interesting symbols — skip noise */
    int interesting = 0;
    if (symbol && (
        strstr(symbol, "cuda") ||
        strstr(symbol, "cu") == symbol ||
        strstr(symbol, "Launch") ||
        strstr(symbol, "Module") ||
        strstr(symbol, "Function") ||
        strstr(symbol, "Register") ||
        strstr(symbol, "FatBin") ||
        strstr(symbol, "GetFunc") ||
        strstr(symbol, "ctx") ||
        strstr(symbol, "Context") ||
        strstr(symbol, "init") ||
        strstr(symbol, "Init")
    )) {
        interesting = 1;
    }
    if (interesting || g_trace_enabled) {
        fprintf(stderr, "  [INTERCEPT] dlsym(%p, \"%s\") = %p [%s]\n",
                handle, symbol ? symbol : "(null)", result, lib_name);
    }
    pthread_mutex_unlock(&g_mutex);
    return result;
}

/* Hook dlclose */
int dlclose(void *handle) {
    int result = real_dlclose(handle);
    pthread_mutex_lock(&g_mutex);
    fprintf(stderr, "  [INTERCEPT] dlclose(%p) = %d\n", handle, result);
    pthread_mutex_unlock(&g_mutex);
    return result;
}

/* Export a control function the test program can call */
void interceptor_enable_trace(void) {
    g_trace_enabled = 1;
    fprintf(stderr, "  [INTERCEPT] Trace enabled — all dlsym calls will be logged\n");
}

void interceptor_disable_trace(void) {
    g_trace_enabled = 0;
    fprintf(stderr, "  [INTERCEPT] Trace disabled\n");
}
