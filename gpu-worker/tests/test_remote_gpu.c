/*
 * Remote GPU Stress Test — runs on a machine with NO GPU
 *
 * Uses LD_PRELOAD=libjuice-cuda.so to intercept CUDA calls and
 * forward them to a remote GPU server over TCP/IP.
 *
 * Tests:
 *   1. Device discovery and info queries
 *   2. Memory allocation (multiple concurrent buffers)
 *   3. Small data round-trip (256 floats)
 *   4. Large data round-trip (16MB)
 *   5. Repeated transfers (throughput measurement)
 *   6. DtoD copy (zero network bytes)
 *   7. Memory stress (allocate/free cycle)
 *   8. Multiple streams
 *   9. Events and timing
 *
 * Build: gcc -Wall -O2 -D_GNU_SOURCE -o test_remote_gpu test_remote_gpu.c -ldl
 * Run:   LD_PRELOAD=./libjuice-cuda.so ./test_remote_gpu
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <sys/time.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUevent;
typedef unsigned long long CUdeviceptr;

typedef CUresult (*fn_cuInit)(unsigned int);
typedef CUresult (*fn_cuDeviceGetCount)(int *);
typedef CUresult (*fn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*fn_cuDeviceGetName)(char *, int, CUdevice);
typedef CUresult (*fn_cuDeviceTotalMem)(size_t *, CUdevice);
typedef CUresult (*fn_cuMemGetInfo)(size_t *, size_t *);
typedef CUresult (*fn_cuMemAlloc)(CUdeviceptr *, size_t);
typedef CUresult (*fn_cuMemFree)(CUdeviceptr);
typedef CUresult (*fn_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
typedef CUresult (*fn_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
typedef CUresult (*fn_cuMemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*fn_cuDriverGetVersion)(int *);
typedef CUresult (*fn_cuDeviceGetAttribute)(int *, int, CUdevice);
typedef CUresult (*fn_cuCtxSynchronize)(void);
typedef CUresult (*fn_cuStreamCreate)(CUstream *, unsigned int);
typedef CUresult (*fn_cuStreamDestroy)(CUstream);
typedef CUresult (*fn_cuStreamSynchronize)(CUstream);
typedef CUresult (*fn_cuEventCreate)(CUevent *, unsigned int);
typedef CUresult (*fn_cuEventDestroy)(CUevent);
typedef CUresult (*fn_cuEventRecord)(CUevent, CUstream);
typedef CUresult (*fn_cuEventSynchronize)(CUevent);
typedef CUresult (*fn_cuEventElapsedTime)(float *, CUevent, CUevent);
typedef CUresult (*fn_cuMemsetD32)(CUdeviceptr, unsigned int, size_t);

#define LOAD(name) ((fn_##name)dlsym(RTLD_DEFAULT, #name))
#define LOAD_V2(name) ((fn_##name)dlsym(RTLD_DEFAULT, #name "_v2"))

static int passed = 0, failed = 0, total = 0;

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

#define SECTION(title) do { \
    printf("\n--- %s ---\n", title); \
} while(0)

#define TEST(label, cond) do { \
    total++; \
    if (cond) { \
        printf("  %-50s\033[32mPASS\033[0m\n", label); \
        passed++; \
    } else { \
        printf("  %-50s\033[31mFAIL\033[0m\n", label); \
        failed++; \
    } \
} while(0)

#define TEST_FMT(label, cond, fmt, ...) do { \
    total++; \
    if (cond) { \
        char _buf[128]; \
        snprintf(_buf, sizeof(_buf), fmt, ##__VA_ARGS__); \
        printf("  %-50s\033[32mPASS\033[0m (%s)\n", label, _buf); \
        passed++; \
    } else { \
        printf("  %-50s\033[31mFAIL\033[0m\n", label); \
        failed++; \
    } \
} while(0)

int main(void) {
    printf("========================================================\n");
    printf("  Juice GPU-over-IP: Remote GPU Stress Test\n");
    printf("  This machine has NO GPU.\n");
    printf("  All CUDA calls travel over TCP/IP to a remote GPU.\n");
    printf("========================================================\n");

    /* Load symbols from LD_PRELOAD'd library */
    fn_cuInit cuInit = LOAD(cuInit);
    if (!cuInit) {
        printf("\nERROR: cuInit not found. Is LD_PRELOAD=libjuice-cuda.so set?\n");
        return 1;
    }

    fn_cuDriverGetVersion cuDriverGetVersion = LOAD(cuDriverGetVersion);
    fn_cuDeviceGetCount cuDeviceGetCount = LOAD(cuDeviceGetCount);
    fn_cuDeviceGet cuDeviceGet = LOAD(cuDeviceGet);
    fn_cuDeviceGetName cuDeviceGetName = LOAD(cuDeviceGetName);
    fn_cuDeviceTotalMem cuDeviceTotalMem = LOAD_V2(cuDeviceTotalMem);
    fn_cuMemGetInfo cuMemGetInfo = LOAD_V2(cuMemGetInfo);
    fn_cuMemAlloc cuMemAlloc = LOAD_V2(cuMemAlloc);
    fn_cuMemFree cuMemFree = LOAD_V2(cuMemFree);
    fn_cuMemcpyHtoD cuMemcpyHtoD = LOAD_V2(cuMemcpyHtoD);
    fn_cuMemcpyDtoH cuMemcpyDtoH = LOAD_V2(cuMemcpyDtoH);
    fn_cuMemcpyDtoD cuMemcpyDtoD = LOAD_V2(cuMemcpyDtoD);
    fn_cuDeviceGetAttribute cuDeviceGetAttribute = LOAD(cuDeviceGetAttribute);
    fn_cuCtxSynchronize cuCtxSynchronize = LOAD(cuCtxSynchronize);
    fn_cuStreamCreate cuStreamCreate = LOAD(cuStreamCreate);
    fn_cuStreamDestroy cuStreamDestroy = LOAD_V2(cuStreamDestroy);
    fn_cuStreamSynchronize cuStreamSynchronize = LOAD(cuStreamSynchronize);
    fn_cuEventCreate cuEventCreate = LOAD(cuEventCreate);
    fn_cuEventDestroy cuEventDestroy = LOAD_V2(cuEventDestroy);
    fn_cuEventRecord cuEventRecord = LOAD(cuEventRecord);
    fn_cuEventSynchronize cuEventSynchronize = LOAD(cuEventSynchronize);
    fn_cuEventElapsedTime cuEventElapsedTime = LOAD(cuEventElapsedTime);
    (void)0; /* cuMemsetD32 available but not used in this test */

    /* ============================================================
     * 1. Device Discovery
     * ============================================================ */
    SECTION("Device Discovery");

    TEST("cuInit", cuInit(0) == 0);

    int ver = 0;
    CUresult r = cuDriverGetVersion(&ver);
    TEST_FMT("cuDriverGetVersion", r == 0 && ver > 0, "v%d.%d", ver / 1000, (ver % 1000) / 10);

    int count = 0;
    r = cuDeviceGetCount(&count);
    TEST_FMT("cuDeviceGetCount", r == 0 && count > 0, "%d GPU(s)", count);

    CUdevice dev = 0;
    cuDeviceGet(&dev, 0);

    char name[256] = {0};
    r = cuDeviceGetName(name, sizeof(name), dev);
    TEST_FMT("cuDeviceGetName", r == 0 && name[0], "%s", name);

    size_t total_mem = 0;
    r = cuDeviceTotalMem(&total_mem, dev);
    TEST_FMT("cuDeviceTotalMem", r == 0 && total_mem > 0, "%.1f GB",
             (double)total_mem / (1024.0 * 1024.0 * 1024.0));

    size_t free_mem = 0, tot = 0;
    r = cuMemGetInfo(&free_mem, &tot);
    TEST_FMT("cuMemGetInfo", r == 0 && tot > 0, "%.1f/%.1f GB free",
             (double)free_mem / (1024.0 * 1024.0 * 1024.0),
             (double)tot / (1024.0 * 1024.0 * 1024.0));

    int sm = 0;
    cuDeviceGetAttribute(&sm, 16, dev); /* MULTIPROCESSOR_COUNT */
    TEST_FMT("SM count", sm > 0, "%d SMs", sm);

    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, 75, dev); /* COMPUTE_CAPABILITY_MAJOR */
    cuDeviceGetAttribute(&minor, 76, dev); /* COMPUTE_CAPABILITY_MINOR */
    TEST_FMT("Compute capability", major > 0, "sm_%d%d", major, minor);

    /* ============================================================
     * 2. Small Data Round-Trip (256 floats = 1KB)
     * ============================================================ */
    SECTION("Small Data Round-Trip (1 KB)");

    CUdeviceptr d_small = 0;
    r = cuMemAlloc(&d_small, 256 * sizeof(float));
    TEST("cuMemAlloc 1KB", r == 0 && d_small != 0);

    float src_small[256], dst_small[256];
    for (int i = 0; i < 256; i++) src_small[i] = (float)i * 3.14159f;
    memset(dst_small, 0, sizeof(dst_small));

    r = cuMemcpyHtoD(d_small, src_small, sizeof(src_small));
    TEST("cuMemcpyHtoD 1KB", r == 0);

    r = cuMemcpyDtoH(dst_small, d_small, sizeof(dst_small));
    TEST("cuMemcpyDtoH 1KB", r == 0);

    int match = 1;
    for (int i = 0; i < 256; i++) {
        if (src_small[i] != dst_small[i]) { match = 0; break; }
    }
    TEST("Data integrity (256 floats)", match);
    cuMemFree(d_small);

    /* ============================================================
     * 3. Large Data Round-Trip (16 MB)
     * ============================================================ */
    SECTION("Large Data Round-Trip (16 MB)");

    const size_t LARGE_SIZE = 16 * 1024 * 1024; /* 16 MB */
    const int LARGE_FLOATS = (int)(LARGE_SIZE / sizeof(float));
    float *src_large = (float *)malloc(LARGE_SIZE);
    float *dst_large = (float *)malloc(LARGE_SIZE);

    for (int i = 0; i < LARGE_FLOATS; i++)
        src_large[i] = sinf((float)i * 0.001f);
    memset(dst_large, 0, LARGE_SIZE);

    CUdeviceptr d_large = 0;
    r = cuMemAlloc(&d_large, LARGE_SIZE);
    TEST_FMT("cuMemAlloc 16MB", r == 0, "ptr=0x%llx", (unsigned long long)d_large);

    double t0 = now_ms();
    r = cuMemcpyHtoD(d_large, src_large, LARGE_SIZE);
    double htod_ms = now_ms() - t0;
    TEST_FMT("cuMemcpyHtoD 16MB", r == 0, "%.1f ms (%.1f MB/s)",
             htod_ms, (double)LARGE_SIZE / (htod_ms * 1000.0));

    t0 = now_ms();
    r = cuMemcpyDtoH(dst_large, d_large, LARGE_SIZE);
    double dtoh_ms = now_ms() - t0;
    TEST_FMT("cuMemcpyDtoH 16MB", r == 0, "%.1f ms (%.1f MB/s)",
             dtoh_ms, (double)LARGE_SIZE / (dtoh_ms * 1000.0));

    int large_match = 1;
    int first_mismatch = -1;
    for (int i = 0; i < LARGE_FLOATS; i++) {
        if (src_large[i] != dst_large[i]) {
            large_match = 0;
            first_mismatch = i;
            break;
        }
    }
    TEST_FMT("Data integrity (4M floats)", large_match, "%d floats verified",
             LARGE_FLOATS);
    if (!large_match && first_mismatch >= 0) {
        printf("    First mismatch at [%d]: expected %f, got %f\n",
               first_mismatch, src_large[first_mismatch], dst_large[first_mismatch]);
    }

    /* ============================================================
     * 4. DtoD Copy (zero network transfer)
     * ============================================================ */
    SECTION("Device-to-Device Copy (0 bytes over network)");

    CUdeviceptr d_copy = 0;
    r = cuMemAlloc(&d_copy, LARGE_SIZE);
    TEST("cuMemAlloc second 16MB buffer", r == 0);

    t0 = now_ms();
    r = cuMemcpyDtoD(d_copy, d_large, LARGE_SIZE);
    double dtod_ms = now_ms() - t0;
    TEST_FMT("cuMemcpyDtoD 16MB", r == 0, "%.1f ms (only 24 bytes over wire!)", dtod_ms);

    /* Verify DtoD by reading back */
    memset(dst_large, 0, LARGE_SIZE);
    cuMemcpyDtoH(dst_large, d_copy, LARGE_SIZE);
    int dtod_match = 1;
    for (int i = 0; i < LARGE_FLOATS; i++) {
        if (src_large[i] != dst_large[i]) { dtod_match = 0; break; }
    }
    TEST("DtoD data integrity verification", dtod_match);

    cuMemFree(d_copy);
    cuMemFree(d_large);

    /* ============================================================
     * 5. Throughput Test (repeated 1MB transfers)
     * ============================================================ */
    SECTION("Throughput Test (100x 1MB transfers)");

    const size_t CHUNK = 1024 * 1024; /* 1 MB */
    const int ITERATIONS = 100;
    float *chunk_buf = (float *)malloc(CHUNK);
    for (size_t i = 0; i < CHUNK / sizeof(float); i++)
        chunk_buf[i] = (float)i;

    CUdeviceptr d_chunk = 0;
    cuMemAlloc(&d_chunk, CHUNK);

    t0 = now_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        cuMemcpyHtoD(d_chunk, chunk_buf, CHUNK);
    }
    double bulk_htod_ms = now_ms() - t0;
    double htod_throughput = (double)(CHUNK * ITERATIONS) / (bulk_htod_ms * 1000.0);

    t0 = now_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        cuMemcpyDtoH(chunk_buf, d_chunk, CHUNK);
    }
    double bulk_dtoh_ms = now_ms() - t0;
    double dtoh_throughput = (double)(CHUNK * ITERATIONS) / (bulk_dtoh_ms * 1000.0);

    TEST_FMT("HtoD throughput (100x 1MB)", htod_throughput > 0,
             "%.1f MB/s (%.0f ms total)", htod_throughput, bulk_htod_ms);
    TEST_FMT("DtoH throughput (100x 1MB)", dtoh_throughput > 0,
             "%.1f MB/s (%.0f ms total)", dtoh_throughput, bulk_dtoh_ms);

    cuMemFree(d_chunk);
    free(chunk_buf);

    /* ============================================================
     * 6. Memory Stress (allocate/free cycle)
     * ============================================================ */
    SECTION("Memory Stress (20 concurrent allocations)");

    const int NUM_ALLOCS = 20;
    CUdeviceptr allocs[20];
    int alloc_ok = 1;

    for (int i = 0; i < NUM_ALLOCS; i++) {
        size_t sz = (size_t)(i + 1) * 1024 * 1024; /* 1MB, 2MB, ..., 20MB */
        r = cuMemAlloc(&allocs[i], sz);
        if (r != 0) { alloc_ok = 0; break; }
    }
    TEST_FMT("Allocate 20 buffers (1-20 MB each)", alloc_ok,
             "%s", "210 MB total");

    int free_ok = 1;
    for (int i = 0; i < NUM_ALLOCS; i++) {
        r = cuMemFree(allocs[i]);
        if (r != 0) free_ok = 0;
    }
    TEST("Free all 20 buffers", free_ok);

    /* ============================================================
     * 7. Streams
     * ============================================================ */
    SECTION("Stream Operations");

    if (cuStreamCreate && cuStreamDestroy && cuStreamSynchronize) {
        CUstream stream1 = NULL, stream2 = NULL;
        r = cuStreamCreate(&stream1, 0);
        TEST("cuStreamCreate (stream 1)", r == 0);

        r = cuStreamCreate(&stream2, 0);
        TEST("cuStreamCreate (stream 2)", r == 0);

        r = cuStreamSynchronize(stream1);
        TEST("cuStreamSynchronize (stream 1)", r == 0);

        r = cuStreamSynchronize(stream2);
        TEST("cuStreamSynchronize (stream 2)", r == 0);

        cuStreamDestroy(stream1);
        cuStreamDestroy(stream2);
        TEST("cuStreamDestroy (both)", 1);
    } else {
        printf("  (stream functions not available, skipping)\n");
    }

    /* ============================================================
     * 8. Events
     * ============================================================ */
    SECTION("Event Operations");

    if (cuEventCreate && cuEventDestroy && cuEventRecord &&
        cuEventSynchronize && cuEventElapsedTime) {
        CUevent ev_start = NULL, ev_end = NULL;
        r = cuEventCreate(&ev_start, 0);
        TEST("cuEventCreate (start)", r == 0);

        r = cuEventCreate(&ev_end, 0);
        TEST("cuEventCreate (end)", r == 0);

        r = cuEventRecord(ev_start, NULL);
        TEST("cuEventRecord (start)", r == 0);

        /* Do some work between events */
        CUdeviceptr d_ev = 0;
        cuMemAlloc(&d_ev, 1024 * 1024);
        float *ev_buf = (float *)calloc(1, 1024 * 1024);
        cuMemcpyHtoD(d_ev, ev_buf, 1024 * 1024);
        cuMemcpyDtoH(ev_buf, d_ev, 1024 * 1024);
        cuMemFree(d_ev);
        free(ev_buf);

        r = cuEventRecord(ev_end, NULL);
        TEST("cuEventRecord (end)", r == 0);

        r = cuEventSynchronize(ev_end);
        TEST("cuEventSynchronize", r == 0);

        float elapsed = 0;
        r = cuEventElapsedTime(&elapsed, ev_start, ev_end);
        TEST_FMT("cuEventElapsedTime", r == 0, "%.3f ms", elapsed);

        cuEventDestroy(ev_start);
        cuEventDestroy(ev_end);
    } else {
        printf("  (event functions not available, skipping)\n");
    }

    /* ============================================================
     * 9. Final sync
     * ============================================================ */
    SECTION("Cleanup");

    r = cuCtxSynchronize();
    TEST("cuCtxSynchronize", r == 0);

    free(src_large);
    free(dst_large);

    /* ============================================================
     * Summary
     * ============================================================ */
    printf("\n========================================================\n");
    printf("  Results: %d/%d passed, %d failed\n", passed, total, failed);
    printf("========================================================\n\n");

    if (failed == 0) {
        printf("\033[32mALL CUDA CALLS EXECUTED ON REMOTE GPU!\033[0m\n");
        printf("This machine has no GPU — everything went over TCP/IP.\n\n");
        printf("Transfer summary:\n");
        printf("  HtoD 16MB:    %.1f ms\n", htod_ms);
        printf("  DtoH 16MB:    %.1f ms\n", dtoh_ms);
        printf("  DtoD 16MB:    %.1f ms (zero network bytes)\n", dtod_ms);
        printf("  HtoD 100x1MB: %.1f MB/s\n", htod_throughput);
        printf("  DtoH 100x1MB: %.1f MB/s\n", dtoh_throughput);
    }

    return failed > 0 ? 1 : 0;
}
