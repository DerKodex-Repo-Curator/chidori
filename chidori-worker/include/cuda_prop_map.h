/*
 * cuda_prop_map.h — Version-aware cudaDeviceProp struct mapping.
 *
 * cudaDeviceProp layout differs between CUDA major versions.
 * CUDA 11.8–12.8 share one layout (1032 bytes).
 * CUDA 13.0–13.1 share another (1008 bytes, removed clockRate, memPitch,
 * deviceOverlap, kernelExecTimeoutEnabled, computeMode, maxTexture1DLinear,
 * memoryClockRate).
 *
 * Two layout families:
 *   LAYOUT_12x (1032 bytes): cu118, cu121, cu126, cu128
 *   LAYOUT_13x (1008 bytes): cu130, cu131
 *
 * Offset tables verified by compiling against each CUDA version's headers.
 *
 * Usage: call cuda_prop_translate(server_blob, server_blob_len, caller_struct, caller_size)
 */

#ifndef CUDA_PROP_MAP_H
#define CUDA_PROP_MAP_H

#include <stdint.h>
#include <string.h>
#include <stdio.h>

/*
 * cudaDeviceProp field descriptor.
 * Each field has a name, offset within its CUDA version's struct, and size.
 * We list only the fields that exist in ALL supported CUDA versions.
 */
typedef struct {
    const char *name;
    uint16_t off_128;   /* offset in CUDA 12.8 layout */
    uint16_t off_131;   /* offset in CUDA 13.1 layout */
    uint16_t size;      /* field size in bytes */
} cuda_prop_field_t;

/*
 * Complete field table for cudaDeviceProp.
 * Only fields present in BOTH CUDA 12.8 and 13.1.
 * Fields removed in 13.1 (clockRate, memPitch, deviceOverlap, etc.)
 * are NOT listed — they only exist in 12.8.
 */
static const cuda_prop_field_t cuda_prop_fields[] = {
    /* Stable fields (same offset in both versions) */
    {"name",                    0,    0,    256},
    {"uuid",                    256,  256,  16},
    {"totalGlobalMem",          288,  288,  8},
    {"sharedMemPerBlock",       296,  296,  8},
    {"regsPerBlock",            304,  304,  4},
    {"warpSize",                308,  308,  4},
    {"maxThreadsPerBlock",      320,  320,  4},
    {"maxThreadsDim",           324,  324,  12},
    {"maxGridSize",             336,  336,  12},
    {"totalConstMem",           352,  352,  8},
    {"major",                   360,  360,  4},
    {"minor",                   364,  364,  4},
    {"textureAlignment",        368,  368,  8},
    {"texturePitchAlignment",   376,  376,  8},

    /* Fields that shift between versions */
    {"multiProcessorCount",     388,  384,  4},
    {"integrated",              396,  388,  4},
    {"canMapHostMemory",        400,  392,  4},
    {"maxTexture1D",            408,  396,  4},
    {"maxTexture1DMipmap",      412,  400,  4},
    {"maxTexture2D",            420,  404,  8},
    {"maxTexture2DMipmap",      428,  412,  8},
    {"maxTexture2DLinear",      436,  420,  12},
    {"maxTexture3D",            456,  440,  12},
    {"maxTexture3DAlt",         468,  452,  12},
    {"maxTextureCubemap",       480,  464,  4},
    {"maxTexture1DLayered",     484,  468,  8},
    {"maxTexture2DLayered",     492,  476,  12},
    {"maxTextureCubemapLayered",504,  488,  8},
    {"maxSurface1D",            512,  496,  4},
    {"maxSurface2D",            516,  500,  8},
    {"maxSurface3D",            524,  508,  12},
    {"maxSurface1DLayered",     536,  520,  8},
    {"maxSurface2DLayered",     544,  528,  12},
    {"maxSurfaceCubemap",       556,  540,  4},
    {"maxSurfaceCubemapLayered",560,  544,  8},
    {"surfaceAlignment",        568,  552,  8},
    {"concurrentKernels",       576,  560,  4},
    {"ECCEnabled",              580,  564,  4},
    {"pciBusID",                584,  568,  4},
    {"pciDeviceID",             588,  572,  4},
    {"pciDomainID",             592,  576,  4},
    {"tccDriver",               596,  580,  4},
    {"asyncEngineCount",        600,  584,  4},
    {"unifiedAddressing",       604,  588,  4},
    {"memoryBusWidth",          612,  592,  4},
    {"l2CacheSize",             616,  596,  4},
    {"maxThreadsPerMultiProcessor", 624, 604, 4},
    {"streamPrioritiesSupported",628, 608,  4},
    {"globalL1CacheSupported",  632,  612,  4},
    {"localL1CacheSupported",   636,  616,  4},
    {"sharedMemPerMultiprocessor",640, 624,  8},
    {"regsPerMultiprocessor",   648,  632,  4},
    {"managedMemory",           652,  636,  4},
    {"isMultiGpuBoard",         656,  640,  4},
    {"multiGpuBoardGroupID",    660,  644,  4},
};
#define CUDA_PROP_FIELD_COUNT (sizeof(cuda_prop_fields) / sizeof(cuda_prop_fields[0]))

/*
 * Fields that only exist in CUDA 12.8 (removed in 13.1).
 * These are at fixed offsets in the 12.8 layout.
 */
static const cuda_prop_field_t cuda_prop_128_only[] = {
    {"memPitch",                312,  0,   8},   /* size_t */
    {"clockRate",               348,  0,   4},   /* int (was removed in 13.1) */
    {"deviceOverlap",           384,  0,   4},   /* int (deprecated, same as concurrentKernels) */
    {"kernelExecTimeoutEnabled",392,  0,   4},   /* int */
    {"computeMode",             404,  0,   4},   /* int */
    {"maxTexture1DLinear",      416,  0,   4},   /* int */
    {"memoryClockRate",         608,  0,   4},   /* int */
};

/*
 * Map struct size → layout family.
 *   1032 → LAYOUT_12x (cu118, cu121, cu126, cu128)
 *   1008 → LAYOUT_13x (cu130, cu131)
 *   0    → unknown (fallback to direct copy)
 *
 * All CUDA 11.x–12.x versions have the same field set and offsets.
 * CUDA 13.x removed several deprecated fields, shrinking the struct by 24 bytes.
 */
static int cuda_version_from_size(size_t struct_size) {
    if (struct_size == 1032) return 1280;  /* LAYOUT_12x: cu118, cu121, cu126, cu128 */
    if (struct_size == 1008) return 1310;  /* LAYOUT_13x: cu130, cu131 */
    return 0;
}

/*
 * Get the offset for a field in a given CUDA layout family.
 * Returns the offset, or -1 if the field doesn't exist in that version.
 */
static int cuda_prop_get_offset(int cuda_version, const cuda_prop_field_t *field) {
    if (cuda_version == 1280) return field->off_128;  /* LAYOUT_12x */
    if (cuda_version == 1310) return field->off_131;  /* LAYOUT_13x */
    return -1;
}

/*
 * Translate a cudaDeviceProp blob from server layout to caller layout.
 *
 * server_blob:     raw bytes from server (server's native layout)
 * server_blob_len: size of server blob (= server's sizeof(cudaDeviceProp))
 * caller_struct:   output buffer (caller's native layout, must be zeroed first)
 * caller_size:     sizeof(caller's cudaDeviceProp)
 *
 * Returns 0 on success.
 */
static int cuda_prop_translate(
    const void *server_blob, size_t server_blob_len,
    void *caller_struct, size_t caller_size)
{
    int server_ver = cuda_version_from_size(server_blob_len);
    int caller_ver = cuda_version_from_size(caller_size);

    if (server_ver == 0 || caller_ver == 0) {
        /* Unknown version — fall back to direct copy (best effort) */
        size_t copy = server_blob_len < caller_size ? server_blob_len : caller_size;
        memcpy(caller_struct, server_blob, copy);
        return -1;
    }

    /* Zero the output */
    memset(caller_struct, 0, caller_size);

    /* Copy each field from server layout to caller layout */
    for (size_t i = 0; i < CUDA_PROP_FIELD_COUNT; i++) {
        int src_off = cuda_prop_get_offset(server_ver, &cuda_prop_fields[i]);
        int dst_off = cuda_prop_get_offset(caller_ver, &cuda_prop_fields[i]);

        if (src_off < 0 || dst_off < 0) continue;
        if (src_off + cuda_prop_fields[i].size > server_blob_len) continue;
        if (dst_off + cuda_prop_fields[i].size > caller_size) continue;

        memcpy((char *)caller_struct + dst_off,
               (const char *)server_blob + src_off,
               cuda_prop_fields[i].size);
    }

    /* If caller is 12.8 and server is 13.1, fill 12.8-only fields from the
     * 13.1 common fields (best effort mapping).
     * If caller is 13.1 and server is 12.8, the 12.8-only fields are simply
     * dropped (they don't exist in 13.1). */
    if (caller_ver == 1280 && server_ver == 1310) {
        /* memPitch: not available from 13.1, set a reasonable default */
        size_t mempitch = 2147483647U; /* 2GB - standard max */
        memcpy((char *)caller_struct + 312, &mempitch, 8);

        /* clockRate: not in 13.1 struct, but we can extract from
         * the 12.8 layout that the server sends if server IS 12.8 */
        /* If server is 13.1, these fields are genuinely unavailable.
         * Set defaults. */
        int clock_rate = 1410000; /* 1410 MHz typical for A100 */
        memcpy((char *)caller_struct + 348, &clock_rate, 4);

        int device_overlap = 1;
        memcpy((char *)caller_struct + 384, &device_overlap, 4);

        /* kernelExecTimeoutEnabled, computeMode, maxTexture1DLinear, memoryClockRate
         * are already zero from memset — acceptable defaults */
    }

    return 0;
}

#endif /* CUDA_PROP_MAP_H */
