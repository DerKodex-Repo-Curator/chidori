/*
 * Chidori Server-Side Call Sequence Replay
 *
 * Records one training step's CUDA dispatch sequence, then replays it
 * on subsequent steps with only the changed data (input batch). This
 * eliminates per-call RPC overhead — the server executes the full
 * recorded sequence in a tight loop with zero network round-trips
 * between CUDA calls.
 *
 * Architecture:
 *   Record: Client sends REPLAY_BEGIN_RECORD, normal RPCs flow,
 *           server snapshots each dispatch, client sends REPLAY_END_RECORD.
 *   Replay: Client sends REPLAY_EXECUTE with variable data (H2D payloads),
 *           server replays the recorded sequence, returns D2H outputs.
 *
 * Call classification:
 *   FIXED   — Entirely reproducible (memset, DtoD, event record)
 *   KERNEL  — Fixed structure, variable param buffer (cuLaunchKernel)
 *   MATMUL  — Fixed descriptors, variable data pointers (cublasLtMatmul)
 *   H2D     — Variable data payload (cuMemcpyHtoD)
 *   D2H     — Output that must be returned to client (cuMemcpyDtoH)
 *   SYNC    — Synchronization point (cudaDeviceSynchronize)
 */

#ifndef CHIDORI_REPLAY_H
#define CHIDORI_REPLAY_H

#include <stdint.h>
#include <stddef.h>

/* Call classification */
typedef enum {
    REPLAY_CALL_FIXED,       /* Entirely fixed: memset, DtoD, event record */
    REPLAY_CALL_KERNEL,      /* cuLaunchKernel: fixed structure, variable param buffer */
    REPLAY_CALL_MATMUL,      /* cublasLtMatmul: fixed descriptors, variable pointers */
    REPLAY_CALL_H2D,         /* cuMemcpyHtoD: variable data payload */
    REPLAY_CALL_D2H,         /* cuMemcpyDtoH: output to return to client */
    REPLAY_CALL_SYNC,        /* Synchronization point */
    REPLAY_CALL_SKIP,        /* Skip during replay (one-time setup like RegisterFatBinary) */
} replay_call_type_t;

/* Max variable regions per recorded call */
#define REPLAY_MAX_VAR_SLOTS 8

/* A single recorded CUDA dispatch call */
typedef struct {
    uint16_t func_id;                   /* Original function ID */
    uint16_t flags;                     /* Original flags */
    replay_call_type_t type;            /* Call classification */

    void    *payload;                   /* malloc'd snapshot of original payload */
    uint32_t payload_size;

    /* Variable regions within the payload (byte offsets where data changes per step) */
    uint16_t var_slot_count;
    struct {
        uint32_t offset;                /* Byte offset into payload */
        uint32_t length;                /* Bytes to substitute */
    } var_slots[REPLAY_MAX_VAR_SLOTS];

    /* For H2D: which variable data slot in REPLAY_EXECUTE this maps to */
    int16_t h2d_var_index;              /* -1 if not H2D */

    /* For D2H: expected output size */
    uint64_t d2h_size;

    /* Was this originally a sync call (needs response)? */
    int needs_response;
} replay_call_t;

/* Limits */
#define REPLAY_MAX_CALLS     8192       /* Max calls in one recorded sequence */
#define REPLAY_MAX_SEQUENCES 4          /* Max simultaneous recordings */
#define REPLAY_MAX_H2D_SLOTS 64         /* Max H2D variable slots per sequence */
#define REPLAY_MAX_D2H_SLOTS 16         /* Max D2H outputs per sequence */

/* A complete recorded sequence (one training step) */
typedef struct {
    int active;                         /* 1 if this slot is in use */
    uint32_t sequence_id;               /* Client-chosen ID */
    uint64_t fingerprint;               /* Hash of func_id sequence + fixed param sizes */

    replay_call_t calls[REPLAY_MAX_CALLS];
    int call_count;

    /* H2D variable data manifest: what the client must send on replay */
    int h2d_slot_count;
    struct {
        int call_index;                 /* Which call this belongs to */
        uint32_t expected_size;         /* Expected byte count for this slot */
    } h2d_manifest[REPLAY_MAX_H2D_SLOTS];

    /* D2H output manifest: what the server must return on replay */
    int d2h_slot_count;
    struct {
        int call_index;                 /* Which call this belongs to */
        uint64_t expected_size;         /* Expected byte count */
    } d2h_manifest[REPLAY_MAX_D2H_SLOTS];

    /* Stats */
    uint64_t total_fixed_bytes;
    uint64_t total_variable_bytes;
    int replay_count;
} replay_sequence_t;

/* ============================================================
 * API
 * ============================================================ */

/* Initialize the replay subsystem */
void replay_init(void);

/* Shutdown and free all recorded sequences */
void replay_shutdown(void);

/* Start recording into a sequence slot. Returns 0 on success. */
int replay_begin_record(uint32_t sequence_id);

/* Record a single dispatch call during recording mode.
 * Called from cuda_dispatch() for each RPC while recording is active.
 * Returns 0 on success, -1 if recording is full. */
int replay_record_call(uint16_t func_id, uint16_t flags,
                       const void *payload, uint32_t payload_size);

/* Finalize recording. Computes fingerprint and builds manifests.
 * Returns 0 on success. Fills out_fingerprint, out_call_count,
 * out_h2d_count, out_total_var_bytes. */
int replay_end_record(uint32_t sequence_id,
                      uint64_t *out_fingerprint,
                      int *out_call_count,
                      int *out_h2d_count,
                      uint64_t *out_total_var_bytes);

/* Execute a recorded sequence with new variable data.
 * h2d_data: array of h2d_count payloads (one per H2D slot)
 * h2d_sizes: size of each H2D payload
 * d2h_out: caller-allocated array to receive D2H outputs
 * d2h_out_sizes: max size for each D2H output
 * Returns 0 on success, cuda error code on failure. */
int replay_execute(uint32_t sequence_id, uint64_t fingerprint,
                   int h2d_count, const void **h2d_data, const uint32_t *h2d_sizes,
                   int d2h_count, void **d2h_out, uint64_t *d2h_out_sizes);

/* Discard a recorded sequence and free its memory */
void replay_invalidate(uint32_t sequence_id);

/* Query if we're currently recording */
int replay_is_recording(void);

/* Get the active recording sequence index (-1 if not recording) */
int replay_get_recording_index(void);

#endif /* CHIDORI_REPLAY_H */
