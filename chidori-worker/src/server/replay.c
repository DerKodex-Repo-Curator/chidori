/*
 * Chidori Server-Side Call Sequence Replay
 *
 * See replay.h for architecture overview.
 */

#include "replay.h"
#include "chidori_protocol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * State
 * ============================================================ */

static replay_sequence_t g_sequences[REPLAY_MAX_SEQUENCES];
static int g_recording = 0;             /* 1 if currently recording */
static int g_recording_seq_idx = -1;    /* Index into g_sequences */

/* ============================================================
 * FNV-1a 64-bit hash for fingerprinting
 * ============================================================ */

#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME        0x100000001b3ULL

static uint64_t fnv1a_init(void) { return FNV_OFFSET_BASIS; }

static uint64_t fnv1a_update(uint64_t hash, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)p[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

/* ============================================================
 * Call classification
 * ============================================================ */

static replay_call_type_t classify_call(uint16_t func_id) {
    switch (func_id) {
    /* H2D transfers — data changes every step */
    case CHIDORI_CU_MEMCPY_H_TO_D:
    case CHIDORI_CUDA_MEMCPY:       /* when kind=HtoD */
    case CHIDORI_CUDA_MEMCPY_ASYNC: /* when kind=HtoD */
        return REPLAY_CALL_H2D;

    /* D2H transfers — output that must be returned */
    case CHIDORI_CU_MEMCPY_D_TO_H:
        return REPLAY_CALL_D2H;

    /* Kernel launches — fixed structure, variable param buffer */
    case CHIDORI_CU_LAUNCH_KERNEL:
    case CHIDORI_CUDA_LAUNCH_KERNEL:
        return REPLAY_CALL_KERNEL;

    /* cublasLt matmul — fixed descriptors, variable data pointers */
    case CHIDORI_CUBLAS_LT_MATMUL:
        return REPLAY_CALL_MATMUL;

    /* Synchronization — must still execute during replay */
    case CHIDORI_CU_CTX_SYNCHRONIZE:
    case CHIDORI_CU_STREAM_SYNCHRONIZE:
    case CHIDORI_CUDA_DEVICE_SYNCHRONIZE:
    case CHIDORI_CUDA_STREAM_SYNCHRONIZE:
    case CHIDORI_CU_EVENT_SYNCHRONIZE:
    case CHIDORI_CUDA_EVENT_SYNCHRONIZE:
        return REPLAY_CALL_SYNC;

    /* One-time setup — skip during replay */
    case CHIDORI_CUDA_REGISTER_FAT_BINARY:
    case CHIDORI_CUDA_REGISTER_FUNCTION:
    case CHIDORI_CU_MODULE_LOAD_DATA:
    case CHIDORI_CU_MODULE_GET_FUNCTION:
    case CHIDORI_CU_MEM_ALLOC:
    case CHIDORI_CUDA_MALLOC:
    case CHIDORI_CU_MEM_FREE:
    case CHIDORI_CUDA_FREE:
    case CHIDORI_CUBLAS_CREATE:
    case CHIDORI_CUBLAS_DESTROY:
    case CHIDORI_CUBLAS_LT_MATMUL_ALGO_HEURISTIC:
        return REPLAY_CALL_SKIP;

    /* Everything else — fixed, replay as-is */
    default:
        return REPLAY_CALL_FIXED;
    }
}

/* Identify variable byte regions within a payload */
static void identify_var_slots(replay_call_t *call) {
    call->var_slot_count = 0;
    call->h2d_var_index = -1;
    call->d2h_size = 0;

    switch (call->type) {
    case REPLAY_CALL_H2D:
        /* H2D: bytes [16..end] are the data payload (after dst(8)+size(8) header) */
        if (call->payload_size > 16) {
            call->var_slots[0].offset = 16;
            call->var_slots[0].length = call->payload_size - 16;
            call->var_slot_count = 1;
        }
        break;

    case REPLAY_CALL_D2H:
        /* D2H: payload is dst(8)+size(8). Size tells us how much output to expect. */
        if (call->payload_size >= 16) {
            uint64_t size = 0;
            memcpy(&size, (char *)call->payload + 8, 8);
            call->d2h_size = size;
        }
        break;

    case REPLAY_CALL_KERNEL:
        /* Kernel launch: param buffer starts after the 44-byte fixed header + 1 byte has_params.
         * If has_params=1, next 4 bytes are buf_size, then buf_size bytes of params.
         * The param buffer contains GPU pointers that MAY change (but usually don't
         * due to PyTorch caching allocator). We mark the entire param buffer as variable
         * to be safe. */
        if (call->payload_size > 45) {
            uint8_t has_params = 0;
            memcpy(&has_params, (char *)call->payload + 44, 1);
            if (has_params) {
                uint32_t buf_size = 0;
                memcpy(&buf_size, (char *)call->payload + 45, 4);
                if (buf_size > 0 && 49 + buf_size <= call->payload_size) {
                    call->var_slots[0].offset = 49;
                    call->var_slots[0].length = buf_size;
                    call->var_slot_count = 1;
                }
            }
        }
        break;

    case REPLAY_CALL_MATMUL:
        /* cublasLtMatmul: 308-byte payload.
         * Bytes [0..187] = descriptors (FIXED)
         * Bytes [188..195] = alpha scalar (usually fixed but mark variable)
         * Bytes [196..203] = beta scalar (usually fixed)
         * Bytes [204..211] = A ptr (VARIABLE)
         * Bytes [212..219] = B ptr (VARIABLE)
         * Bytes [220..227] = C ptr (VARIABLE)
         * Bytes [228..235] = D ptr (VARIABLE)
         * Bytes [236..243] = stream handle
         * Bytes [244..307] = algorithm (FIXED) */
        if (call->payload_size >= 236) {
            /* Mark data pointers A,B,C,D as variable (bytes 204-235) */
            call->var_slots[0].offset = 204;
            call->var_slots[0].length = 32;  /* 4 pointers × 8 bytes */
            call->var_slot_count = 1;
        }
        break;

    default:
        break;
    }
}

/* ============================================================
 * API Implementation
 * ============================================================ */

void replay_init(void) {
    memset(g_sequences, 0, sizeof(g_sequences));
    g_recording = 0;
    g_recording_seq_idx = -1;
}

void replay_shutdown(void) {
    for (int i = 0; i < REPLAY_MAX_SEQUENCES; i++) {
        replay_invalidate(i);
    }
    g_recording = 0;
    g_recording_seq_idx = -1;
}

int replay_begin_record(uint32_t sequence_id) {
    if (g_recording) {
        fprintf(stderr, "[replay] Already recording sequence %u\n",
                g_sequences[g_recording_seq_idx].sequence_id);
        return -1;
    }

    /* Find a free slot or reuse the one with this sequence_id */
    int idx = -1;
    for (int i = 0; i < REPLAY_MAX_SEQUENCES; i++) {
        if (g_sequences[i].active && g_sequences[i].sequence_id == sequence_id) {
            /* Reuse — invalidate old recording first */
            replay_invalidate(sequence_id);
            idx = i;
            break;
        }
        if (!g_sequences[i].active && idx < 0) {
            idx = i;
        }
    }
    if (idx < 0) {
        fprintf(stderr, "[replay] No free sequence slots\n");
        return -1;
    }

    memset(&g_sequences[idx], 0, sizeof(replay_sequence_t));
    g_sequences[idx].active = 1;
    g_sequences[idx].sequence_id = sequence_id;
    g_recording = 1;
    g_recording_seq_idx = idx;

    fprintf(stderr, "[replay] Recording started: seq_id=%u slot=%d\n", sequence_id, idx);
    return 0;
}

int replay_record_call(uint16_t func_id, uint16_t flags,
                       const void *payload, uint32_t payload_size) {
    if (!g_recording || g_recording_seq_idx < 0)
        return -1;

    replay_sequence_t *seq = &g_sequences[g_recording_seq_idx];
    if (seq->call_count >= REPLAY_MAX_CALLS) {
        fprintf(stderr, "[replay] Recording overflow: %d calls max\n", REPLAY_MAX_CALLS);
        return -1;
    }

    replay_call_t *call = &seq->calls[seq->call_count];
    call->func_id = func_id;
    call->flags = flags;
    call->type = classify_call(func_id);
    call->needs_response = !(flags & CHIDORI_FLAG_ASYNC);

    /* Snapshot payload */
    if (payload_size > 0 && payload) {
        call->payload = malloc(payload_size);
        if (!call->payload) return -1;
        memcpy(call->payload, payload, payload_size);
    } else {
        call->payload = NULL;
    }
    call->payload_size = payload_size;

    /* Identify variable regions */
    identify_var_slots(call);

    seq->call_count++;
    return 0;
}

int replay_end_record(uint32_t sequence_id,
                      uint64_t *out_fingerprint,
                      int *out_call_count,
                      int *out_h2d_count,
                      uint64_t *out_total_var_bytes) {
    if (!g_recording || g_recording_seq_idx < 0)
        return -1;

    replay_sequence_t *seq = &g_sequences[g_recording_seq_idx];
    if (seq->sequence_id != sequence_id)
        return -1;

    /* Compute fingerprint: hash of func_ids + fixed payload portions */
    uint64_t fp = fnv1a_init();
    uint64_t total_fixed = 0;
    uint64_t total_var = 0;

    for (int i = 0; i < seq->call_count; i++) {
        replay_call_t *c = &seq->calls[i];

        /* Hash func_id */
        fp = fnv1a_update(fp, &c->func_id, sizeof(c->func_id));

        /* Hash fixed portions of payload */
        if (c->payload && c->payload_size > 0) {
            if (c->var_slot_count == 0) {
                /* Entirely fixed — hash whole payload */
                fp = fnv1a_update(fp, c->payload, c->payload_size);
                total_fixed += c->payload_size;
            } else {
                /* Hash non-variable portions */
                uint32_t prev_end = 0;
                for (int j = 0; j < c->var_slot_count; j++) {
                    uint32_t gap = c->var_slots[j].offset - prev_end;
                    if (gap > 0) {
                        fp = fnv1a_update(fp, (char *)c->payload + prev_end, gap);
                        total_fixed += gap;
                    }
                    total_var += c->var_slots[j].length;
                    prev_end = c->var_slots[j].offset + c->var_slots[j].length;
                }
                /* Hash trailing fixed portion */
                if (prev_end < c->payload_size) {
                    fp = fnv1a_update(fp, (char *)c->payload + prev_end,
                                      c->payload_size - prev_end);
                    total_fixed += c->payload_size - prev_end;
                }
            }
        }

        /* Build H2D manifest */
        if (c->type == REPLAY_CALL_H2D && seq->h2d_slot_count < REPLAY_MAX_H2D_SLOTS) {
            c->h2d_var_index = seq->h2d_slot_count;
            seq->h2d_manifest[seq->h2d_slot_count].call_index = i;
            seq->h2d_manifest[seq->h2d_slot_count].expected_size =
                (c->var_slot_count > 0) ? c->var_slots[0].length : 0;
            seq->h2d_slot_count++;
        }

        /* Build D2H manifest */
        if (c->type == REPLAY_CALL_D2H && seq->d2h_slot_count < REPLAY_MAX_D2H_SLOTS) {
            seq->d2h_manifest[seq->d2h_slot_count].call_index = i;
            seq->d2h_manifest[seq->d2h_slot_count].expected_size = c->d2h_size;
            seq->d2h_slot_count++;
        }
    }

    seq->fingerprint = fp;
    seq->total_fixed_bytes = total_fixed;
    seq->total_variable_bytes = total_var;

    /* Stop recording */
    g_recording = 0;
    g_recording_seq_idx = -1;

    /* Output */
    if (out_fingerprint) *out_fingerprint = fp;
    if (out_call_count) *out_call_count = seq->call_count;
    if (out_h2d_count) *out_h2d_count = seq->h2d_slot_count;
    if (out_total_var_bytes) *out_total_var_bytes = total_var;

    fprintf(stderr, "[replay] Recording finalized: seq_id=%u calls=%d h2d=%d d2h=%d "
            "fixed=%llu var=%llu fp=0x%016llx\n",
            sequence_id, seq->call_count, seq->h2d_slot_count, seq->d2h_slot_count,
            (unsigned long long)total_fixed, (unsigned long long)total_var,
            (unsigned long long)fp);

    return 0;
}

int replay_execute(uint32_t sequence_id, uint64_t fingerprint,
                   int h2d_count, const void **h2d_data, const uint32_t *h2d_sizes,
                   int d2h_count, void **d2h_out, uint64_t *d2h_out_sizes) {
    /* Find the sequence */
    replay_sequence_t *seq = NULL;
    for (int i = 0; i < REPLAY_MAX_SEQUENCES; i++) {
        if (g_sequences[i].active && g_sequences[i].sequence_id == sequence_id) {
            seq = &g_sequences[i];
            break;
        }
    }
    if (!seq) {
        fprintf(stderr, "[replay] Execute: sequence %u not found\n", sequence_id);
        return -1;
    }

    /* Validate fingerprint */
    if (seq->fingerprint != fingerprint) {
        fprintf(stderr, "[replay] Execute: fingerprint mismatch (expected 0x%016llx, got 0x%016llx)\n",
                (unsigned long long)seq->fingerprint, (unsigned long long)fingerprint);
        return -2;
    }

    /* Validate H2D slot count */
    if (h2d_count != seq->h2d_slot_count) {
        fprintf(stderr, "[replay] Execute: H2D count mismatch (expected %d, got %d)\n",
                seq->h2d_slot_count, h2d_count);
        return -3;
    }

    /* Substitute variable data into recorded payloads */
    for (int i = 0; i < seq->h2d_slot_count; i++) {
        int call_idx = seq->h2d_manifest[i].call_index;
        replay_call_t *c = &seq->calls[call_idx];
        if (c->var_slot_count > 0 && h2d_data[i] && h2d_sizes[i] > 0) {
            /* Replace the variable portion of the payload */
            uint32_t off = c->var_slots[0].offset;
            uint32_t len = h2d_sizes[i];
            if (len > c->var_slots[0].length) len = c->var_slots[0].length;
            memcpy((char *)c->payload + off, h2d_data[i], len);
        }
    }

    /* Replay loop — dispatch each recorded call */
    int d2h_idx = 0;
    extern int cuda_dispatch(int fd, chidori_request_t *req, void *payload);

    for (int i = 0; i < seq->call_count; i++) {
        replay_call_t *c = &seq->calls[i];

        /* Skip one-time setup calls */
        if (c->type == REPLAY_CALL_SKIP)
            continue;

        /* Build a fake request header */
        chidori_request_t fake_req = {0};
        fake_req.func_id = c->func_id;
        fake_req.flags = c->flags | CHIDORI_FLAG_ASYNC; /* suppress individual responses */
        fake_req.payload_size = c->payload_size;

        /* For D2H calls, we need to capture the output */
        int dispatch_fd = -1; /* -1 = no response (async) */
        if (c->type == REPLAY_CALL_D2H) {
            /* D2H needs special handling — we need the output data */
            /* For now, dispatch as sync but capture output differently */
            /* TODO: Implement proper D2H output capture */
            fake_req.flags = c->flags; /* keep original flags */
        }

        /* Dispatch */
        cuda_dispatch(dispatch_fd, &fake_req, c->payload);
    }

    seq->replay_count++;

    if (seq->replay_count % 100 == 0 || seq->replay_count == 1) {
        fprintf(stderr, "[replay] Execute #%d: seq_id=%u, %d calls replayed\n",
                seq->replay_count, sequence_id, seq->call_count);
    }

    return 0;
}

void replay_invalidate(uint32_t sequence_id) {
    for (int i = 0; i < REPLAY_MAX_SEQUENCES; i++) {
        if (g_sequences[i].active && g_sequences[i].sequence_id == sequence_id) {
            /* Free all payload snapshots */
            for (int j = 0; j < g_sequences[i].call_count; j++) {
                free(g_sequences[i].calls[j].payload);
            }
            memset(&g_sequences[i], 0, sizeof(replay_sequence_t));
            fprintf(stderr, "[replay] Invalidated sequence %u\n", sequence_id);
            return;
        }
    }
}

int replay_is_recording(void) {
    return g_recording;
}

int replay_get_recording_index(void) {
    return g_recording_seq_idx;
}
