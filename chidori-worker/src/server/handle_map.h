/*
 * Chidori GPU Worker — Server-side handle mapping
 *
 * Maps wire protocol handles (uint64_t) to real CUDA handles.
 * Uses simple arrays since the number of concurrent handles is small.
 */
#ifndef CHIDORI_HANDLE_MAP_H
#define CHIDORI_HANDLE_MAP_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define HANDLE_MAP_MAX 1024

typedef struct {
  uint64_t keys[HANDLE_MAP_MAX];
  uint64_t vals[HANDLE_MAP_MAX];
  int count;
} handle_map_t;

static inline void hmap_init(handle_map_t *m) { memset(m, 0, sizeof(*m)); }

static inline int hmap_put(handle_map_t *m, uint64_t key, uint64_t val) {
  if (m->count >= HANDLE_MAP_MAX)
    return -1;
  m->keys[m->count] = key;
  m->vals[m->count] = val;
  m->count++;
  return 0;
}

static inline uint64_t hmap_get(handle_map_t *m, uint64_t key) {
  for (int i = 0; i < m->count; i++) {
    if (m->keys[i] == key)
      return m->vals[i];
  }
  return 0;
}

static inline void hmap_remove(handle_map_t *m, uint64_t key) {
  for (int i = 0; i < m->count; i++) {
    if (m->keys[i] == key) {
      m->keys[i] = m->keys[m->count - 1];
      m->vals[i] = m->vals[m->count - 1];
      m->count--;
      return;
    }
  }
}

#endif /* CHIDORI_HANDLE_MAP_H */
