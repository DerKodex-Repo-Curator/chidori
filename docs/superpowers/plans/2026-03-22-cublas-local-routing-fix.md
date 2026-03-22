# cuBLAS Local Routing Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix cuBLAS (error 14) and kernel launch (FPE) on local GPU during partial offload.

**Architecture:** cuBLAS resolves CUDA symbols via `dlopen("libcuda.so.1")` + `dlsym()` at runtime (NOT via PLT linking — confirmed by `readelf -d`). Currently, `LD_LIBRARY_PATH=build` causes cuBLAS to find our `build/libcuda.so.1` symlink (SONAME `libchidori.so`), which resolves to our interceptor library. cuBLAS caches our intercepted function pointers, causing reentrance failures. The fix: remove `build/` from `LD_LIBRARY_PATH` so cuBLAS finds the real `/usr/lib/libcuda.so.1` (SONAME `libcuda.so.1`). cuBLAS then gets real driver function pointers via `dlsym(real_handle, ...)`, completely bypassing our interception layer. No DEEPBIND needed, no dlmopen needed, no reentrance.

**Tech Stack:** C (gcc), LD_PRELOAD, dlopen/dlsym, CUDA Driver/Runtime API

**Key Evidence:**
- `readelf -d /opt/cuda/lib64/libcublas.so` shows NO DT_NEEDED for libcuda.so.1 or libcudart.so
- cuBLAS imports: `dlopen@GLIBC_2.2.5`, `dlsym@GLIBC_2.2.5` (runtime resolution)
- Our library SONAME is already `libchidori.so` (commit 6e59bd2) — no SONAME collision
- Real driver at `/usr/lib/libcuda.so.1` has SONAME `libcuda.so.1`
- System CUDA: 13.1 | PyTorch: cu128 (12.8) — version mismatch may cause separate FPE issue

---

### Task 1: Verify Theory — Quick Smoke Test with Modified Environment

**Files:**
- No code changes — environment variable test only

- [ ] **Step 1: SSH to Lenovo, kill old server, start fresh**

```bash
ssh fabricio@lenovo.local 'cd ~/chidori/chidori-worker && \
  kill $(pgrep chidori-worker) 2>/dev/null; sleep 1; \
  nohup build/chidori-worker --standalone 43211 > /tmp/server.log 2>&1 & sleep 1'
```

- [ ] **Step 2: Run partial offload test WITHOUT build/ in LD_LIBRARY_PATH**

```bash
ssh fabricio@lenovo.local 'cd ~/chidori/chidori-worker && \
  CHIDORI_LOCAL_DEVICES=0 \
  CHIDORI_DIRECT_CONNECT=1 \
  CHIDORI_CFG_OVERRIDE='\''{"id":"test","servers":["127.0.0.1:43211"]}'\'' \
  LD_PRELOAD="build/libchidori.so build/libnvidia-ml.so.1" \
  CHIDORI_DEBUG=1 \
  ~/chidori/.venv/bin/python -c "
import torch; torch._C._cuda_init()
print(\"Devices:\", torch.cuda.device_count())
x0 = torch.tensor([1.0, 2.0, 3.0]).cuda(0)
print(\"cuda:0 tensor:\", x0.cpu().tolist())
x1 = torch.tensor([4.0, 5.0, 6.0]).cuda(1)
print(\"cuda:1 tensor:\", x1.cpu().tolist())
a = torch.randn(256, 256)
b = torch.randn(256, 256)
cpu_ref = torch.mm(a, b)
gpu0 = torch.mm(a.cuda(0), b.cuda(0)).cpu()
err0 = (gpu0-cpu_ref).abs().max().item()
print(\"cuda:0 matmul err=%f\" % err0)
gpu1 = torch.mm(a.cuda(1), b.cuda(1)).cpu()
err1 = (gpu1-cpu_ref).abs().max().item()
print(\"cuda:1 matmul err=%f\" % err1)
print(\"=== PARTIAL OFFLOAD SMOKE TEST PASSED ===\")
" 2>&1' | tee .claude/test_output.txt
```

Expected outcomes:
- `Devices: 2` — device enumeration still works
- `cuda:0 tensor:` / `cuda:1 tensor:` — tensor transfers work
- `cuda:0 matmul err < 0.01` — cuBLAS on local GPU works (the main fix)
- `cuda:1 matmul err < 0.01` — cuBLAS on remote GPU works (regression check)

If **cuda:0 matmul works**: Theory confirmed. Proceed to Task 2.
If **fails with "cannot find libcuda.so.1"**: PyTorch can't resolve libcuda dep. Need to adjust — either pre-load real driver or add system paths.
If **cuBLAS still fails error 14**: cuBLAS is resolving libcuda differently than expected. Add `LD_DEBUG=libs` trace to see what dlopen finds.
If **FPE on torch.randn**: Separate issue (CUDA 12.8 vs 13.1 mismatch). cuBLAS fix is still valid, kernel launch needs separate fix (Task 5).

- [ ] **Step 3: If smoke test fails, add LD_DEBUG trace**

```bash
# Same command but add LD_DEBUG=libs to see dynamic linker decisions
LD_DEBUG=libs <same command> 2>&1 | grep -E "libcuda|libcublas" | head -50
```

This shows exactly which libcuda.so.1 cuBLAS's dlopen finds.

---

### Task 2: Simplify cuBLAS Loading Code

**Files:**
- Modify: `chidori-worker/src/client/chidori_client.c:329-377` (ensure_real_cublas)
- Modify: `chidori-worker/src/client/chidori_client.c:6222-6251` (ensure_real_cublaslt)
- Modify: `chidori-worker/src/client/chidori_client.c:2183-2206` (ensure_local_cublas_handle)

- [ ] **Step 1: Simplify ensure_real_cublas() — remove DEEPBIND and LD_LIBRARY_PATH hack**

Replace lines 329-377 with:

```c
static inline void *ensure_real_cublas(void) {
    if (g_real_cublas_handle) return g_real_cublas_handle;
    if (g_cublas_load_attempted) return NULL;
    g_cublas_load_attempted = 1;

    /* cuBLAS resolves CUDA symbols via dlopen("libcuda.so.1") + dlsym() internally.
     * Since our SONAME is libchidori.so (not libcuda.so.1), cuBLAS's dlopen finds
     * the real NVIDIA driver as long as LD_LIBRARY_PATH doesn't contain our symlink.
     * No DEEPBIND needed — cuBLAS uses dlsym on the real driver handle directly. */

    /* Try already-loaded cuBLAS first (PyTorch usually loads it) */
    g_real_cublas_handle = dlopen("libcublas.so.12", RTLD_LAZY | RTLD_NOLOAD);
    if (!g_real_cublas_handle)
        g_real_cublas_handle = dlopen("libcublas.so", RTLD_LAZY | RTLD_NOLOAD);

    /* If not loaded yet, try system paths */
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

    if (g_real_cublas_handle)
        LOG_DEBUG("Loaded real cuBLAS: %p", g_real_cublas_handle);
    else
        LOG_DEBUG("WARNING: Could not load real cuBLAS — local device cuBLAS calls will fall through to RPC");
    return g_real_cublas_handle;
}
```

- [ ] **Step 2: Simplify ensure_real_cublaslt() — same approach**

Replace the partial offload branch in ensure_real_cublaslt() (lines ~6222-6251) with the same pattern: try RTLD_NOLOAD first, then system paths without DEEPBIND.

- [ ] **Step 3: Simplify ensure_local_cublas_handle() — remove explicit cudaSetDevice call**

The current code calls `cudaSetDevice(0)` + `cudaDeviceSynchronize()` before cublasCreate. This calls our interceptor which sets up the real driver context. This is correct — keep the context setup but remove the `extern` declarations and add a comment explaining why:

```c
static cublasHandle_t ensure_local_cublas_handle(void) {
    if (g_local_cublas_handle) return g_local_cublas_handle;
    void *h = ensure_real_cublas();
    if (!h) return NULL;
    typedef cublasStatus_t (*create_fn)(cublasHandle_t *);
    create_fn real_create = (create_fn)dlsym(h, "cublasCreate_v2");
    if (!real_create) return NULL;

    /* Ensure real CUDA context is current for the local device.
     * Our cudaSetDevice(0) creates a real driver context via
     * cuDevicePrimaryCtxRetain + cuCtxSetCurrent on the dlopen'd driver.
     * cuBLAS's cublasCreate internally calls cuCtxGetCurrent via the
     * real driver (resolved by its own dlopen of libcuda.so.1), so it
     * sees the same context we set up here. */
    extern int cudaSetDevice(int);
    cudaSetDevice(0);

    cublasStatus_t r = real_create(&g_local_cublas_handle);
    if (r != CUBLAS_STATUS_SUCCESS) {
        LOG_DEBUG("ensure_local_cublas_handle: cublasCreate_v2 failed: %d", r);
        g_local_cublas_handle = NULL;
        return NULL;
    }
    LOG_DEBUG("Created local cuBLAS handle: %p", (void*)g_local_cublas_handle);
    return g_local_cublas_handle;
}
```

- [ ] **Step 4: Build**

```bash
ssh fabricio@lenovo.local 'cd ~/chidori/chidori-worker && make clean && make all CUDA_PATH=/opt/cuda && cd build && ln -sf libchidori.so libcuda.so.1'
```

- [ ] **Step 5: Re-run smoke test from Task 1 Step 2**

- [ ] **Step 6: Commit**

```bash
git add chidori-worker/src/client/chidori_client.c
git commit -m "fix: remove DEEPBIND from cuBLAS loading — cuBLAS uses dlsym not PLT

cuBLAS resolves CUDA symbols via dlopen('libcuda.so.1') + dlsym(), not
via PLT linking (confirmed by readelf -d showing no DT_NEEDED for libcuda).
DEEPBIND only affects PLT resolution, so it was never going to work.

The real fix: ensure LD_LIBRARY_PATH doesn't contain our build/ dir
(which has a libcuda.so.1 symlink to our interceptor). Without that
symlink in the search path, cuBLAS's dlopen finds the real NVIDIA driver
at /usr/lib/libcuda.so.1 (SONAME libcuda.so.1, different from our
libchidori.so). cuBLAS then uses real driver functions via dlsym on
that handle, completely bypassing our LD_PRELOAD interception."
```

---

### Task 3: Add Local Device Routing to Module Functions

cuBLAS loads its internal GEMM kernels via `cuModuleLoadData` / `cuModuleGetFunction`. These currently only have a reentrance guard (`g_in_real_cuda > 0`) but NO `is_local_device()` check. If cuBLAS uses the real driver (after Task 2 fix), these calls bypass our interceptor entirely. But if any other code path hits them for local device, they need routing.

**Files:**
- Modify: `chidori-worker/src/client/chidori_client.c:1191-1234`

- [ ] **Step 1: Add is_local_device routing to cuModuleLoadData**

```c
CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (g_in_real_cuda > 0 && g_real.cuModuleLoadData) return g_real.cuModuleLoadData(module, image);
    /* Local device: forward to real driver */
    if (g_partial_offload && is_local_device(g_current_device) && g_real.cuModuleLoadData) {
        g_in_real_cuda++;
        CUresult r = g_real.cuModuleLoadData(module, image);
        g_in_real_cuda--;
        return r;
    }
    /* Remote: send via RPC ... (existing code) */
```

- [ ] **Step 2: Same for cuModuleUnload and cuModuleGetFunction**

Add `is_local_device()` routing before the RPC path.

- [ ] **Step 3: Build and test**

```bash
ssh fabricio@lenovo.local 'cd ~/chidori/chidori-worker && make all CUDA_PATH=/opt/cuda'
```

Re-run smoke test.

- [ ] **Step 4: Commit**

```bash
git add chidori-worker/src/client/chidori_client.c
git commit -m "fix: add is_local_device routing to cuModuleLoadData/GetFunction/Unload"
```

---

### Task 4: Update Test Environment for Partial Offload

**Files:**
- Modify: `chidori-worker/CLAUDE.md` (test commands)
- Modify: test scripts if any hardcode LD_LIBRARY_PATH

- [ ] **Step 1: Define canonical partial offload environment**

The correct environment for partial offload (client HAS a local GPU):
```bash
CHIDORI_LOCAL_DEVICES=0
CHIDORI_DIRECT_CONNECT=1
CHIDORI_CFG_OVERRIDE='{"id":"test","servers":["<server>:<port>"]}'
LD_PRELOAD="build/libchidori.so build/libnvidia-ml.so.1"
# NO LD_LIBRARY_PATH=build — this is critical
```

The correct environment for remote-only (client has NO GPU):
```bash
CHIDORI_DIRECT_CONNECT=1
CHIDORI_CFG_OVERRIDE='{"id":"test","servers":["<server>:<port>"]}'
LD_PRELOAD=build/libcuda.so.1
LD_LIBRARY_PATH=build
```

- [ ] **Step 2: Run full partial offload test**

```bash
ssh fabricio@lenovo.local 'cd ~/chidori/chidori-worker && \
  kill $(pgrep chidori-worker) 2>/dev/null; sleep 1; \
  nohup build/chidori-worker --standalone 43211 > /tmp/server.log 2>&1 & sleep 1; \
  CHIDORI_LOCAL_DEVICES=0 \
  CHIDORI_DIRECT_CONNECT=1 \
  CHIDORI_CFG_OVERRIDE='\''{"id":"test","servers":["127.0.0.1:43211"]}'\'' \
  LD_PRELOAD="build/libchidori.so build/libnvidia-ml.so.1" \
  CHIDORI_DEBUG=1 \
  ~/chidori/.venv/bin/python -c "
import torch; torch._C._cuda_init()
print(\"Devices:\", torch.cuda.device_count())

# Tensor transfer
x0 = torch.tensor([1.0, 2.0, 3.0]).cuda(0)
print(\"cuda:0:\", x0.cpu().tolist())
x1 = torch.tensor([4.0, 5.0, 6.0]).cuda(1)
print(\"cuda:1:\", x1.cpu().tolist())

# Sync
torch.cuda.synchronize(0); print(\"cuda:0 sync OK\")
torch.cuda.synchronize(1); print(\"cuda:1 sync OK\")

# Matmul on remote (should still work via RPC)
a = torch.randn(256, 256)
b = torch.randn(256, 256)
cpu_ref = torch.mm(a, b)
gpu1 = torch.mm(a.cuda(1), b.cuda(1)).cpu()
err1 = (gpu1-cpu_ref).abs().max().item()
print(\"cuda:1 matmul err=%f\" % err1)
assert err1 < 0.01, \"Remote matmul FAILED\"

# Matmul on local (the main test)
gpu0 = torch.mm(a.cuda(0), b.cuda(0)).cpu()
err0 = (gpu0-cpu_ref).abs().max().item()
print(\"cuda:0 matmul err=%f\" % err0)
assert err0 < 0.01, \"Local matmul FAILED\"

# Randn on both (kernel launch test)
r1 = torch.randn(4, device=\"cuda:1\")
print(\"cuda:1 randn:\", r1.cpu().tolist())
r0 = torch.randn(4, device=\"cuda:0\")
print(\"cuda:0 randn:\", r0.cpu().tolist())

print(\"=== ALL PARTIAL OFFLOAD TESTS PASSED ===\")
" 2>&1' | tee .claude/test_output.txt
```

- [ ] **Step 3: Verify remote-only mode is NOT broken**

```bash
ssh fabricio@lenovo.local 'cd ~/chidori/chidori-worker && \
  kill $(pgrep chidori-worker) 2>/dev/null; sleep 1; \
  nohup build/chidori-worker --standalone 43211 > /tmp/server.log 2>&1 & sleep 1; \
  LD_PRELOAD=build/libcuda.so.1 \
  LD_LIBRARY_PATH=build \
  CHIDORI_DIRECT_CONNECT=1 \
  CHIDORI_CFG_OVERRIDE='\''{"id":"test","servers":["127.0.0.1:43211"]}'\'' \
  ~/chidori/.venv/bin/python tests/test_inference.py' 2>&1
```

Expected: Existing tests still pass (remote-only regression check).

- [ ] **Step 4: Commit environment documentation updates**

---

### Task 5: Investigate Kernel Launch FPE (if still present after Task 1-4)

The FPE on `torch.randn(device="cuda:0")` may be a **CUDA version mismatch** between PyTorch (cu128 = CUDA 12.8) and the system CUDA toolkit (13.1). The `cudaDeviceProp` struct layout differs between versions. PyTorch reads wrong offsets → invalid kernel launch parameters → FPE.

**Files:**
- Modify: `chidori-worker/src/client/chidori_client.c` (if cudaGetDeviceProperties interception needed)

- [ ] **Step 1: Check if FPE persists after Tasks 1-4**

If the cuBLAS fix also fixes kernel launches (because cuBLAS was the one crashing, not randn), skip this task.

- [ ] **Step 2: If FPE persists, verify CUDA version match**

```bash
ssh fabricio@lenovo.local 'python3 -c "import torch; print(torch.version.cuda)"'
# Should print 12.8 — check against system:
ssh fabricio@lenovo.local '/opt/cuda/bin/nvcc --version | grep release'
# If mismatch (12.8 vs 13.1), that's the cause
```

- [ ] **Step 3: If version mismatch, intercept cudaGetDeviceProperties**

Add a `cudaGetDeviceProperties` interceptor that:
1. For remote device: uses existing RPC path (already works)
2. For local device: calls real driver, then transforms the struct layout from CUDA 13.x to 12.8 format that PyTorch expects

OR: Install CUDA 12.8 toolkit on Lenovo alongside 13.1 and use that cuBLAS.

OR: Build PyTorch with CUDA 13.1.

- [ ] **Step 4: Test and commit**

---

### Task 6: Clean Up Dead Code

After all fixes are verified:

**Files:**
- Modify: `chidori-worker/src/client/chidori_client.c`

- [ ] **Step 1: Remove diagnostic logging from rapid-fire debugging**

Remove `fprintf(stderr, "[chidori-diag] ...")` calls that were added during debugging commits.

- [ ] **Step 2: Remove unused DEEPBIND-related code**

- The `#ifndef RTLD_DEEPBIND` fallback definition at line 31 — can stay (harmless)
- LD_LIBRARY_PATH save/restore logic — already removed in Task 2

- [ ] **Step 3: Build, test, commit**

```bash
git add chidori-worker/src/client/chidori_client.c
git commit -m "cleanup: remove diagnostic logging and dead DEEPBIND code"
```

---

## Why This Works

The critical chain of reasoning:

1. **cuBLAS uses `dlopen("libcuda.so.1")` + `dlsym()`** (confirmed: `nm -D` shows dlopen/dlsym imports, no libcuda DT_NEEDED)
2. **Our SONAME is `libchidori.so`** (confirmed: `readelf -d`, commit 6e59bd2)
3. **`dlopen("libcuda.so.1")` searches**: loaded libs by SONAME → LD_LIBRARY_PATH → RPATH → system paths
4. **If `LD_LIBRARY_PATH=build`**: finds `build/libcuda.so.1` (symlink) → opens → SONAME `libchidori.so` → matches our LD_PRELOAD'd library → cuBLAS gets OUR functions → **REENTRANCE**
5. **If no build/ in LD_LIBRARY_PATH**: finds `/usr/lib/libcuda.so.1` → SONAME `libcuda.so.1` → real driver → cuBLAS gets REAL functions → **WORKS**
6. **`dlsym(real_handle, "cuInit")`** returns the real driver's function, completely bypassing LD_PRELOAD (dlsym on a specific handle is NOT affected by LD_PRELOAD)
7. **Our LD_PRELOAD still intercepts all PLT-based calls** from PyTorch, libtorch_cuda, etc. — remote device routing unaffected

## Why DEEPBIND Never Worked

DEEPBIND changes PLT symbol lookup order (library deps before global scope). But cuBLAS doesn't USE PLT for CUDA symbols — it uses `dlopen` + `dlsym` at runtime. DEEPBIND has zero effect on `dlsym()` calls. This is why 10+ commits trying different DEEPBIND configurations all failed.

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| PyTorch can't find libcuda.so.1 without LD_LIBRARY_PATH=build | Real driver at /usr/lib/libcuda.so.1 satisfies the dependency. LD_PRELOAD ensures our symbols override. |
| cuBLAS finds wrong libcuda.so.1 | LD_DEBUG=libs trace to verify. Our SONAME libchidori.so won't match. |
| Remote-only mode breaks | Separate env vars for remote-only (keep LD_LIBRARY_PATH=build there). |
| CUDA 12.8 vs 13.1 struct mismatch | Separate issue (Task 5), doesn't affect cuBLAS fix. |
| Two libcuda.so.1 instances in process | Only ONE instance — our dlopen in init_real_cuda() and cuBLAS's dlopen both get the same handle (SONAME dedup). |
