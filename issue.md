 Replay System Bug Fix — Implementation Plan                                                                             
                                                                                  
 ▎ For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
 superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

 Goal: Fix all 13 critical bugs in Chidori's replay system so training with CHIDORI_REPLAY=1 produces bitwise-identical
 results to training without it.

 Architecture: The replay system records one training step's CUDA calls on the server, then replays them with
 substituted input data. The bugs are in 4 areas: (1) missing H2D capture on client causes crash, (2) wrong payload
 offsets corrupt data, (3) D2H output unimplemented so loss.item() returns 0, (4) memcpy direction ignored in
 classification.

 Tech Stack: C (gcc), CUDA driver API, Chidori wire protocol, Python test scripts

 Servers: Client = fabricio@lenovo.local, GPU Server = ubuntu@185.216.20.105 (A100)

 ---
 Critical Files

 ┌──────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┐
 │               File               │                                  What changes                                  │
 ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ src/client/chidori_client.c      │ Add cudaMemcpyAsync H2D capture, stop intercepting D2H calls, fix multi-sync   │
 ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ src/server/replay.c              │ Fix classify_call() to inspect payload direction, fix H2D offsets, fix         │
 │                                  │ shutdown leak                                                                  │
 ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ src/server/replay.h              │ Add helper function declarations                                               │
 ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ src/server/cuda_dispatch.c       │ Check recording overflow return value                                          │
 ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ src/protocol/chidori_net.c       │ Make chidori_send_response() succeed silently when fd=-1                       │
 ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ .claude/settings.local.json      │ Add pre-commit replay validation hook                                          │
 ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ tests/test_replay_correctness.py │ Already exists — validation test                                               │
 └──────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────┘

 Payload Layouts (reference)

 cuMemcpyHtoD (0x0072):      dst(8) + size(8) + data(N)           — data at offset 16
 cuMemcpyHtoDAsync (0x0075):  dst(8) + size(8) + stream(8) + data(N) — data at offset 24
 cudaMemcpy (0x0102):         dst(8) + src(8) + size(8) + kind(4) + data(N) — data at 28, kind at 24
 cudaMemcpyAsync (0x0103):    dst(8) + src(8) + size(8) + kind(4) + stream(8) + data(N) — data at 36, kind at 24
 cuMemcpyDtoH (0x0073):       src(8) + size(8)  — 16 bytes, output in response
 cuLaunchKernel:              44-byte header + has_params(1) + buf_size(4) + params(N) — params at 49
 cublasLtMatmul:              308 bytes. A/B/C/D ptrs at 204-235. Alpha 188, beta 196.

 ---
 Task 1: Pre-commit Replay Validation Hook

 Files:
 - Modify: chidori-worker/.claude/settings.local.json
 - Step 1: Add hook to settings.local.json

 The hook runs test_replay_correctness.py in A/B mode (baseline vs replay) before commits. Skip when CLAUDE_AGENT_DEPTH
 > 0 (subagent).

 ```json
 {
   "permissions": {
     "allow": [
       "mcp__obsidian__list_directory",
       "mcp__obsidian__write_note"
     ]
   },
   "hooks": {
     "PreCommit": [
       {
         "command": "bash -c '[ \"${CLAUDE_AGENT_DEPTH:-0}\" -gt 0 ] && exit 0; cd chidori-worker && test -f build/libcuda.so.1 || exit 0; echo \"[hook] Running replay validation...\"; SERVER_IP=${CHIDORI_TEST_SERVER:-185.216.20.105}; PORT=${CHIDORI_TEST_PORT:-43211}; ENV=\"CHIDORI_DIRECT_CONNECT=1 CHIDORI_CFG_OVERRIDE={\\\"id\\\":\\\"test\\\",\\\"servers\\\":[\\\"$SERVER_IP:$PORT\\\"]} LD_PRELOAD=build/libcuda.so.1 LD_LIBRARY_PATH=build\"; BASELINE=$(eval $ENV python tests/test_replay_correctness.py 2>&1 | grep -E \"step.*:|WEIGHT HASH\"); ssh ubuntu@$SERVER_IP \"pkill chidori-worker; sleep 1; cd ~/chidori/chidori-worker && nohup build/chidori-worker --standalone $PORT > /tmp/server.log 2>&1 &\" && sleep 3; REPLAY=$(eval $ENV CHIDORI_REPLAY=1 python tests/test_replay_correctness.py 2>&1 | grep -E \"step.*:|WEIGHT HASH\"); if [ \"$BASELINE\" = \"$REPLAY\" ]; then echo \"[hook] Replay validation PASSED\"; else echo \"[hook] Replay validation FAILED\"; echo \"BASELINE:\"; echo \"$BASELINE\"; echo \"REPLAY:\"; echo \"$REPLAY\"; exit 1; fi'"
       }
     ]
   }
 }
 ```

 - Step 2: Verify hook loads — Run any commit attempt and confirm the hook message appears.

 ---
 Task 2: Fix Immediate Crash — Client H2D Collection (Bug #9)

 This is the #1 priority. REPLAY_EXECUTE fails with error -3 because client doesn't send cudaMemcpyAsync H2D data.

 Files:
 - Modify: src/client/chidori_client.c:478-511 (replay_intercept function)
 - Step 1: Add cudaMemcpyAsync H2D capture

 After the existing CHIDORI_CUDA_MEMCPY block (~line 508), add:
```c
 /* Collect H2D data (runtime API async: cudaMemcpyAsync HtoD)
  * payload = dst(8)+src(8)+size(8)+kind(4)+stream(8)+data */
 if (func_id == CHIDORI_CUDA_MEMCPY_ASYNC && payload_len > 36) {
     int32_t kind = 0;
     memcpy(&kind, (char*)payload + 24, 4);
     if (kind == 1 && g_replay_h2d_count < REPLAY_MAX_H2D_COLLECT) {
         uint32_t data_size = payload_len - 36;
         g_replay_h2d_collected[g_replay_h2d_count].data = malloc(data_size);
         memcpy(g_replay_h2d_collected[g_replay_h2d_count].data,
                (char*)payload + 36, data_size);
         g_replay_h2d_collected[g_replay_h2d_count].size = data_size;
         g_replay_h2d_count++;
     }
 }
 ```

 - Step 2: Add cuMemcpyHtoDAsync H2D capture (future-proofing)
```c
 /* Collect H2D data (driver API async: cuMemcpyHtoDAsync)
  * payload = dst(8)+size(8)+stream(8)+data */
 if (func_id == 0x0075 && payload_len > 24) {
     if (g_replay_h2d_count < REPLAY_MAX_H2D_COLLECT) {
         uint32_t data_size = payload_len - 24;
         g_replay_h2d_collected[g_replay_h2d_count].data = malloc(data_size);
         memcpy(g_replay_h2d_collected[g_replay_h2d_count].data,
                (char*)payload + 24, data_size);
         g_replay_h2d_collected[g_replay_h2d_count].size = data_size;
         g_replay_h2d_count++;
     }
 }
 ```

 - Step 3: Build, push, test — REPLAY_EXECUTE should no longer return -3.

 ---
 Task 3: Fix D2H — Stop Intercepting D2H Calls (Bug #11, #12)

 This fixes loss.item() returning 0.0 during replay. D2H calls should pass through as normal RPCs.

 Files:
 - Modify: src/client/chidori_client.c:478-511 (replay_intercept function)
 - Step 1: Add early return for D2H calls in replay_intercept()

 At the TOP of replay_intercept(), after the mode/func_id checks, add:

 /* Do NOT intercept D2H calls — let them pass through as normal RPCs.
  * The GPU has the data from REPLAY_EXECUTE, D2H reads it back normally. */
 if (func_id == CHIDORI_CU_MEMCPY_D_TO_H || func_id == 0x0076 /*DtoH async*/) {
     return 0;
 }
 if ((func_id == CHIDORI_CUDA_MEMCPY || func_id == CHIDORI_CUDA_MEMCPY_ASYNC) &&
     payload_len >= 28) {
     int32_t kind = 0;
     memcpy(&kind, (char*)payload + 24, 4);
     if (kind == 2) return 0; /* DtoH — don't intercept */
 }

 When replay_intercept returns 0, the caller in chidori_rpc() falls through to the normal RPC path, which sends the D2H
 request to the server, gets the real data back, and returns it to PyTorch.

 - Step 2: Build, push, test — loss.item() should return real values during replay.

 ---
 Task 4: Fix classify_call() Direction Handling (Bugs #1-5)

 Files:
 - Modify: src/server/replay.c:43-91 (classify_call function)
 - Modify: src/server/replay.c:216 (call site in replay_record_call)
 - Step 1: Change classify_call signature to accept payload

 static replay_call_type_t classify_call(uint16_t func_id,
                                          const void *payload,
                                          uint32_t payload_size)

 - Step 2: Fix CUDA_MEMCPY/CUDA_MEMCPY_ASYNC cases to inspect kind field

 Replace lines 47-48 with:

 case CHIDORI_CUDA_MEMCPY:
 case CHIDORI_CUDA_MEMCPY_ASYNC: {
     if (payload && payload_size >= 28) {
         int32_t kind = 0;
         memcpy(&kind, (const char *)payload + 24, 4);
         if (kind == 1) return REPLAY_CALL_H2D;
         if (kind == 2) return REPLAY_CALL_D2H;
     }
     return REPLAY_CALL_FIXED; /* DtoD or unknown */
 }

 - Step 3: Add async driver API memcpy variants

 Add to H2D case block: case 0x0075: /* CHIDORI_CU_MEMCPY_H_TO_D_ASYNC */
 Add to D2H case block: case 0x0076: /* CHIDORI_CU_MEMCPY_D_TO_H_ASYNC */

 - Step 4: Update call site

 In replay_record_call (~line 230), change:
 call->type = classify_call(func_id, payload, payload_size);

 - Step 5: Build, test

 ---
 Task 5: Fix identify_var_slots() H2D Offsets (Bug #6)

 Files:
 - Modify: src/server/replay.c:100-107 (H2D case in identify_var_slots)
 - Step 1: Replace hardcoded offset 16 with func_id-dependent offsets

 case REPLAY_CALL_H2D: {
     uint32_t data_offset;
     switch (call->func_id) {
     case 0x0072: /* cuMemcpyHtoD */
         data_offset = 16; break;
     case 0x0075: /* cuMemcpyHtoDAsync */
         data_offset = 24; break;
     case 0x0102: /* cudaMemcpy */
         data_offset = 28; break;
     case 0x0103: /* cudaMemcpyAsync */
         data_offset = 36; break;
     default:
         data_offset = 16; break;
     }
     if (call->payload_size > data_offset) {
         call->var_slots[0].offset = data_offset;
         call->var_slots[0].length = call->payload_size - data_offset;
         call->var_slot_count = 1;
     }
     break;
 }

 - Step 2: Build, test

 ---
 Task 6: Fix chidori_send_response for Replay Dispatch (Bug #14)

 Files:
 - Modify: src/protocol/chidori_net.c (chidori_send_response function)
 - Step 1: Add early return for fd=-1

 At the top of chidori_send_response():

 if (fd < 0) return 0; /* replay mode: suppress response silently */

 This makes the replay dispatch loop stop producing error returns for every call.

 - Step 2: Build, test

 ---
 Task 7: Fix Multi-Sync and Remaining Bugs (Bugs #15, #16, #17)

 Files:
 - Modify: src/client/chidori_client.c (cudaDeviceSynchronize, add g_replay_executed_this_step)
 - Modify: src/server/replay.c (replay_shutdown, fix sequence_id)
 - Modify: src/server/cuda_dispatch.c (check replay_record_call return value)
 - Step 1: Fix multi-sync per step

 Add state variable near line 444:
 static int g_replay_executed_this_step = 0;

 In cudaDeviceSynchronize replay block (~line 2648), wrap the REPLAY_EXECUTE code:
 if (g_replay_mode == 2 && !g_replay_executed_this_step) {
     g_replay_executed_this_step = 1;
     /* ... existing REPLAY_EXECUTE code ... */
     /* Reset at end of block: */
     g_replay_executed_this_step = 0;

 For additional syncs within the same step (after replay already fired):
 } else if (g_replay_mode == 2 && g_replay_executed_this_step) {
     /* Extra sync within step — just do a real sync */
     chidori_rpc(CHIDORI_CUDA_DEVICE_SYNCHRONIZE, NULL, 0, NULL, NULL);
     g_replay_step_count++;
     return cudaSuccess;
 }

 - Step 2: Fix replay_shutdown() memory leak

 In replay.c replay_shutdown(), change:
 for (int i = 0; i < REPLAY_MAX_SEQUENCES; i++) {
     if (g_sequences[i].active)
         replay_invalidate(g_sequences[i].sequence_id);
 }

 - Step 3: Fix recording overflow handling

 In cuda_dispatch.c (~line 293), change:
 if (replay_is_recording() && req->func_id < 0x0400) {
     if (replay_record_call(req->func_id, req->flags, payload, req->payload_size) < 0) {
         fprintf(stderr, "[replay] Recording overflow, aborting\n");
     }
 }

 - Step 4: Build, push, full test

 ---
 Task 8: Final Validation — Over Wire A/B Test

 - Step 1: Rebuild on both machines

 # Lenovo
 ssh fabricio@lenovo.local 'cd ~/chidori/chidori-worker && git pull && make clean && make all CUDA_PATH=/opt/cuda && cd
 build && ln -sf libchidori.so libcuda.so.1'

 # A100
 ssh ubuntu@185.216.20.105 'cd ~/chidori && git pull && cd chidori-worker && make clean && make all
 CUDA_PATH=/usr/local/cuda CUDNN_INCLUDE=/usr/include/x86_64-linux-gnu'

 - Step 2: Run baseline (no replay)

 ssh ubuntu@185.216.20.105 'pkill chidori-worker; sleep 1; cd ~/chidori/chidori-worker && nohup build/chidori-worker
 --standalone 43211 > /tmp/server.log 2>&1 &'
 sleep 3
 ssh fabricio@lenovo.local 'cd ~/chidori/chidori-worker && \
   CHIDORI_DIRECT_CONNECT=1 CHIDORI_CFG_OVERRIDE=... LD_PRELOAD=build/libcuda.so.1 LD_LIBRARY_PATH=build \
   python tests/test_replay_correctness.py' > /tmp/baseline.txt

 - Step 3: Run with replay

 Restart server, then:
 ssh fabricio@lenovo.local '... CHIDORI_REPLAY=1 python tests/test_replay_correctness.py' > /tmp/replay.txt

 - Step 4: Compare

 diff <(grep -E 'step|WEIGHT HASH' /tmp/baseline.txt) <(grep -E 'step|WEIGHT HASH' /tmp/replay.txt)

 Empty diff = PASS. All losses identical, weight hash identical.

 - Step 5: Run full training suite with CHIDORI_REPLAY=1 to verify no regressions

 ---
 Minimum Viable for Demo

 Tasks 1-6 are required. Task 7 is robustness. Task 8 is validation. Total estimate: ~200 lines of C changes across 4
 files.