# Partial GPU Offloading — GPU to GPU

One client with a local GPU (cuda:0) + one remote chidori server (cuda:1). PyTorch sees 2 devices. CPU-to-GPU works. What remains:

1. Fix cuda:0 compute crash — cuGetProcAddress forwarding to the real driver crashes during compute. Debug and fix the routing/NULL pointer issue.
2. GPU-to-GPU transfers — cuMemcpyDtoD/cuMemcpyDtoDAsync between local and remote device. Stage through host memory when crossing devices.
3. Compute on both devices — kernel launches, cuBLAS, cuDNN must work on cuda:0 (real local driver) and cuda:1 (remote via chidori).
4. All existing tests pass: test_train.py, test_training.py, test_training_fp16.py, test_inference.py under partial offload config.

One server. Never multi-server. Done means all tests pass.
