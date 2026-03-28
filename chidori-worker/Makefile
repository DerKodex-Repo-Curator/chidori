# Chidori GPU-over-IP Build System (Linux only)
#
# Targets:
#   make server    - Build chidori-worker (runs on GPU host, links real CUDA)
#   make client    - Build libchidori.so (LD_PRELOAD on client)
#   make all       - Build everything
#   make test      - Run protocol tests (no GPU required)
#   make clean     - Remove build artifacts

CC ?= gcc
CFLAGS = -D_GNU_SOURCE -Wall -Wextra -O2 -fPIC -I include

CUDA_PATH ?= /usr/local/cuda
CUDA_CFLAGS = -I$(CUDA_PATH)/include
CUDA_LDFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcublas -lcublasLt -lcudnn

LDFLAGS_SERVER = -lpthread -lzstd $(CUDA_LDFLAGS)
LDFLAGS_CLIENT = -lpthread -ldl -lzstd -shared

BUILD_DIR = build
SRC_PROTO = src/protocol/chidori_net.c
SRC_SERVER = src/server/chidori_worker.c src/server/cuda_dispatch.c src/server/replay.c
SRC_CLIENT = src/client/chidori_client.c

.PHONY: all server client test test-e2e clean

all: server client

server: $(BUILD_DIR)/chidori-worker

client: $(BUILD_DIR)/libchidori.so $(BUILD_DIR)/libnvidia-ml.so.1 $(BUILD_DIR)/libcublas.so.12 $(BUILD_DIR)/libcublasLt.so.12 $(BUILD_DIR)/libcublas.so.13 $(BUILD_DIR)/libcublasLt.so.13

$(BUILD_DIR)/chidori-worker: $(SRC_SERVER) $(SRC_PROTO) include/chidori_protocol.h src/server/handle_map.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) -I src/server -o $@ $(SRC_SERVER) $(SRC_PROTO) $(LDFLAGS_SERVER)

$(BUILD_DIR)/libchidori.so: $(SRC_CLIENT) $(SRC_PROTO) include/chidori_protocol.h src/client/chidori_client.map
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS_CLIENT) -Wl,-soname,libchidori.so -Wl,--version-script=src/client/chidori_client.map -o $@ $(SRC_CLIENT) $(SRC_PROTO)

# cuBLAS/cuBLASLt proxy libs — same interceptor code compiled with cuBLAS soname.
# When LD_LIBRARY_PATH=build, the dynamic linker finds these instead of real cuBLAS.
# This is how llama.cpp's cuBLAS calls get routed through our interceptor.
$(BUILD_DIR)/libcublas.so.12: $(SRC_CLIENT) $(SRC_PROTO) include/chidori_protocol.h
	$(CC) $(CFLAGS) $(LDFLAGS_CLIENT) -Wl,-soname,libcublas.so.12 -o $@ $(SRC_CLIENT) $(SRC_PROTO)

$(BUILD_DIR)/libcublasLt.so.12: $(SRC_CLIENT) $(SRC_PROTO) include/chidori_protocol.h
	$(CC) $(CFLAGS) $(LDFLAGS_CLIENT) -Wl,-soname,libcublasLt.so.12 -o $@ $(SRC_CLIENT) $(SRC_PROTO)

# CUDA 13.x versions (for systems with CUDA 13 toolkit like Arch Linux)
$(BUILD_DIR)/libcublas.so.13: $(SRC_CLIENT) $(SRC_PROTO) include/chidori_protocol.h
	$(CC) $(CFLAGS) $(LDFLAGS_CLIENT) -Wl,-soname,libcublas.so.13 -o $@ $(SRC_CLIENT) $(SRC_PROTO)

$(BUILD_DIR)/libcublasLt.so.13: $(SRC_CLIENT) $(SRC_PROTO) include/chidori_protocol.h
	$(CC) $(CFLAGS) $(LDFLAGS_CLIENT) -Wl,-soname,libcublasLt.so.13 -o $@ $(SRC_CLIENT) $(SRC_PROTO)

$(BUILD_DIR)/libnvidia-ml.so.1: src/client/chidori_nvml.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC -Wl,-soname,libnvidia-ml.so.1 -o $@ $< -ldl

$(BUILD_DIR)/test_protocol: tests/test_protocol.c $(SRC_PROTO) include/chidori_protocol.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ tests/test_protocol.c $(SRC_PROTO) -lpthread

$(BUILD_DIR)/test_e2e_cuda: tests/test_e2e_cuda.c $(SRC_PROTO) include/chidori_protocol.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) -o $@ tests/test_e2e_cuda.c $(SRC_PROTO) -lpthread $(CUDA_LDFLAGS)

test: $(BUILD_DIR)/test_protocol
	@echo "=== Protocol Tests ==="
	@./$(BUILD_DIR)/test_protocol

test-e2e: $(BUILD_DIR)/test_e2e_cuda
	@echo "=== E2E CUDA Tests ==="
	@./$(BUILD_DIR)/test_e2e_cuda

test-all: all
	@./tests/run_all.sh

clean:
	rm -rf $(BUILD_DIR)
