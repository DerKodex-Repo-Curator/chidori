# Juice GPU-over-IP Build System (Linux only)
#
# Targets:
#   make server    - Build juice-gpu-worker (runs on GPU host, links real CUDA)
#   make client    - Build libjuice-cuda.so (LD_PRELOAD on client)
#   make all       - Build everything
#   make test      - Run protocol tests (no GPU required)
#   make clean     - Remove build artifacts

CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -I include

CUDA_PATH ?= /usr/local/cuda
CUDA_CFLAGS = -I$(CUDA_PATH)/include
CUDA_LDFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart

LDFLAGS_SERVER = -lpthread $(CUDA_LDFLAGS)
LDFLAGS_CLIENT = -lpthread -shared

BUILD_DIR = build
SRC_PROTO = src/protocol/juice_net.c
SRC_SERVER = src/server/gpu_worker.c src/server/cuda_dispatch.c
SRC_CLIENT = src/client/juice_client.c

.PHONY: all server client test test-e2e clean

all: server client

server: $(BUILD_DIR)/juice-gpu-worker

client: $(BUILD_DIR)/libjuice-cuda.so

$(BUILD_DIR)/juice-gpu-worker: $(SRC_SERVER) $(SRC_PROTO) include/juice_protocol.h src/server/handle_map.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) -I src/server -o $@ $(SRC_SERVER) $(SRC_PROTO) $(LDFLAGS_SERVER)

$(BUILD_DIR)/libjuice-cuda.so: $(SRC_CLIENT) $(SRC_PROTO) include/juice_protocol.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS_CLIENT) -o $@ $(SRC_CLIENT) $(SRC_PROTO)

$(BUILD_DIR)/test_protocol: tests/test_protocol.c $(SRC_PROTO) include/juice_protocol.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ tests/test_protocol.c $(SRC_PROTO) -lpthread

$(BUILD_DIR)/test_e2e_cuda: tests/test_e2e_cuda.c $(SRC_PROTO) include/juice_protocol.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) -o $@ tests/test_e2e_cuda.c $(SRC_PROTO) -lpthread $(CUDA_LDFLAGS)

test: $(BUILD_DIR)/test_protocol
	@echo "=== Protocol Tests ==="
	@./$(BUILD_DIR)/test_protocol

test-e2e: $(BUILD_DIR)/test_e2e_cuda
	@echo "=== E2E CUDA Tests ==="
	@./$(BUILD_DIR)/test_e2e_cuda

clean:
	rm -rf $(BUILD_DIR)
