# Chidori GPU-over-IP Build System (Linux only)
#
# Targets:
#   make server    - Build chidori-worker (runs on GPU host, links real CUDA)
#   make client    - Build libchidori.so (LD_PRELOAD on client)
#   make all       - Build everything
#   make test      - Run protocol tests (no GPU required)
#   make clean     - Remove build artifacts

CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -I include

CUDA_PATH ?= /usr/local/cuda
CUDA_CFLAGS = -I$(CUDA_PATH)/include
CUDA_LDFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcublas -lcublasLt -lcudnn

LDFLAGS_SERVER = -lpthread -lzstd $(CUDA_LDFLAGS)
LDFLAGS_CLIENT = -lpthread -ldl -shared

BUILD_DIR = build
SRC_PROTO = src/protocol/chidori_net.c
SRC_SERVER = src/server/chidori_worker.c src/server/cuda_dispatch.c
SRC_CLIENT = src/client/chidori_client.c

.PHONY: all server client test test-e2e clean

all: server client

server: $(BUILD_DIR)/chidori-worker

client: $(BUILD_DIR)/libchidori.so $(BUILD_DIR)/libnvidia-ml.so.1

$(BUILD_DIR)/chidori-worker: $(SRC_SERVER) $(SRC_PROTO) include/chidori_protocol.h src/server/handle_map.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) -I src/server -o $@ $(SRC_SERVER) $(SRC_PROTO) $(LDFLAGS_SERVER)

$(BUILD_DIR)/libchidori.so: $(SRC_CLIENT) $(SRC_PROTO) include/chidori_protocol.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS_CLIENT) -Wl,-soname,libchidori.so -o $@ $(SRC_CLIENT) $(SRC_PROTO)

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
