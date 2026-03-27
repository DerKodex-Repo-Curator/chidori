CUDA_PATH ?= /opt/cuda
CC = gcc
NVCC = $(CUDA_PATH)/bin/nvcc

CFLAGS = -Wall -O0 -g -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/targets/x86_64-linux/lib -lcuda -lcudart -ldl -lpthread
RUNPATH = -Wl,-rpath,$(CUDA_PATH)/targets/x86_64-linux/lib

# Fat binary for testing (sm_89 matches local RTX 4090)
FATBIN = simple_sm89.fatbin
KERNEL = simple_add

all: test_isolated_context_first test_isolated_register_first probe_runtime_state $(FATBIN)

# Compile the isolated test programs
test_isolated_context_first: test_isolated_context_first.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(RUNPATH)

test_isolated_register_first: test_isolated_register_first.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(RUNPATH)

probe_runtime_state: probe_runtime_state.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(RUNPATH)

# Compile test kernel fat binary (sm_89, uncompressed)
$(FATBIN): simple.cu
	$(NVCC) -gencode arch=compute_89,code=sm_89 -fatbin -o $@ $<

# Run the proof tests
test: all
	@echo "================================================================"
	@echo "TEST 1: Register BEFORE context (should PASS)"
	@echo "================================================================"
	./test_isolated_register_first $(FATBIN) $(KERNEL)
	@echo ""
	@echo "================================================================"
	@echo "TEST 2: Context BEFORE register (should FAIL with error 4)"
	@echo "================================================================"
	./test_isolated_context_first $(FATBIN) $(KERNEL)
	@echo ""
	@echo "================================================================"
	@echo "RUNTIME STATE PROBE"
	@echo "================================================================"
	./probe_runtime_state $(FATBIN) $(KERNEL)

clean:
	rm -f test_isolated_context_first test_isolated_register_first probe_runtime_state
	rm -f *.fatbin *.cubin *.ptx *.o *.cudafe1.* *.cpp1.ii *.module_id

.PHONY: all test clean
