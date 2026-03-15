# Juice GPU-over-IP Monorepo
# Linux only.

.PHONY: all gpu-worker control-plane test clean

all: gpu-worker control-plane

gpu-worker:
	$(MAKE) -C gpu-worker all

control-plane:
	cd control-plane && go build ./...

test:
	$(MAKE) -C gpu-worker test

test-e2e:
	$(MAKE) -C gpu-worker test-e2e

clean:
	$(MAKE) -C gpu-worker clean
