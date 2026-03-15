# Chidori ⚡ GPU-over-IP
# Linux only.

.PHONY: all chidori-worker control-plane test clean

all: chidori-worker control-plane

chidori-worker:
	$(MAKE) -C chidori-worker all

control-plane:
	cd control-plane && go build ./...

test:
	$(MAKE) -C chidori-worker test

test-e2e:
	$(MAKE) -C chidori-worker test-e2e

clean:
	$(MAKE) -C chidori-worker clean
