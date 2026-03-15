# Plugin base name
PLUGIN_NAME := juice

# Output .so file
TARGET := $(PLUGIN_NAME).so

# Paths to Slurm headers and libraries (update if necessary)
SLURM_INC := /usr/include/slurm
SLURM_LIB := /usr/lib64

# Compiler and flags
CC := gcc
CFLAGS := -Wall -Wextra -O2 -fPIC -I$(SLURM_INC)
LDFLAGS := -shared
SRCS := $(PLUGIN_NAME).c
OBJS := $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
