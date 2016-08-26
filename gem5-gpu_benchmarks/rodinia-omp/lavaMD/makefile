CSRC = kernel/kernel_cpu.c util/num/num.c util/timer/timer.c
MAIN = main.c
OBJS = $(notdir $(CSRC:.c=.o))
CC = gcc
CFLAGS = -pthread -fopenmp -O3
LDFLAGS = -pthread -lgomp -lm -static
EXECUTABLE = lavaMD
hw ?= 'false'
ifeq ($(hw), 'false')
        # Link m5op_x86 for work unit instrumentation
        CFLAGS += -DGEM5_WORK
        LDFLAGS +=  -L../../libcuda/ -lm5op_x86 -static
endif

all: $(OBJS)
	$(CC) $(OBJS) $(MAIN) -o $(EXECUTABLE) $(LDFLAGS)

$(OBJS): $(CSRC)
	$(CC) -c $(CFLAGS) $(CSRC)

clean:
	rm -f $(OBJS) $(EXECUTABLE)

