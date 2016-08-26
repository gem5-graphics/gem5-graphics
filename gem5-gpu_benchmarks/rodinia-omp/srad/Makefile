# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O3
LD_FLAGS =

hw ?= false
ifeq ($(hw), false)
	# Link m5op_x86 for work unit instrumentation
	CC_FLAGS += -DGEM5_WORK
	LD_FLAGS += -L../../libcuda/ -lm5op_x86 -static
else
	CC_FLAGS += -DTIMING
endif

prefetch ?= false
ifeq ($(prefetch), true)
	CC_FLAGS += -fprefetch-loop-arrays
endif

all: 
	$(CC) $(CC_FLAGS) srad.cpp -o srad $(LD_FLAGS)

clean:
	rm -f srad
