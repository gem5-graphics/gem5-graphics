# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O3
LD_FLAGS = -lm
hw ?= false
ifeq ($(hw), false)
	# Link m5op_x86 for work unit instrumentation
	CC_FLAGS += -DGEM5_WORK
	LD_FLAGS +=  -L../../libcuda/ -lm5op_x86 -static
else
	CC_FLAGS += -DTIMING
endif
prefetch ?= false
ifeq ($(prefetch), true)
	CC_FLAGS += -fprefetch-loop-arrays
endif
backprop: backprop.o facetrain.o imagenet.o backprop_kernel.o 
	$(CC) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_kernel.o -o backprop $(LD_FLAGS)

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_kernel.o: backprop_kernel.c backprop.h
	$(CC) $(CC_FLAGS) backprop_kernel.c -c

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop
