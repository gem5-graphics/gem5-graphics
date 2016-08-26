CC_FLAGS = -fopenmp -O3
LD_FLAGS = -lm
CC = gcc
EXECUTABLE = heartwall
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

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o
	$(CC) main.o ./AVI/avilib.o ./AVI/avimod.o $(CC_FLAGS) -o $(EXECUTABLE) $(LD_FLAGS)

# compile main function file into object (binary)
main.o: main.c define.c kernel.c
	$(CC) main.c -I./AVI -c $(CC_FLAGS)

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o $(EXECUTABLE)
