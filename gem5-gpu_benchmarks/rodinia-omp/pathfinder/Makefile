CC = g++
SRC = pathfinder.cpp
EXE = pathfinder
FLAGS = -fopenmp -O3

hw ?= false
ifeq ($(hw), false)
	# Link m5op_x86 for work unit instrumentation
	FLAGS += -DGEM5_WORK -L../../libcuda/ -lm5op_x86 -static
else
	FLAGS += -DTIMING
endif

prefetch ?= false
ifeq ($(prefetch), true)
	FLAGS += -fprefetch-loop-arrays
endif

release:
	$(CC) $(SRC) $(FLAGS) -o $(EXE)

debug:
	$(CC) $(SRC) -g -Wall -o $(EXE)

clean:
	rm -f pathfinder


