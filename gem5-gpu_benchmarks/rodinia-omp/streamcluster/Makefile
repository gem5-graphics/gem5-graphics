TARGET = streamcluster
FLAGS = -Wall -Wunused
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

all: omp

omp:
	g++ -O3 -fopenmp -o $(TARGET) streamcluster.cpp $(FLAGS)

clean:
	rm -f *.o *~ *.txt $(TARGET)
