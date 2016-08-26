# C compiler
CC = gcc
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

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o 
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o $(LD_FLAGS) -o kmeans

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c
	
getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c
	
kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) kmeans.c -c

kmeans_clustering.o: kmeans_clustering.c kmeans.h
	$(CC) $(CC_FLAGS) kmeans_clustering.c -c

clean:
	rm -f *.o *~ kmeans 
