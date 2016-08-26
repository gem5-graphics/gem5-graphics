CC = gcc
CFLAGS = -fopenmp -Wall -DGEM5_WORK
LDFLAGS = -static -lm -L../../libcuda/ -lm5op_x86

all : nn

clean :
	rm -rf *.o nn

nn : nn_openmp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

