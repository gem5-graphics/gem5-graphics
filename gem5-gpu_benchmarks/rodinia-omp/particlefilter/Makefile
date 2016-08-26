#makefile

openmp: ex_particle_OPENMP_seq.c
	gcc -O3 -ffast-math -fopenmp -DGEM5_WORK ex_particle_OPENMP_seq.c -o particlefilter -lm -L../../libcuda/ -lm5op_x86 -static

clean:
	rm particlefilter
