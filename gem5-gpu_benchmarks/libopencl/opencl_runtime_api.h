#ifndef OPENCL_RUNTIME_API_H
#define OPENCL_RUNTIME_API_H

typedef struct gpucall {
    int total_bytes;
    int num_args;
    int* arg_lengths;
    char* args;
    char* ret;
} gpusyscall_t;

#endif
