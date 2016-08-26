#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda.h>
#include <stdio.h>

void CHECK_ERROR(const char *error_message) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
                error_message, __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#endif
