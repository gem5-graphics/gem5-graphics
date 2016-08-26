#include <cuda.h>
#include <stdio.h>

__device__ float *dev_global_array;
__constant__ float *dev_constant_array;

__global__ void check_pointers(float **global_ptr, float **constant_ptr) {
    constant_ptr[0] = dev_constant_array;
    global_ptr[0] = dev_global_array;
}

__global__ void sum_values(bool use_constant, float *out_val) {
    float *array = dev_global_array;
    if (use_constant) {
        array = dev_constant_array;
    }
    float sum = 0.0;
    for (int i = 0; i < 1024; i++) {
        sum += array[i];
    }
    out_val[0] = sum;
}

int main(int argc, char **argv) {
    // Variables
    unsigned array_size = 1024;

    // cudaMalloc data array
    float *dev_global;
    cudaMalloc(&dev_global, array_size * sizeof(float));
    float *dev_constant;
    cudaMalloc(&dev_constant, array_size * sizeof(float));

    float array[array_size];
    for (int i = 0; i < array_size; i++) {
        array[i] = (float)i;
    }

    cudaMemcpy(dev_global, array, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_constant, array, array_size * sizeof(float), cudaMemcpyHostToDevice);

    float **dev_global_ptr;
    cudaMalloc(&dev_global_ptr, sizeof(float*));
    float **dev_constant_ptr;
    cudaMalloc(&dev_constant_ptr, sizeof(float*));

    float *global_ptr = NULL;
    float *constant_ptr = NULL;

    printf("Before copy:   ");
    printf("d_g: %p (@%p), d_c: %p (@%p), ", dev_global, &dev_global, dev_constant, &dev_constant);
    printf("d_g_p: %p (@%p), d_c_p: %p (@%p), ", dev_global_ptr, &dev_global_ptr, dev_constant_ptr, &dev_constant_ptr);
    printf("d_g_a: %p (@%p), d_c_a: %p (@%p), ", dev_global_array, &dev_global_array, dev_constant_array, &dev_constant_array);
    printf("g_p: %p (@%p), c_p: %p (@%p)\n", global_ptr, &global_ptr, constant_ptr, &constant_ptr);

    cudaMemcpyToSymbol(dev_global_array, &dev_global, sizeof(float*));
    cudaMemcpyToSymbol(dev_constant_array, &dev_constant, sizeof(float*));

    printf("Before kernel: ");
    printf("d_g: %p (@%p), d_c: %p (@%p), ", dev_global, &dev_global, dev_constant, &dev_constant);
    printf("d_g_p: %p (@%p), d_c_p: %p (@%p), ", dev_global_ptr, &dev_global_ptr, dev_constant_ptr, &dev_constant_ptr);
    printf("d_g_a: %p (@%p), d_c_a: %p (@%p), ", dev_global_array, &dev_global_array, dev_constant_array, &dev_constant_array);
    printf("g_p: %p (@%p), c_p: %p (@%p)\n", global_ptr, &global_ptr, constant_ptr, &constant_ptr);

    check_pointers<<<1, 1>>>(dev_global_ptr, dev_constant_ptr);

    cudaMemcpy(&global_ptr, dev_global_ptr, sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&constant_ptr, dev_constant_ptr, sizeof(float*), cudaMemcpyDeviceToHost);

    printf("After kernel:  ");
    printf("d_g: %p (@%p), d_c: %p (@%p), ", dev_global, &dev_global, dev_constant, &dev_constant);
    printf("d_g_p: %p (@%p), d_c_p: %p (@%p), ", dev_global_ptr, &dev_global_ptr, dev_constant_ptr, &dev_constant_ptr);
    printf("d_g_a: %p (@%p), d_c_a: %p (@%p), ", dev_global_array, &dev_global_array, dev_constant_array, &dev_constant_array);
    printf("g_p: %p (@%p), c_p: %p (@%p)\n", global_ptr, &global_ptr, constant_ptr, &constant_ptr);

    printf("After copy:    ");
    printf("d_g: %p (@%p), d_c: %p (@%p), ", dev_global, &dev_global, dev_constant, &dev_constant);
    printf("d_g_p: %p (@%p), d_c_p: %p (@%p), ", dev_global_ptr, &dev_global_ptr, dev_constant_ptr, &dev_constant_ptr);
    printf("d_g_a: %p (@%p), d_c_a: %p (@%p), ", dev_global_array, &dev_global_array, dev_constant_array, &dev_constant_array);
    printf("g_p: %p (@%p), c_p: %p (@%p)\n", global_ptr, &global_ptr, constant_ptr, &constant_ptr);

    float *dev_output;
    cudaMalloc(&dev_output, sizeof(float));
    float output;

    printf("\nSum Test Global:\n");
    sum_values<<<1, 1>>>(false, dev_output);
    cudaMemcpy(&output, dev_output, sizeof(float), cudaMemcpyDeviceToHost);
    if (output != (1023.0 * 1024.0 / 2.0)) {
        printf("TEST FAILED: %f\n", output);
    } else {
        printf("TEST PASSED\n");
    }

    printf("\nSum Test Constant:\n");
    cudaMemset(dev_output, 0, sizeof(float));
    sum_values<<<1, 1>>>(true, dev_output);
    cudaMemcpy(&output, dev_output, sizeof(float), cudaMemcpyDeviceToHost);
    if (output != (1023.0 * 1024.0 / 2.0)) {
        printf("TEST FAILED: %f\n", output);
    } else {
        printf("TEST PASSED\n");
    }

    cudaFree(dev_global);
    cudaFree(dev_constant);
    cudaFree(dev_global_ptr);
    cudaFree(dev_constant_ptr);
    cudaFree(dev_output);
    return 0;
}
