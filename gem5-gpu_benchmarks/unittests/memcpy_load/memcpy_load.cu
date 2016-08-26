/*****************************************************************************
 * A microbenchmark to test the performance of varying memory copy operations
 * including different sizes and different sources and destinations
 ****************************************************************************/

#include <cassert>
#include <cmath>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#ifdef GEM5_FUSION
extern "C" {
    void m5_dumpreset_stats(uint64_t workid, uint64_t threadid);
    uint64_t rpns();
}
#endif

#define PAGE_SIZE_BYTES 4096
unsigned char touchArrayPages(unsigned char *ptr, size_t size)
{
    unsigned char sum = 0;
    for (unsigned i = 0; i < size; i += PAGE_SIZE_BYTES) {
        sum += ptr[i];
    }
    sum += ptr[size-1];
    return sum;
}

int main(int argc, char** argv) {

#ifdef GEM5GPU_AUTOMAP_COPIES
    cudaSetDeviceFlags(cudaDeviceMapHost);
#endif

    unsigned minimum_elements = 2;
    unsigned maximum_elements = 16 * 1024 * 1024;

    for (int index = 0; index < argc; ++index) {
        if (strcmp(argv[index], "-m") == 0) {
            if (argc > index+1) {
                minimum_elements = atoi(argv[++index]);
            } else {
                printf("ERROR: Must specify minimum array size to -m option\n");
                exit(0);
            }
        } else if (strcmp(argv[index], "-M") == 0) {
            if (argc > index+1) {
                maximum_elements = atoi(argv[++index]);
            } else {
                printf("ERROR: Must specify maximum array size to -M option\n");
                exit(0);
            }
        }
    }

    unsigned total_iterations = 2 * ((unsigned)log2((float)maximum_elements) - (unsigned)log2((float)minimum_elements)) + 1;
    unsigned *sizes = (unsigned*) malloc(total_iterations * sizeof(unsigned));
    double *malloc_times = (double*) malloc(total_iterations * sizeof(double));
    double *copy_times = (double*) malloc(total_iterations * sizeof(double));
    unsigned num_iterations = 0;

    bool power_two = true;
    unsigned *array = (unsigned*) malloc(maximum_elements * sizeof(unsigned));
    // In a real application, the host-side memory is likely to have been used
    // (touched) and thus, mapped before memory copies over to the device.
    unsigned hash = touchArrayPages((unsigned char*)array, maximum_elements * sizeof(unsigned));
    printf("Touch pages proof hash: %u\n", hash);
    unsigned *d_array;

    printf("Testing copy host-to-device:\n");
    printf("Size:\tMalloc (s):\tCopy (s):\tCopy (GB/s):\n");
#ifdef GEM5_FUSION
    uint64_t start;
#else
    float tmp_t;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    for (unsigned i = minimum_elements; i <= maximum_elements;) {
        sizes[num_iterations] = i;
#ifdef GEM5_FUSION
        start = rpns();
        m5_dumpreset_stats(0, 0);
#else
        cudaEventRecord(start, 0);
#endif
        cudaMalloc(&d_array, i * sizeof(unsigned));
#ifdef GEM5_FUSION
        m5_dumpreset_stats(0, 0);
        malloc_times[num_iterations] = (double) (rpns() - start) / 1000000000.0;
        start = rpns();
#else
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tmp_t, start, stop);
        malloc_times[num_iterations] = (double) tmp_t / 1000.0;
        cudaEventRecord(start, 0);
#endif
        cudaMemcpy(d_array, array, i * sizeof(unsigned), cudaMemcpyHostToDevice);
#ifdef GEM5_FUSION
        m5_dumpreset_stats(0, 0);
        copy_times[num_iterations] = (double) (rpns() - start) / 1000000000.0;
#else
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tmp_t, start, stop);
        copy_times[num_iterations] = (double) tmp_t / 1000.0;
#endif
        if (power_two) {
            if (i > 1) {
                i = 3 * i / 2;
                power_two = false;
            } else {
                i = 2;
            }
        } else {
            i = 4 * i / 3;
            power_two = true;
        }
        num_iterations++;
    }

    // Print the copy time of the cudaMemcpyHostToDevice operations
    assert(num_iterations == total_iterations);
    for (unsigned i = 0; i < num_iterations; i++) {
        double bandwidth = ((double)(sizes[i] * 4) / copy_times[i]) / 1073741824.0;
        printf("%u\t%f\t%f\t%f\n", sizes[i], malloc_times[i], copy_times[i], bandwidth);
    }

    num_iterations = 0;
    power_two = true;

    printf("\n\nTesting copy device-to-host:\n");
    printf("Size:\tMalloc (s):\tCopy (s):\tCopy (GB/s):\n");
    for (unsigned i = minimum_elements; i <= maximum_elements;) {
        sizes[num_iterations] = i;
#ifdef GEM5_FUSION
        start = rpns();
        m5_dumpreset_stats(0, 0);
#else
        cudaEventRecord(start, 0);
#endif
        cudaMalloc(&d_array, i * sizeof(unsigned));
#ifdef GEM5_FUSION
        m5_dumpreset_stats(0, 0);
        malloc_times[num_iterations] = (double) (rpns() - start) / 1000000000.0;
        start = rpns();
#else
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tmp_t, start, stop);
        malloc_times[num_iterations] = (double) tmp_t / 1000.0;
        cudaEventRecord(start, 0);
#endif
        cudaMemcpy(array, d_array, i * sizeof(unsigned), cudaMemcpyDeviceToHost);
#ifdef GEM5_FUSION
        m5_dumpreset_stats(0, 0);
        copy_times[num_iterations] = (double) (rpns() - start) / 1000000000.0;
#else
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tmp_t, start, stop);
        copy_times[num_iterations] = (double) tmp_t / 1000.0;
#endif
        if (power_two) {
            if (i > 1) {
                i = 3 * i / 2;
                power_two = false;
            } else {
                i = 2;
            }
        } else {
            i = 4 * i / 3;
            power_two = true;
        }
        num_iterations++;
    }

    // Print the copy time of the cudaMemcpyDeviceToHost operations
    assert(num_iterations == total_iterations);
    for (unsigned i = 0; i < num_iterations; i++) {
        double bandwidth = ((double)(sizes[i] * 4) / copy_times[i]) / 1073741824.0;
        printf("%u\t%f\t%f\t%f\n", sizes[i], malloc_times[i], copy_times[i], bandwidth);
    }
    printf("\n");

    free(array);
    free(malloc_times);
    free(copy_times);
    cudaFree(d_array);

    return 0;
}
