/*****************************************************************************
 * Test cudaMemcpy functionality, including copying over portions of arrays
 * that have not yet been written to, only allocated.
 *
 * TODO: Add cudaMemcpy device-to-device, host-to-device and
 * cudaMemcpyFromSymbol.
 ****************************************************************************/

#include <stdio.h>
#include <sys/time.h>

#define REFS_PER_THREAD 2
#define LINE_SIZE_INTS 32       // 32 ints per 128B cache line
#define THREADS_PER_BLOCK 16
#define ARRAY_SIZE_BREAK_ME 8192
#define SUM_SIZE_BREAK_ME 2112
__constant__ unsigned int refs_per_thread = REFS_PER_THREAD;
__constant__ unsigned int d_array[ARRAY_SIZE_BREAK_ME];

__global__ void kernel(unsigned int *sums) {
    unsigned int unique = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    unsigned int start_index = unique * LINE_SIZE_INTS * refs_per_thread;
    unsigned int sum = 0;
    unsigned int *array_ptr = &d_array[start_index];
    for (unsigned int i = 0; i < refs_per_thread; ++i) {
        sum += *array_ptr;
        array_ptr += LINE_SIZE_INTS;
    }
    sums[unique] = sum;
}

int main() {
    // Setup grid and threads
    unsigned int num_blocks = 4;
    unsigned int num_cache_lines = num_blocks * THREADS_PER_BLOCK * REFS_PER_THREAD;
    unsigned int array_size = num_cache_lines * LINE_SIZE_INTS;
    unsigned int sum_size = (num_cache_lines / REFS_PER_THREAD);
    dim3 grid(num_blocks, 1, 1);
    dim3 threads(THREADS_PER_BLOCK, 1, 1);
    printf("array_size = %d\n", array_size);
    printf("ARRAY_SIZE_BREAK_ME = %d\n", ARRAY_SIZE_BREAK_ME);
    printf("sum_size = %d\n", sum_size);
    printf("SUM_SIZE_BREAK_ME = %d\n", SUM_SIZE_BREAK_ME);

    unsigned int *d_sums;
    cudaMalloc(&d_sums, SUM_SIZE_BREAK_ME * sizeof(unsigned int));
    unsigned int *array = new unsigned int[ARRAY_SIZE_BREAK_ME];
    unsigned int *sums = new unsigned int[SUM_SIZE_BREAK_ME];
    printf("array ptr: %x\n", array);
    printf("d_array ptr: %x\n", d_array);
    printf("sums ptr: %x\n", sums);
    printf("d_sums ptr: %x\n", d_sums);
    memset(array, 1, array_size * sizeof(unsigned int));
    cudaMemcpyToSymbol(d_array, array, ARRAY_SIZE_BREAK_ME * sizeof(unsigned int));

    kernel<<< grid, threads >>>(d_sums);

    cudaMemcpy(sums, d_sums, SUM_SIZE_BREAK_ME * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int val = (((((1 << 8) + 1) << 8) + 1) << 8) + 1;
    printf("val = %u\n", val);
    unsigned int sum_val = val * REFS_PER_THREAD;
    printf("sum_val = %u\n", sum_val);
    unsigned int count_incorrect = 0;
    for (unsigned int i = 0; i < sum_size; ++i) {
        if (sums[i] != sum_val) {
            count_incorrect++;
        }
    }
    if (count_incorrect) {
        printf("Number incorrect sums = %u\n", count_incorrect);
    } else {
        printf("All sums correct!\n");
    }

    delete[] sums;
    return 0;
}
