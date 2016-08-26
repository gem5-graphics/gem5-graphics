#include <stdio.h>
#include <cassert>
#include <sys/time.h>

#define LINE_SIZE_INTS 32       // 32 ints per 128B cache line
#define MAX_ARRAY_SIZE 4*1024*1024     // 16MB max array (4B ints)

__global__ void kernel(unsigned *array, unsigned size, unsigned *sums, unsigned threads_per_block, unsigned refs_per_thread, unsigned delay_count) {
    unsigned *end_array = array + size;
    unsigned unique = blockIdx.x * threads_per_block + threadIdx.x;
    unsigned start_index = unique * LINE_SIZE_INTS * refs_per_thread;
    unsigned sum = 0;
    unsigned *array_ptr = &array[start_index];
    for (unsigned i = 0; i < refs_per_thread; ++i) {
        while (array_ptr >= end_array) {
            array_ptr -= size;
        }
        sum += *array_ptr;
        array_ptr += LINE_SIZE_INTS;
        if (delay_count > 0) {
            unsigned subsum = 0;
            for (unsigned j = 0; j < delay_count; ++j) {
                subsum += 1;
            }
            subsum -= delay_count;
            sum += subsum;
        }
    }
    sums[unique] = sum;
}

#ifndef NO_TIMING
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}
#endif

int main(int argc, char** argv) {

    // Setup default parameters
    unsigned threads_per_block = 512;
    unsigned num_refs_per_thread = 32;
    unsigned num_blocks = 16;
    unsigned delay_count = 12;

    // Parse command line params
    if (argc > 1) {
        printf("%s <threads_per_block> <num_gpu_blocks> <num_refs_per_thread> <delay_count>\n", argv[0]);
        threads_per_block = atoi(argv[1]);
        if (argc > 2)
            num_blocks = atoi(argv[2]);
        if (argc > 3)
            num_refs_per_thread = atoi(argv[3]);
        if (argc > 4)
            delay_count = atoi(argv[4]);
        assert(threads_per_block);
        assert(num_blocks);
        assert(num_refs_per_thread);
    }

    // Print parameters
    printf("Number of threads per GPU block: %u\n", threads_per_block);
    printf("Number of GPU blocks to execute: %u\n", num_blocks);
    printf("Number of memory references per GPU thread: %u\n", num_refs_per_thread);
    printf("Number of extra instructions between loads: %u\n", delay_count);

    // Setup grid and threads
    unsigned num_cache_lines = num_blocks * threads_per_block * num_refs_per_thread;
    unsigned array_size = num_cache_lines * LINE_SIZE_INTS;
    if (array_size > MAX_ARRAY_SIZE) {
        array_size = MAX_ARRAY_SIZE;
    }
    unsigned sum_size = num_cache_lines / num_refs_per_thread;
    assert(sum_size);
    dim3 grid(num_blocks, 1, 1);
    dim3 threads(threads_per_block, 1, 1);
    printf("Array size (B): %u\n", array_size * sizeof(unsigned));

    unsigned *d_array;
    unsigned *d_sums;

#ifndef NO_TIMER
    printf("Timer granularity ticks per second: %ld\n", CLOCKS_PER_SEC);
    long long* ticks = new long long[4];
    ticks[0] = get_time();
#endif

    cudaMalloc(&d_array, array_size * sizeof(unsigned));
    cudaMalloc(&d_sums, sum_size * sizeof(unsigned));
    cudaMemset(d_array, 1, array_size * sizeof(unsigned));

#ifndef NO_TIMER
    ticks[1] = get_time();
#endif

    kernel<<< grid, threads >>>(d_array, array_size, d_sums, threads_per_block, num_refs_per_thread, delay_count);

#ifndef NO_TIMER
    ticks[2] = get_time();
#endif

    unsigned *sums = new unsigned[sum_size];
    cudaMemcpy(sums, d_sums, sum_size * sizeof(unsigned), cudaMemcpyDeviceToHost);

#ifndef NO_TIMER
    ticks[3] = get_time();
#endif

    unsigned val = (((((1 << 8) + 1) << 8) + 1) << 8) + 1;
    printf("val = %u\n", val);
    unsigned sum_val = val * num_refs_per_thread;
    printf("sum_val = %u\n", sum_val);
    unsigned count_incorrect = 0;
    for (unsigned i = 0; i < sum_size; ++i) {
        if (sums[i] != sum_val) {
            count_incorrect++;
        }
    }
    if (count_incorrect) {
        printf("Number incorrect sums = %u\n", count_incorrect);
    } else {
        printf("ALL SUMS CORRECT!\n");
    }

#ifndef NO_TIMER
    printf("Timing:\n");
    printf("Mallocs + Memset = %lld\n", ticks[1] - ticks[0]);
    printf("Kernel = %lld\n", ticks[2] - ticks[1]);
    printf("Memcpy = %lld\n", ticks[3] - ticks[2]);
    delete[] ticks;
#endif

    delete[] sums;
    return 0;
}
