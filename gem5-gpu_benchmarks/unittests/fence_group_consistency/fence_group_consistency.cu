#include <cstring>
#include <ctime>
#include <iostream>

using namespace std;

#define NOFENCE 0
#define SYNCTHREADS 1
#define CTAMEMBAR 2
#define GLOBALMEMBAR 3
#define SYSTEMMEMBAR 4

#ifndef FENCE_TYPE
    #define FENCE_TYPE NO_FENCE
#endif

__device__ inline void
__ctamembar()
{
    asm("membar.cta;");
}

__device__ inline void
__globalmembar()
{
    asm("membar.gl;");
}

__device__ inline void
__systemmembar()
{
    asm("membar.sys;");
}

__device__ inline void
__fencethreads()
{
#if FENCE_TYPE == SYNCTHREADS
    __syncthreads();
#elif FENCE_TYPE == CTAMEMBAR
    __ctamembar();
#elif FENCE_TYPE == GLOBALMEMBAR
    __globalmembar();
#elif FENCE_TYPE == SYSTEMMEMBAR
    __systemmembar();
#else // NOFENCE

#endif
}

__global__ void consistency_tests(unsigned iterations,
                                  unsigned *global_first_array,
                                  unsigned *global_second_array) {
    // Write ascending values to separate memory locations with or without
    // memory fences between the writes. We want to mix up store locality as a
    // way to cause both L2 hits and misses to shake up the timings/orderings.
    // To do this, use a small GPU L2 cache, and numerous threads.
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iterations; i++) {
        global_first_array[tid] = i;
        __fencethreads();
        if (threadIdx.x == 0) {
            global_second_array[blockIdx.x] = i;
        }
    }
}

__global__ void clear_array(unsigned *global_first_array,
                            unsigned *global_second_array) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    global_first_array[tid] = 0;
    if (threadIdx.x == 0) {
        global_second_array[blockIdx.x] = 0;
    }
}

typedef struct {
    unsigned thread_id;
    unsigned *first_array;
    unsigned *second_array;
    unsigned threads_per_block;
    unsigned threads_to_complete;
    unsigned num_gpu_iters;
} consistency_check_params;

void set_params(consistency_check_params *params, int _tid,
                unsigned *_first_array, unsigned *_second_array,
                unsigned _threads_per_block, unsigned _thread_count,
                unsigned _num_gpu_iters) {
    params->thread_id = _tid;
    params->first_array = _first_array;
    params->second_array = _second_array;
    params->threads_per_block = _threads_per_block;
    params->threads_to_complete = _thread_count;
    params->num_gpu_iters = _num_gpu_iters;
}

void *check_consistency(void *arg) {
    consistency_check_params *mine = (consistency_check_params*)arg;
    unsigned tid = mine->thread_id;
    unsigned *first_array = mine->first_array;
    unsigned *second_array = mine->second_array;
    unsigned threads_per_block = mine->threads_per_block;
    unsigned lines_per_block = threads_per_block / 32;
    unsigned num_to_complete = mine->threads_to_complete;
    unsigned num_gpu_iters = mine->num_gpu_iters;
    bool done = false;
    unsigned line_stride = 32;
    unsigned num_tests = 0;
    unsigned num_races_detected = 0;
    unsigned index = 0;
    while (!done) {
        // Fence to ensure complete count is re-read
        asm volatile("" ::: "memory");
        unsigned second_val = second_array[index/threads_per_block];
        unsigned first_val;
        // Check the first value in each line to make sure it didn't race with
        // the block's completion of the i-th epoch
        for (int i = 0; i < lines_per_block; i++) {
            // The first array must get updated before the second array, so we
            // want to verify that all threads have first array value at least
            // equal to the second array value
            first_val = first_array[index];
            if (second_val > first_val) {
                num_races_detected++;
            }
            index += line_stride;
            num_tests++;
        }
        index %= num_to_complete;
        if (second_val == (num_gpu_iters - 1)) done = true;
    }
    cout << tid << ": All GPU threads complete! Races detected: " << num_races_detected << "/" << num_tests << endl;
    return NULL;
}

int main(int argc, char** argv) {
    // Variables
    unsigned num_blocks = 16;
    unsigned num_iterations = 500;
    unsigned num_cpu_threads = 1;
    unsigned threads_per_block = 256;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-b")) {
            if (i < argc) {
                num_blocks = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of blocks to '-b'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-i")) {
            if (i < argc) {
                num_iterations = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of iterations to '-i'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-t")) {
            if (i < argc) {
                threads_per_block = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of threads per block to '-t'\n";
                exit(-1);
            }
        }
    }

    cout << "Number of thread blocks: " << num_blocks << endl;
    cout << "Threads per block: " << threads_per_block << endl;
    unsigned total_threads = num_blocks * threads_per_block;

    unsigned *global_first_array;
    unsigned *global_second_array;
    cudaMalloc((void**)&global_first_array, total_threads * sizeof(unsigned));
    cudaMalloc((void**)&global_second_array, num_blocks * sizeof(unsigned));

    unsigned *thread_complete_count_var;
    cudaMalloc((void**)&thread_complete_count_var, sizeof(unsigned));
    *thread_complete_count_var = 0;

    clear_array<<<num_blocks, threads_per_block>>>(global_first_array, global_second_array);

    consistency_tests<<<num_blocks, threads_per_block>>>(num_iterations,
                                                         global_first_array,
                                                         global_second_array);

#ifdef PTHREADS
    // Spin up many CPU threads to check the data being produced by the GPU
    pthread_t *threads = new pthread_t[num_cpu_threads];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 4096);
    consistency_check_params *params = new consistency_check_params[num_cpu_threads];
    float *final_vals = new float[num_cpu_threads];
    for (unsigned tid = 0; tid < num_cpu_threads; tid++) {
        set_params(&params[tid], tid, global_first_array, global_second_array, threads_per_block, total_threads, num_iterations);
        pthread_create(&threads[tid], &attr, &check_consistency, &params[tid]);
    }

    void *ret_val;
    for (unsigned tid = 0; tid < num_cpu_threads; tid++) {
        pthread_join(threads[tid], &ret_val);
    }
    delete [] params;

#else

    // Use single CPU thread for reduction (without pthreads)
    consistency_check_params params;
    set_params(&params, 0, global_first_array, global_second_array, threads_per_block, total_threads, num_iterations);
    check_consistency((void*)&params);

#endif

    cudaThreadSynchronize();
    return 0;
}
