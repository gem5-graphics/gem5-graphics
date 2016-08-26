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
                                  unsigned num_writes_per_warp,
                                  unsigned *global_first_array,
                                  unsigned *global_second_array,
                                  unsigned *threads_complete) {
    // Write ascending values to separate memory locations with or without
    // memory fences between the writes. We want to mix up store locality as a
    // way to cause both L2 hits and misses to shake up the timings/orderings.
    // To do this, use a small GPU L2 cache, and numerous threads.
    unsigned tid = num_writes_per_warp * blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iterations; i++) {
        if (i % 50 == 0) {
            // Periodically sync, so that no-fence version does not allow
            // some warps to run far ahead. This increases the likelihood of
            // detecting consistency races by having many separate warps
            // concurrently issuing memory accesses.
            __syncthreads();
        }
        global_first_array[tid] = i;
        __fencethreads();
        global_second_array[tid] = i;
    }
    // Count up the complete threads to signal to the CPU when the GPU is done
    // sending all stores.
    atomicInc(threads_complete, 2 * blockDim.x * gridDim.x);
}

__global__ void clear_array(unsigned num_writes_per_warp,
                            unsigned *global_first_array,
                            unsigned *global_second_array) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < blockDim.x * gridDim.x * num_writes_per_warp) {
        global_first_array[tid] = 0;
        global_second_array[tid] = 0;
        tid += blockDim.x * gridDim.x;
    }
}

typedef struct {
    unsigned thread_id;
    unsigned *first_array;
    unsigned *second_array;
    unsigned *threads_complete;
    unsigned threads_to_complete;
    unsigned num_writes_per_warp;
} consistency_check_params;

void set_params(consistency_check_params *params, int _tid,
                unsigned *_first_array, unsigned *_second_array,
                unsigned *_threads_complete, unsigned _thread_count,
                unsigned _num_writes_per_warp) {
    params->thread_id = _tid;
    params->first_array = _first_array;
    params->second_array = _second_array;
    params->threads_complete = _threads_complete;
    params->threads_to_complete = _thread_count;
    params->num_writes_per_warp = _num_writes_per_warp;
}

void *check_consistency(void *arg) {
    consistency_check_params *mine = (consistency_check_params*)arg;
    unsigned tid = mine->thread_id;
    unsigned *first_array = mine->first_array;
    unsigned *second_array = mine->second_array;
    unsigned *threads_complete = mine->threads_complete;
    unsigned num_writes_per_warp = mine->num_writes_per_warp;
    unsigned num_to_complete = mine->threads_to_complete * num_writes_per_warp;
    bool done = false;
    // Randomize stride through array to shake up which chunks of the array are
    // pulled to and from CPU-side cache hierarchy
    unsigned random_stride = 3137 * num_writes_per_warp;
    unsigned num_tests = 0;
    unsigned num_races_detected = 0;
    while (!done) {
        // Fence to ensure complete count is re-read
        asm volatile("lfence" ::: "memory");
        // If not yet complete, randomly select an array index to start reading
        // and checking data from
        unsigned index = 4;
        for (int i = 0; i < 50; i++) {
            unsigned second_val = second_array[index];
            asm volatile("lfence" ::: "memory");
            unsigned first_val = first_array[index];
            if (second_val > first_val) {
                num_races_detected++;
            }
            index += random_stride;
            index %= num_to_complete;
            num_tests++;
        }
        // Check the GPU's number of completed threads, and if they're all
        // complete, exit the loop.
        if (*threads_complete == num_to_complete) done = true;
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
    unsigned num_writes_per_warp = 1;

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
        } else if (!strcmp(argv[i], "-w")) {
            if (i < argc) {
                num_writes_per_warp = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of writes per warp to '-w'\n";
                exit(-1);
            }
        }
    }

    cout << "Number of GPU iterations: " << num_iterations << endl;
    cout << "Number of thread blocks: " << num_blocks << endl;
    cout << "Threads per block: " << threads_per_block << endl;
    unsigned total_threads = num_blocks * threads_per_block;

    unsigned *global_first_array;
    unsigned *global_second_array;
    cudaMalloc((void**)&global_first_array, num_writes_per_warp * total_threads * sizeof(unsigned));
    cudaMalloc((void**)&global_second_array, num_writes_per_warp * total_threads * sizeof(unsigned));

    unsigned *thread_complete_count_var;
    cudaMalloc((void**)&thread_complete_count_var, sizeof(unsigned));
    *thread_complete_count_var = 0;

    unsigned *threads_complete;
    cudaMalloc((void**)&threads_complete, sizeof(unsigned));
    cudaMemset(threads_complete, 0, sizeof(unsigned));

    clear_array<<<num_blocks, threads_per_block>>>(num_writes_per_warp,
                                                   global_first_array,
                                                   global_second_array);

    consistency_tests<<<num_blocks, threads_per_block>>>(num_iterations,
                                                         num_writes_per_warp,
                                                         global_first_array,
                                                         global_second_array,
                                                         threads_complete);

#ifdef PTHREADS
    // Spin up many CPU threads to check the data being produced by the GPU
    pthread_t *threads = new pthread_t[num_cpu_threads];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 4096);
    consistency_check_params *params = new consistency_check_params[num_cpu_threads];
    float *final_vals = new float[num_cpu_threads];
    for (unsigned tid = 0; tid < num_cpu_threads; tid++) {
        set_params(&params[tid], tid, global_first_array, global_second_array, threads_complete, total_threads, num_writes_per_warp);
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
    set_params(&params, 0, global_first_array, global_second_array, threads_complete, total_threads, num_writes_per_warp);
    check_consistency((void*)&params);

#endif

    cudaThreadSynchronize();
    return 0;
}
