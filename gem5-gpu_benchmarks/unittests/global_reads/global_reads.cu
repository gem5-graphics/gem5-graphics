#include <cstring>
#include <ctime>
#include <iostream>

using namespace std;

#define REPEAT256(S) \
S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S S 

__global__ void setup_global(int **global, unsigned num_elements, unsigned stride) {
    unsigned num_iters = num_elements / 32;
    num_iters += (num_elements % num_iters) ? 1 : 0;
    unsigned id = threadIdx.x;
    int *end_array = (int*)&global[num_elements];
    int **ptr = &global[id];
    unsigned type_corrected_stride = stride * sizeof(int*) / sizeof(unsigned);
    for (unsigned i = 0; i < num_iters; i++) {
        if ((int*)ptr < end_array) {
            int *next_address = (int*)ptr + type_corrected_stride;
            if (next_address >= end_array)
                next_address -= num_elements;
            *ptr = next_address;
        }
        ptr += 32;
    }
}

__global__ void global_reads_opt(int iters, unsigned array_size, int **array, int **final_ptr, unsigned thread_stride) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned start_index = (thread_stride * id) % array_size;
    int *ptr = array[start_index];

    __syncthreads();

    for (int i = 0; i < iters; i++) {
        REPEAT256(ptr = *(int**)ptr;)
    }

    __syncthreads();

    final_ptr[id] = ptr;
}

__global__ void global_reads(int warmup, int iters, unsigned array_size, int **array, unsigned block_start_offset, unsigned *total_clocks, unsigned *start_clocks, int **final_ptr, unsigned thread_stride) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    start_clocks[id] = 0;
    total_clocks[id] = 0;
    unsigned start_index = (thread_stride * id + block_start_offset) % array_size;
    unsigned start_time, end_time, real_start_time;
    double total_time = 0.0;
    int *ptr = array[start_index];

    // Warmup the icache and dcache as necessary
    for (int i = 0; i < warmup; i++) {
        REPEAT256(ptr = *(int**)ptr;)
    }

    __syncthreads();

    real_start_time = clock();
    for (int i = 0; i < iters; i++) {
        start_time = clock();
        REPEAT256(ptr = *(int**)ptr;)
        end_time = clock();
        if (end_time < start_time) {
            total_time += ((double)(0xFFFFFFFF - (start_time - end_time)) / 1000.0);
        } else {
            total_time += ((double)(end_time - start_time) / 1000.0);
        }
    }

    __syncthreads();

    start_clocks[id] = real_start_time;
    total_clocks[id] = (unsigned) total_time;
    final_ptr[id] = ptr;
}

int main(int argc, char** argv) {
    clock_t start_timer, end_timer;
    int num_iterations = 8;
    unsigned num_elements = 2048;
    unsigned block_start_offset = 0;
    unsigned warp_stride = 16;
    unsigned thread_stride = 1;
    int num_threads = -1;
    int num_blocks = -1;
    int threads_per_block = -1;
    bool nice_output = false;
    bool register_optimized = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-b")) {
            if (i < argc) {
                num_blocks = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of blocks to '-b'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-e")) {
            if (i < argc) {
                num_elements = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of array elements to '-e'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-i")) {
            if (i < argc) {
                num_iterations = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of iterations to '-i'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-n")) {
            nice_output = true;
        } else if (!strcmp(argv[i], "-o")) {
            if (i < argc) {
                block_start_offset = atoi(argv[++i]);
            } else {
                cout << "Need to specify block offset to '-o'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-p")) {
            if (i < argc) {
                threads_per_block = atoi(argv[++i]);
            } else {
                cout << "Need to specify threads per block to '-p'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-r")) {
            register_optimized = true;
        } else if (!strcmp(argv[i], "-s")) {
            if (i < argc) {
                thread_stride = atoi(argv[++i]);
            } else {
                cout << "Need to specify thread stride to '-s'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-t")) {
            if (i < argc) {
                num_threads = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of threads to '-t'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-w")) {
            if (i < argc) {
                warp_stride = atoi(argv[++i]);
            } else {
                cout << "Need to specify warp stride to '-w'\n";
                exit(-1);
            }
        }
    }

    // Setup blocks and threads
    if (num_threads < 0) {
        if (num_blocks < 0) {
            num_blocks = 1;
            if (threads_per_block < 0) {
                threads_per_block = 1;
            }
            num_threads = num_blocks * threads_per_block;
        } else {
            if (threads_per_block < 0) {
                threads_per_block = 1;
            }
            num_threads = num_blocks * threads_per_block;
        }
    } else {
        if (num_blocks < 0) {
            if (threads_per_block < 0) {
                threads_per_block = 32;
            }
            num_blocks = num_threads / threads_per_block;
            num_threads = num_blocks * threads_per_block;
        } else {
            if (threads_per_block < 0) {
                threads_per_block = num_threads / num_blocks;
                num_threads = num_blocks * threads_per_block;
            } else {
                if (num_blocks * threads_per_block != num_threads) {
                    cout << "WARNING: Your math is wrong, fixing it up\n";
                    threads_per_block = num_threads / num_blocks;
                    num_threads = num_blocks * threads_per_block;
                }
            }
        }
    }

    // Host data and pointers
    unsigned *start_clocks = new unsigned[num_threads];
    unsigned *total_clocks = new unsigned[num_threads];
    int **final_ptr = new int*[num_threads];

    // Device data and pointers
    int **d_global;
    unsigned *d_total_clocks, *d_start_clocks;
    int **d_final_ptr;
    cudaMalloc(&d_global, num_elements * sizeof(int*));
    cudaMalloc(&d_start_clocks, num_threads * sizeof(unsigned));
    cudaMalloc(&d_total_clocks, num_threads * sizeof(unsigned));
    cudaMalloc(&d_final_ptr, num_threads * sizeof(int*));

    setup_global<<<1, 32>>>(d_global, num_elements, warp_stride);
    cudaThreadSynchronize();

    start_timer = std::clock();

    if (!register_optimized) {
        global_reads<<<num_blocks, threads_per_block>>>(3, num_iterations, num_elements, d_global, block_start_offset, d_total_clocks, d_start_clocks, d_final_ptr, thread_stride);
    } else {
        global_reads_opt<<<num_blocks, threads_per_block>>>(num_iterations, num_elements, d_global, d_final_ptr, thread_stride);
    }

    cudaThreadSynchronize();

    end_timer = std::clock();

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "ERROR: Kernel execution failed with code: " << err
             << ", message: " << cudaGetErrorString(err) << endl;
        exit(-1);
    }

    cudaMemcpy(start_clocks, d_start_clocks, num_threads * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(total_clocks, d_total_clocks, num_threads * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(final_ptr, d_final_ptr, num_threads * sizeof(int*), cudaMemcpyDeviceToHost);

    unsigned min_kernel_time = (unsigned)0xffffffff;
    unsigned max_kernel_time = 0;
    for (int i = 0; i < num_threads; i++) {
        if (total_clocks[i] < min_kernel_time) min_kernel_time = total_clocks[i];
        if (total_clocks[i] > max_kernel_time) max_kernel_time = total_clocks[i];
    }

    unsigned overall_kernel_time = end_timer - start_timer;

    if (!nice_output) {
        cout << "Number of blocks = " << num_blocks << endl;
        cout << "Threads per block = " << threads_per_block << endl;
        cout << "Number of threads = " << num_threads << endl;
        cout << "Stride within warp (B) = " << thread_stride * sizeof(int*) << endl;
        cout << "Stride between loads (B) = " << warp_stride * sizeof(int*) << endl;
        cout << "Number of iterations = " << num_iterations << endl;
        cout << "Number of array elements = " << num_elements << endl;
        cout << "Array size (B) = " << num_elements * sizeof(int*) << endl;
        cout << "Total kernel time = " << overall_kernel_time << endl;
        cout << "Min kernel time = " << min_kernel_time << endl;
        cout << "Max kernel time = " << max_kernel_time << endl;
        cout << "Per thread timings:\n";
        for (int i = 0; i < num_threads; i++) {
            cout << "  " << i
                 << ": start = " << start_clocks[i]
                 << ", total = " << total_clocks[i]
                 << ", per = " << ((double)total_clocks[i] * 1000.0 / (double)(256.0 * num_iterations))
                 << ", ptr = " << final_ptr[i]
                 << endl;
        }
    } else {
        float bandwidth = ((num_iterations * num_threads * sizeof(int*) * 256) /
                          (((float)overall_kernel_time) / 1000000.0)) /
                          (1024*1024*1024);
        cout << num_iterations << ", "
             << num_threads << ", "
             << num_blocks << ", "
             << threads_per_block << ", "
             << (num_elements * sizeof(int*)) << ", "
             << (thread_stride * sizeof(int*)) << ", "
             << (warp_stride * sizeof(int*)) << ", "
             << overall_kernel_time << ", "
             << min_kernel_time << ", "
             << max_kernel_time << ", "
             << bandwidth
             << endl;
    }

    return 0;
}
