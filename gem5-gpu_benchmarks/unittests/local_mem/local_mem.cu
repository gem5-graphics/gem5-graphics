#include <iostream>

using namespace std;

#define NUM_ELTS 128
#define MAGIC 1234


// This function passes arguments and the return value on the GPU stack (local
// memory). By calling it, this tests pushing a call stack to local memory.
__device__ float d_randu(int *seed, int index)
{

    int M = INT_MAX;
    int A = 1103515245;
    int C = 12345;
    int num = A * seed[index] + C;
    seed[index] = num % M;
    num = seed[index];
    return fabs(num / ((float) M));
}

__device__ float d_randn(int *seed, int index)
{
    //Box-Muller algortihm
    float pi = 3.14159265358979323846;
    float u = d_randu(seed, index);
    float v = d_randu(seed, index);
    float cosine = cos(2 * pi * v);
    float rt = -2 * log(u);
    return sqrt(rt) * cosine;
}

__device__ unsigned mangle_start(unsigned unique_id, unsigned start) {
    unsigned my_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (my_tid != unique_id) {
        unique_id = my_tid;
    }
    return sqrt((float)start) + unique_id;
}

// This function contains a local memory array called copy_array. By increasing
// the NUM_ELTS variable above about 32, the compiler will allocate local
// memory for the array
__global__ void kernel(unsigned *array, unsigned start) {
    unsigned total_threads = gridDim.x * blockDim.x;
    unsigned unique_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned copy_array[NUM_ELTS];
    copy_array[0] = mangle_start(unique_id, start);
    for (unsigned i = 0; i < NUM_ELTS - 1; i++) {
        copy_array[i+1] = copy_array[i] + 1;
    }
    __syncthreads();
    for (unsigned i = 0; i < NUM_ELTS; i++) {
        array[i * total_threads + unique_id] = 2 * copy_array[i];
    }
}


int main(int argc, char** argv) {
    unsigned num_blocks = 1;
    unsigned threads_per_block = 1;
    for (unsigned i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-b")) {
            if (i < argc) {
                num_blocks = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of blocks to '-b'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-t")) {
            if (i < argc) {
                threads_per_block = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of threads to '-t'\n";
                exit(-1);
            }
        }
    }

    unsigned total_threads = num_blocks * threads_per_block;

	unsigned *global = new unsigned[NUM_ELTS * total_threads];
	unsigned *d_global;
	cudaMalloc(&d_global, NUM_ELTS * total_threads * sizeof(unsigned));

	kernel <<<num_blocks, threads_per_block>>> (d_global, MAGIC);

    cudaThreadSynchronize();
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "ERROR: Kernel execution failed with code: " << err
             << ", message: " << cudaGetErrorString(err) << endl;
        exit(-1);
    }

    cudaMemcpy(global, d_global, NUM_ELTS * total_threads * sizeof(unsigned), cudaMemcpyDeviceToHost);

	unsigned num_incorrect = 0;
    for (unsigned i = 0; i < NUM_ELTS; i++) {
        unsigned value = 2 * (sqrt((float)MAGIC) + i);
        for (unsigned thd = 0; thd < total_threads; thd++) {
            if (global[i * total_threads + thd] != value + 2 * thd) {
                // cout << i << ":" << thd << " = " << global[i * total_threads + thd] << " should be " << (value + 2 * thd) << endl;
                num_incorrect++;
            }
        }
    }

	if (num_incorrect == 0) {
		cout << "Test Passed" << endl;
	} else {
		cout << "Test Failed, " << num_incorrect << " incorrect" << endl;
	}
	return 0;
}
