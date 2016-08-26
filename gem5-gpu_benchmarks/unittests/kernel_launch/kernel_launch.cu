
#include <iostream>
#include <cstdlib>

#define BLOCKS 32
#define THREADS 256

using namespace std;

__global__ void kernel(int *input, int *output)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    output[idx] = input[idx];
}

__global__ void shmKernel(int *input, int *output)
{
    __shared__ int buffer0[BLOCKS*THREADS/4];
    __shared__ int buffer1[BLOCKS*THREADS/4];
    __shared__ int buffer2[BLOCKS*THREADS/4];
    __shared__ int buffer3[BLOCKS*THREADS/4];

    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    int offset = 0;
    switch (idx%4) {
    case 0:
        offset = idx-BLOCKS*THREADS/4*0;
        buffer0[offset] = input[idx];
        break;
    case 1:
        offset = idx-BLOCKS*THREADS/4*1;
        buffer1[offset] = input[idx];
        break;
    case 2:
        offset = idx-BLOCKS*THREADS/4*2;
        buffer2[offset] = input[idx];
        break;
    case 3:
        offset = idx-BLOCKS*THREADS/4*3;
        buffer3[offset] = input[idx];
        break;
    }

    switch (idx%4) {
    case 0:
        offset = idx-BLOCKS*THREADS/4*0;
        output[idx] = buffer0[offset];
        break;
    case 1:
        offset = idx-BLOCKS*THREADS/4*1;
        output[idx] = buffer1[offset];
        break;
    case 2:
        offset = idx-BLOCKS*THREADS/4*2;
        output[idx] = buffer2[offset];
        break;
    case 3:
        offset = idx-BLOCKS*THREADS/4*3;
        output[idx] = buffer3[offset];
        break;
    }
}

int main(int argc, char* argv[])
{
    int launches;
    bool sharedMem;

    if (argc != 3) {
        cout << "Usage: kernel_launch <num launches> <0|1 (shared mem)>" << endl;
    } else {
        launches = atoi(argv[1]);
        sharedMem = atoi(argv[2]);
    }

    int *input, *output;
    int *dinput, *doutput;

    input = (int*)malloc(sizeof(int)*BLOCKS*THREADS);
    output = (int*)malloc(sizeof(int)*BLOCKS*THREADS);

    cudaMalloc(&dinput, sizeof(int)*BLOCKS*THREADS);
    cudaMalloc(&doutput, sizeof(int)*BLOCKS*THREADS);

    for (int i=0; i<BLOCKS*THREADS; i++) {
        input[i] = i;
        output[i] = i+4000;
    }

    cudaMemcpy(dinput, input, sizeof(int)*BLOCKS*THREADS, cudaMemcpyHostToDevice);
    cudaMemcpy(doutput, output, sizeof(int)*BLOCKS*THREADS, cudaMemcpyHostToDevice);

    for (int i=0; i<launches; i++) {
        if (!sharedMem) {
            kernel<<<BLOCKS,THREADS>>>(dinput, doutput);
        } else {
            shmKernel<<<BLOCKS,THREADS>>>(dinput, doutput);
        }
    }

    cudaMemcpy(output, doutput, sizeof(int)*BLOCKS*THREADS, cudaMemcpyDeviceToHost);

    for (int i=0; i<BLOCKS*THREADS; i+=rand()%100) {
        if (output[i] != i) {
            cerr << "Error at " << i << " got " << output[i] << endl;
            return -1;
        }
    }

    cout << "Launch test passed" << endl;

    return 0;
}
