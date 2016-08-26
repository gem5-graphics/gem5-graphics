#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#define TPB 32               // Threads per block
unsigned elts_per_dim;       // Array dimension is elts_per_dimxelts_per_dim
unsigned pitched_elements_per_row;
unsigned threadblocks_per_row;
size_t pitch;
bool debug_output;

//load element from da to db to verify correct memcopy
__global__ void kernel(float *da, float *db, unsigned *indices,
                       unsigned elts_per_dim,
                       unsigned pitched_elements_per_row,
                       unsigned threadblocks_per_row,
                       bool debug_indices) {
    unsigned aindex = blockDim.x * blockIdx.x + threadIdx.x;
    if (debug_indices) {
        indices[aindex] = aindex;
    }
    if(aindex % pitched_elements_per_row < elts_per_dim &&
       aindex < pitched_elements_per_row * elts_per_dim) {
        unsigned row = aindex / elts_per_dim;
        unsigned col = aindex % elts_per_dim;
        unsigned bindex = row*elts_per_dim + col;
        db[bindex] = da[aindex];
    }
}

void verify(float * A, float * B, int size);
void init(float * array, int size);

int main(int argc, char * argv[]) {
    bool debug_indices = false;
    debug_output = false;
    elts_per_dim = 184;
    if (argc > 1) {
        for (int index = 1; index < argc; ++index) {
            if (strcmp(argv[index], "-w") == 0) {
                if (argc > index+1) {
                    elts_per_dim = atoi(argv[++index]);
                } else {
                    printf("ERROR: Must specify row elements to -w option\n");
                    exit(0);
                }
            } else if (strcmp(argv[index], "-d") == 0) {
                debug_indices = true;
            } else if (strcmp(argv[index], "-o") == 0) {
                debug_output = true;
            }
        }
    }
    if (elts_per_dim < TPB) {
        fprintf(stderr, "WARNING: Fewer elements per row than threads per block...\n"
               "Setting number of elements to threads per block!\n");
        elts_per_dim = TPB;
    }
    printf("Elements Per Dimension = %u\n", elts_per_dim);

    float *A, *dA, *B, *dB;
    A = (float*)malloc(sizeof(float)*elts_per_dim*elts_per_dim);
    B = (float*)malloc(sizeof(float)*elts_per_dim*elts_per_dim);

    init(A, elts_per_dim*elts_per_dim);
    cudaMallocPitch(&dA, &pitch, sizeof(float)*elts_per_dim, elts_per_dim);
    printf("cudaMallocPitch Returned Pitch = %uB\n", (unsigned)pitch);
    pitched_elements_per_row = pitch / sizeof(float);
    if (pitched_elements_per_row < TPB) {
        threadblocks_per_row = 1;
    } else {
        threadblocks_per_row = pitched_elements_per_row / TPB;
        threadblocks_per_row += (pitched_elements_per_row % TPB) ? 1 : 0;
    }
    printf("Pitched Elements Per Row = %u\n", pitched_elements_per_row);
    printf("Threadblocks Per Row = %u\n", threadblocks_per_row);
    cudaMalloc(&dB, sizeof(float)*elts_per_dim*elts_per_dim);

    unsigned total_elements = pitched_elements_per_row * elts_per_dim;
    unsigned threads_per_block = TPB;
    unsigned blocks_per_grid = total_elements / threads_per_block;
    blocks_per_grid += (total_elements % threads_per_block) ? 1 : 0;
    unsigned total_threads = threads_per_block * blocks_per_grid;
    printf("Threads Per Block = %u\n", threads_per_block);
    printf("Blocks = %u\n", blocks_per_grid);

    unsigned *indices, *dIndices;
    if (debug_indices) {
        indices = (unsigned*)malloc(sizeof(unsigned)*total_threads);
        cudaMalloc(&dIndices, sizeof(unsigned)*total_threads);
    }

    // copy memory from unpadded array A of elts_per_dim by elts_per_dim dimensions
    // to more efficient dimensions of pitch by pitch on the device
    cudaMemcpy2D(dA, pitch, A, sizeof(float)*elts_per_dim, sizeof(float)*elts_per_dim, elts_per_dim, cudaMemcpyHostToDevice);
    kernel<<<blocks_per_grid,threads_per_block>>>(dA, dB, dIndices, elts_per_dim, pitched_elements_per_row, threadblocks_per_row, debug_indices);
    cudaMemcpy(B, dB, sizeof(float)*elts_per_dim*elts_per_dim, cudaMemcpyDeviceToHost);
    if (debug_indices) {
        cudaMemcpy(indices, dIndices, sizeof(unsigned)*pitched_elements_per_row*elts_per_dim, cudaMemcpyDeviceToHost);
        for (unsigned i = 0; i < pitched_elements_per_row*elts_per_dim; ++i) {
            printf("indices[%u] = %u\n", i, indices[i]);
        }
    }
    verify(A, B, elts_per_dim*elts_per_dim);

    free(A);
    free(B);
    cudaFree(dA);
    cudaFree(dB);
}

void init(float * array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = i;
    }
}

void verify(float * A, float * B, int size) {
    bool output_correct = true;
    for (int i = 0; i < size; i++) {
        if (A[i] != B[i]) {
            if (debug_output) {
                printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, B[i]);
            }
            output_correct = false;
        }
    }
    if (output_correct) {
        printf("Correct!\n");
    } else {
        printf("FAILED!\n");
    }
}
