#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


texture<float, 2, cudaReadModeElementType> tex;

void runTest( int argc, char** argv);

int main(int argc, char** argv){
    runTest( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    float h_data[] = {0, 1, 2, 3};
    int height,width;
    height = width = 2;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cu_array;
    cudaMallocArray( &cu_array, &channelDesc, width, height);
    //printf("cu_array width=%d, height=%d, dims=% \n", cu_array->width, cu_array->height, cu_array->dims);
    cudaMemcpy2DToArray( cu_array, 0, 0, h_data, width*sizeof(float)*2, width*sizeof(float)*2, height, cudaMemcpyHostToDevice);

    // set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates

    printf("tex=%llx, cu_array=%llx\n",  &tex, cu_array);
    cudaBindTextureToArray( tex, cu_array, channelDesc);
}
