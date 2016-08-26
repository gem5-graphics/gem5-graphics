#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define BLOCK_SIZE 8

#define STR_SIZE 256

#define DEVICE 1


#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

void run(int argc, char** argv);


int J, K, L;
int* data;
int** space2D;
int*** space3D;
#define M_SEED 9
int pyramid_height;
int timesteps;

int bornMin = 5, bornMax = 8;
int dieMax = 3, dieMin = 10;

void
init(int argc, char** argv)
{
	if(argc==6){
		J = atoi(argv[1]);
		K = atoi(argv[2]);
                L = atoi(argv[3]);
                timesteps = atoi(argv[4]);
                pyramid_height=atoi(argv[5]);
	}else{
                printf("Usage: cell dim3 dim2 dim1 timesteps pyramid_height\n");
                exit(0);
        }
	data = new int[J*K*L];
        space2D = new int*[J*K];
	space3D = new int**[J];
	for(int n=0; n<J*K; n++)
          space2D[n]=data+L*n;
	for(int n=0; n<J; n++)
          space3D[n]=space2D+K*n;

	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < J*K*L; i++)
            data[i] = rand()%2;

}

void
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))
#define EXPAND_RATE 2 // add one iteration will extend the pyramid base by 2 per each borderline
__global__ void evolve(
                int iteration,
                int *gpuDataSrc,
                int *gpuDataDst,
                int J, int K, int L,
                int bornMin, int bornMax,
                int dieMin, int dieMax,
                int blockL, int border)
{
        __shared__ int data[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int result[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

	int bj = blockIdx.x;
	int bk = blockIdx.y/blockL;
	int bl = blockIdx.y%blockL;
	int tj=threadIdx.x;
	int tk=threadIdx.y;
	int tl=threadIdx.z;

        // each block finally computes result for a small block
        // after N iterations.
        // it is the non-overlapping small blocks that cover
        // all the input data

        // calculate the small block size
	int small_block = BLOCK_SIZE-iteration*EXPAND_RATE;

        // calculate the boundary for the block according to
        // the boundary of its small block
        int blkJ = small_block*bj-border;
        int blkK = small_block*bk-border;
        int blkL = small_block*bl-border;

        int blkJmax = blkJ+BLOCK_SIZE-1;
        int blkKmax = blkK+BLOCK_SIZE-1;
        int blkLmax = blkL+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int idxJ = blkJ+tj;
	int idxK = blkK+tk;
	int idxL = blkL+tl;

        int index = idxL+L*(idxK+idxJ*K);

	if(IN_RANGE(idxJ, 0, J-1) &&
          IN_RANGE(idxK, 0, K-1) &&
          IN_RANGE(idxL, 0, L-1) ){
            data[tj][tk][tl] = gpuDataSrc[index];
	}

        // effective range within this block that falls within
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validJmin = (blkJ < 0) ? -blkJ : 0;
        int validJmax = (blkJmax > J-1) ? BLOCK_SIZE-1-(blkJmax-J+1) : BLOCK_SIZE-1;
        int validKmin = (blkK < 0) ? -blkK : 0;
        int validKmax = (blkKmax > K-1) ? BLOCK_SIZE-1-(blkKmax-K+1) : BLOCK_SIZE-1;
        int validLmin = (blkL < 0) ? -blkL : 0;
        int validLmax = (blkLmax > L-1) ? BLOCK_SIZE-1-(blkLmax-L+1) : BLOCK_SIZE-1;

        int isValid =(IN_RANGE(tj, validJmin, validJmax) &&
                      IN_RANGE(tk, validKmin, validKmax) &&
                      IN_RANGE(tl, validLmin, validLmax));

        bool computed=false;
        for (int i=0; i<iteration ; i++){
            computed = false;
            int inLayer =   IN_RANGE(tj, i+1, BLOCK_SIZE-i-2) &&
                            IN_RANGE(tk, i+1, BLOCK_SIZE-i-2) &&
                            IN_RANGE(tl, i+1, BLOCK_SIZE-i-2);

            if( isValid && inLayer){
                  computed = true;
                  int sum=0;
                  int orig = data[tj][tk][tl];
                  int a, b, c;
                  for(int j = tj-1; j <= tj+1; j++){
                      for(int k = tk-1; k <= tk+1; k++){
                          for(int l = tl-1; l<=tl+1; l++){
                              a = j; b = k; c = l;
                              CLAMP_RANGE(a, validJmin, validJmax);
                              CLAMP_RANGE(b, validKmin, validKmax);
                              CLAMP_RANGE(c, validLmin, validLmax);
                              sum += data[a][b][c];
                          }
                      }
                  }
                  sum -= orig;
                  if(orig>0 && (sum <= dieMax || sum >= dieMin))
                      result[tj][tk][tl]=0;
                  else if(orig==0 && (sum >= bornMin && sum <= bornMax))
                      result[tj][tk][tl]=1;
                  else
                      result[tj][tk][tl]=orig;
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                data[tj][tk][tl]= result[tj][tk][tl];
            __syncthreads();
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuDataDst[index]=result[tj][tk][tl];
      }
}

/*
   compute N time steps
*/
int simulate(int *gpuData[2],
              int J, int K, int L,
              int timesteps,
              int pyramid_height,
              int blockJ, int blockK, int blockL,
              int border)
{
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
        // TODO: why 3D grid doesn't work??  blockL = 1;
        dim3 dimGrid(blockJ, blockK*blockL);

        int src = 1, dst = 0;
	for (int t = 0; t < timesteps; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
            evolve<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, timesteps-t),
                gpuData[0], gpuData[1], J, K, L,
                bornMin, bornMax,
                dieMin,  dieMax,
                blockL, border);
	}
        return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    init(argc, argv);

    run(argc,argv);

    delete [] data;
    delete [] space2D;
    delete [] space3D;

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int border = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockEdge = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockJ = J/smallBlockEdge+((J%smallBlockEdge==0)?0:1);
    int blockK = K/smallBlockEdge+((K%smallBlockEdge==0)?0:1);
    int blockL = L/smallBlockEdge+((L%smallBlockEdge==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d, %d, %d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d, %d, %d]\ntargetBlock:[%d]\n",\
	pyramid_height, J, K, L, border, BLOCK_SIZE, blockJ, blockK, blockL, smallBlockEdge);

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    int *gpuData[2];
    int size = J*K*L;
    // cudaMalloc((void**)&gpuData[0], sizeof(int)*size);
    // cudaMalloc((void**)&gpuData[1], sizeof(int)*size);
    // cudaMemcpy(gpuData[0], data, sizeof(int)*size, cudaMemcpyHostToDevice);
    gpuData[0] = data;
    gpuData[1] = new int[J*K*L];


    int ret = simulate(gpuData, J, K, L, timesteps,
	 pyramid_height, blockJ, blockK, blockL,
         border);

    //cudaMemcpy(data, gpuData[ret], sizeof(int)*size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    data = gpuData[ret];

    // free the other one
    delete [] gpuData[ret?0:1];

    //cudaFree(gpuData[0]);
    //cudaFree(gpuData[1]);
}

