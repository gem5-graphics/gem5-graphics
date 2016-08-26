#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#ifdef GEM5_WORK
#include <stdint.h>
extern "C" {
    void m5_dumpreset_stats(uint64_t ns_delay, uint64_t ns_period);
    void m5_work_begin(uint64_t workid, uint64_t threadid);
    void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#ifdef TIMING
#include <sys/time.h>
double gettime() {
    struct timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec+t.tv_usec*1e-6;
}
#endif

void run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9

void
init(int argc, char** argv)
{
    if(argc == 3){
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
    } else {
        printf("Usage: pathfinder width num_of_steps\n");
        exit(0);
    }
    data = new int[rows*cols];
    wall = new int*[rows];
    for(int n=0; n<rows; n++)
        wall[n]=data+cols*n;
    result = new int[cols];

    int seed = M_SEED;
    srand(seed);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
    for (int j = 0; j < cols; j++)
        result[j] = wall[0][j];
#ifdef OUTPUT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]) ;
        }
        printf("\n") ;
    }
#endif
}

void 
fatal(char *s)
{
    fprintf(stderr, "error: %s\n", s);
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

int main(int argc, char** argv)
{
    run(argc,argv);
    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    unsigned long long cycles;

    int *src, *dst, *temp;
    int min;

    dst = result;
    src = new int[cols];

#ifdef GEM5_WORK
    m5_work_begin(0, 0);
    m5_dumpreset_stats(0, 0);
#endif

#ifdef TIMING
    double start_time = gettime();
#endif

    for (int t = 0; t < rows-1; t++) {
        temp = src;
        src = dst;
        dst = temp;
        #pragma omp parallel for private(min)
        for(int n = 0; n < cols; n++){
            min = src[n];
            if (n > 0)
                min = MIN(min, src[n-1]);
            if (n < cols-1)
                min = MIN(min, src[n+1]);
            dst[n] = wall[t+1][n]+min;
        }
    }

#ifdef TIMING
    double end_time = gettime();
    printf("ROI Runtime: %f\n", end_time - start_time);
#endif

#ifdef GEM5_WORK
    m5_dumpreset_stats(0, 0);
    m5_work_end(0, 0);
#endif

#ifdef OUTPUT    for (int i = 0; i < cols; i++)            printf("%d ",data[i]);
    printf("\n") ;
    for (int i = 0; i < cols; i++)            printf("%d ",dst[i]);
    printf("\n") ;
#endif

    delete [] data;
    delete [] wall;
    delete [] dst;
    delete [] src;
}

