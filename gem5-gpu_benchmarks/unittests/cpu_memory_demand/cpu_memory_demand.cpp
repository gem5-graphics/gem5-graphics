#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <stdint.h>
extern "C" {
    void m5_dumpreset_stats(uint64_t ns_delay, uint64_t ns_period);
}

unsigned long long int arraySize = 1024;
bool uncached = false;
bool dumpStats = false;
int *bigArray;
int *smallArray;
unsigned numThreads = 1;
unsigned numIters = 30;
unsigned parallelism = 6;
unsigned parallelAccesses = 100000;
unsigned stride = 32;    // 32 ints = 128B cache line
pthread_barrier_t barr;
struct timeval lapCounter;
double lapStart, lapEnd;
double averageLap = 0.0;
double* lapTimes;

void *worker(void *arg) {
    unsigned threadID = *((int*)arg);
    unsigned i, j;
    unsigned index = threadID * arraySize / numThreads;
    int doublestride = 2*stride;
    int triplestride = 3*stride;
    int quadstride = 4*stride;
    int quintstride = 5*stride;
    int sextstride = 6*stride;
    int septstride = 7*stride;
    int octstride = 8*stride;
    int parallel = 0;
    unsigned long mask = 1 << threadID;
    pthread_setaffinity_np(pthread_self(), sizeof(mask), (cpu_set_t*)&mask);

    if (threadID == 0) {
        gettimeofday(&lapCounter, NULL);
        lapStart = (double)lapCounter.tv_sec + (double)lapCounter.tv_usec*1e-6;
    }

    for (i = 0; i < numIters; i++) {
        pthread_barrier_wait(&barr);
        switch (parallelism) {
          case 1:
              for (j = 0; j < parallelAccesses; j += parallelism) {
                  parallel += bigArray[index];
                  index += stride;
                  if (index >= arraySize) {
                      index -= arraySize;
                  }
              }
              break;
          case 2:
              for (j = 0; j < parallelAccesses; j += parallelism) {
                  parallel += bigArray[index] + bigArray[index+stride];
                  index += doublestride;
                  if (index >= arraySize) {
                      index -= arraySize;
                  }
              }
              break;
          case 3:
              for (j = 0; j < parallelAccesses; j += parallelism) {
                  parallel += bigArray[index] + bigArray[index+stride] + bigArray[index+doublestride];
                  index += triplestride;
                  if (index >= arraySize) {
                      index -= arraySize;
                  }
              }
              break;
          case 4:
              for (j = 0; j < parallelAccesses; j += parallelism) {
                  parallel += bigArray[index] + bigArray[index+stride] + bigArray[index+doublestride] + bigArray[index+triplestride];
                  index += quadstride;
                  if (index >= arraySize) {
                      index -= arraySize;
                  }
              }
              break;
          case 5:
              for (j = 0; j < parallelAccesses; j += parallelism) {
                  parallel += bigArray[index] + bigArray[index+stride] + bigArray[index+doublestride] + bigArray[index+triplestride] + bigArray[index+quadstride];
                  index += quintstride;
                  if (index >= arraySize) {
                      index -= arraySize;
                  }
              }
              break;
          case 6:
              for (j = 0; j < parallelAccesses; j += parallelism) {
                  parallel += bigArray[index] + bigArray[index+stride] + bigArray[index+doublestride] + bigArray[index+triplestride] + bigArray[index+quadstride] + bigArray[index+quintstride];
                  index += sextstride;
                  if (index >= arraySize) {
                      index -= arraySize;
                  }
              }
              break;
          case 7:
              for (j = 0; j < parallelAccesses; j += parallelism) {
                  parallel += bigArray[index] + bigArray[index+stride] + bigArray[index+doublestride] + bigArray[index+triplestride] + bigArray[index+quadstride] + bigArray[index+quintstride] + bigArray[index+sextstride];
                  index += septstride;
                  if (index >= arraySize) {
                      index -= arraySize;
                  }
              }
              break;
          case 8:
              for (j = 0; j < parallelAccesses; j += parallelism) {
                  parallel += bigArray[index] + bigArray[index+stride] + bigArray[index+doublestride] + bigArray[index+triplestride] + bigArray[index+quadstride] + bigArray[index+quintstride] + bigArray[index+sextstride] + bigArray[index+septstride];
                  index += octstride;
                  if (index >= arraySize) {
                      index -= arraySize;
                  }
              }
              break;
          default:
              break;
        }
        pthread_barrier_wait(&barr);
        if (threadID == 0) {
            gettimeofday(&lapCounter, NULL);
            lapEnd = (double)lapCounter.tv_sec + (double)lapCounter.tv_usec*1e-6;
            lapTimes[i] = lapEnd - lapStart;
            averageLap += lapTimes[i];
            lapStart = lapEnd;
        }
    }

    smallArray[threadID] = parallel;
    return NULL;
}

int main(int argc, char** argv) {

    unsigned i;
    int fd = 0;
    pthread_t *threads;
    pthread_attr_t pthread_custom_attr;

    for (int index = 0; index < argc; index++) {
        if (strcmp(argv[index], "-t") == 0) {
            if (argc > index+1) {
                numThreads = atoi(argv[++index]);
            } else {
                printf("ERROR: Must specify number of threads to -t option\n");
                exit(0);
            }
        } else if (strcmp(argv[index], "-a") == 0) {
            if (argc > index+1) {
                arraySize = atoi(argv[++index]);
            } else {
                printf("ERROR: Must specify number of array elements to -a option\n");
                exit(0);
            }
        } else if (strcmp(argv[index], "-d") == 0) {
            dumpStats = true;
        } else if (strcmp(argv[index], "-h") == 0) {
            printf("Usage: %s [options]\n\nOptions:\n", argv[0]);
            printf("-a #     Specify number of array elements\n");
            printf("-i #     Specify number of iterations to run\n");
            printf("-p #     Specify number of parallel accesses per-thread\n");
            printf("-q #     Specify per-thread memory-level parallelism\n");
            printf("-s #     Specify memory access stride in ints\n");
            printf("-t #     Specify number of threads\n");
            printf("-u       Use uncacheable memory\n");
            exit(0);
        } else if (strcmp(argv[index], "-i") == 0) {
            if (argc > index+1) {
                numIters = atoi(argv[++index]);
            } else {
                printf("ERROR: Must specify number of iterations to -i option\n");
                exit(0);
            }
        } else if (strcmp(argv[index], "-p") == 0) {
            if (argc > index+1) {
                parallelAccesses = atoi(argv[++index]);
            } else {
                printf("ERROR: Must specify number of memory parallel accesses to -p option\n");
                exit(0);
            }
        } else if (strcmp(argv[index], "-q") == 0) {
            if (argc > index+1) {
                parallelism = atoi(argv[++index]);
            } else {
                printf("ERROR: Must specify number of memory-level parallelism to -q option\n");
                exit(0);
            }
        } else if (strcmp(argv[index], "-s") == 0) {
            if (argc > index+1) {
                stride = atoi(argv[++index]);
            } else {
                printf("ERROR: Must specify int stride to -s option\n");
                exit(0);
            }
        } else if (strcmp(argv[index], "-u") == 0) {
            uncached = true;
        }
    }

    parallelAccesses -= (parallelAccesses % parallelism);
    int threadIDs[numThreads];
    pthread_attr_init(&pthread_custom_attr);
    arraySize *= numThreads;
    if (!uncached) {
        bigArray = (int *) malloc(arraySize * sizeof(int));
    } else {
        fd = open("/dev/mem", O_CREAT | O_RDWR | O_SYNC, 0755);
        if (fd < 0) {
            fprintf(stderr, "Open failed uncached\n");
            exit(1);
        }
        bigArray = (int*) mmap(0x0, arraySize * sizeof(int), (PROT_READ | PROT_WRITE), (MAP_SHARED), fd, 0);
        if (bigArray == MAP_FAILED) {
            fprintf (stderr, "mmap uncached\n");
            unlink("/dev/mem");
            exit(1);
        }
    }
    smallArray = (int *) malloc(numThreads * sizeof(int));
    threads = (pthread_t *) malloc(numThreads * sizeof(pthread_t));
    lapTimes = (double*) malloc(numIters * sizeof(double));

    printf("Number of threads: %u\n", numThreads);
    printf("Number of iterations: %u\n", numIters);
    printf("Memory-level parallelism: %u\n", parallelism);
    printf("Number of accesses: %u\n", parallelAccesses);
    printf("Number of array elements: %llu\n", arraySize);
    printf("Size of bigArray (MB): %.3f\n", (double)(arraySize * sizeof(int))/(1024.0*1024.0));
    printf("Data stride: %u int (%luB) = %luB\n", stride, sizeof(int), stride * sizeof(int));
    for (i = 0; i < arraySize; i++) {
        bigArray[i] = i;
    }
    if (pthread_barrier_init(&barr, NULL, numThreads)) {
        printf("Could not create a barrier\n");
        return -1;
    }

    if (dumpStats) m5_dumpreset_stats(0, 0);
    for (i = 0; i < numThreads; i++) {
        threadIDs[i] = i;
        pthread_create(&threads[i], &pthread_custom_attr, worker, &threadIDs[i]);
    }
    for (i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }
    if (dumpStats) m5_dumpreset_stats(0, 0);

    double stdev = 0.0;
    averageLap /= numIters;
    fprintf(stderr, "%d %d %llu", numThreads, parallelAccesses, arraySize);
    for (i = 0; i < numIters; i++) {
        fprintf(stderr, " %.6f", lapTimes[i]);
        if (i > 1) {
            double temp = averageLap - lapTimes[i];
            stdev += temp * temp;
        }
    }
    fprintf(stderr, "\n");
    stdev = sqrt(stdev/numIters);
    double percent = 100 * stdev / averageLap;
    printf("AVG: %.5f\n", averageLap);
    printf("STDEV: %f (%f%%)\n", stdev, percent);

    if (!uncached) {
        free(bigArray);
    } else {
        munmap(bigArray, arraySize * sizeof(int));
        if (fd != 0)
            close(fd);
        unlink("/dev/mem");
    }
    free(smallArray);
    free(threads);
    free(lapTimes);
    return 0;
}
