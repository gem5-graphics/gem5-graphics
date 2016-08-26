/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "args.h"

#include "model.h"
#include "scan_largearray_kernel.cu"

#define CUDA_ERRCK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

extern float *dev_binb;
extern unsigned int NUM_SETS;
extern unsigned int NUM_ELEMENTS;

int
main(int argc, char** argv)
{
    struct pb_TimerSet timers;

    pb_InitializeTimerSet(&timers);

    options args;
    parse_args(argc, argv, &args);
    if (args.npoints <= 0) {
        fprintf(stderr, "ERROR: Need to specify number of points to -p opt\n");
        exit(-1);
    }

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    NUM_ELEMENTS = args.npoints;
    NUM_SETS = args.random_count;
    NUM_BINS = args.nbins;
    int num_elements = NUM_ELEMENTS;

    printf("Min distance: %f arcmin\n", min_arcmin);
    printf("Max distance: %f arcmin\n", max_arcmin);
    printf("Bins per dec: %i\n", bins_per_dec);
    if (NUM_SETS > 0) {
        printf("Number sets : %i\n", NUM_SETS);
        if (args.random_name == NULL) {
            fprintf(stderr, "ERROR: Need to specify random file prefix to -r opt\n");
            exit(-1);
        }
        printf("Rand prefix : %s\n", args.random_name);
    }
    printf("Number elts : %i\n", NUM_ELEMENTS);
    printf("Total bins  : %i\n", NUM_BINS);

    //read in files
    unsigned mem_size = (1 + NUM_SETS) * num_elements * sizeof(struct cartesian);
    unsigned f_mem_size = (1 + NUM_SETS) * num_elements * sizeof(float);

    // container for all the points read from files
    struct cartesian *h_all_data;
    h_all_data = (struct cartesian*) malloc(mem_size);
    // Until I can get libs fixed

    // iterator for data files
    struct cartesian *working = h_all_data;

    // go through and read all data and random points into h_all_data
    working += num_elements;
    for (int i = 1; i <= (NUM_SETS); i++) {
        pb_SwitchToTimer(&timers, pb_TimerID_IO);
        char filename[2048];
        snprintf(filename, 2048, "%s.%d", args.random_name, i);
        readdatafile(filename, working, num_elements);
        pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

        working += num_elements;
    }

    // split into x, y, and z arrays
    float * h_x_data = (float*) malloc(3 * f_mem_size);
    float * h_y_data = h_x_data + NUM_ELEMENTS * (NUM_SETS + 1);
    float * h_z_data = h_y_data + NUM_ELEMENTS * (NUM_SETS + 1);
    for (int i = 0; i < (NUM_SETS + 1); ++i) {
        for (int j = 0; j < NUM_ELEMENTS; ++j) {
            h_x_data[i * NUM_ELEMENTS + j] = h_all_data[i * NUM_ELEMENTS + j].x;
            h_y_data[i * NUM_ELEMENTS + j] = h_all_data[i * NUM_ELEMENTS + j].y;
            h_z_data[i * NUM_ELEMENTS + j] = h_all_data[i * NUM_ELEMENTS + j].z;
        }
    }

    // from on use x, y, and z arrays, free h_all_data
    free(h_all_data);
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    // allocate cuda memory to hold all points
    float * d_x_data;
    cudaMalloc((void**) & d_x_data, 3 * f_mem_size);
    CUDA_ERRCK
    float * d_y_data = d_x_data + NUM_ELEMENTS * (NUM_SETS + 1);
    float * d_z_data = d_y_data + NUM_ELEMENTS * (NUM_SETS + 1);

    // allocate cuda memory to hold final histograms
    // (1 for dd, and NUM_SETS for dr and rr apiece)
    hist_t * d_hists;
    cudaMalloc((void**) & d_hists, NUM_BINS * (NUM_SETS * 2 + 1) * sizeof(hist_t));
    CUDA_ERRCK
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    // allocate system memory for final histograms
    hist_t *new_hists = (hist_t *) malloc(NUM_BINS * (NUM_SETS * 2 + 1) * sizeof(hist_t));

    // Initialize the boundary constants for bin search
    cudaMalloc(&dev_binb, (NUM_BINS + 1) * sizeof(float));

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    initBinB(&timers);
    CUDA_ERRCK

    // **===------------------ Kick off TPACF on CUDA------------------===**
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    cudaMemcpy(d_x_data, h_x_data, 3 * f_mem_size, cudaMemcpyHostToDevice);
    CUDA_ERRCK
    pb_SwitchToTimer(&timers, pb_TimerID_GPU);

    TPACF(d_hists, d_x_data, d_y_data, d_z_data);
    cudaThreadSynchronize();

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    cudaMemcpy(new_hists, d_hists, NUM_BINS * (NUM_SETS * 2 + 1) * sizeof(hist_t), cudaMemcpyDeviceToHost);
    CUDA_ERRCK
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    // **===-----------------------------------------------------------===**

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    // references into output histograms
    hist_t *dd_hist = new_hists;
    hist_t *rr_hist = dd_hist + NUM_BINS;
    hist_t *dr_hist = rr_hist + NUM_BINS * NUM_SETS;

    // add up values within dr and rr
    int rr[NUM_BINS];
    for (int i = 0; i < NUM_BINS; i++) {
        rr[i] = 0;
    }
    for (int i = 0; i < NUM_SETS; i++) {
        for (int j = 0; j < NUM_BINS; j++) {
            rr[j] += rr_hist[i * NUM_BINS + j];
        }
    }
    int dr[NUM_BINS];
    for (int i = 0; i < NUM_BINS; i++) {
        dr[i] = 0;
    }
    for (int i = 0; i < NUM_SETS; i++) {
        for (int j = 0; j < NUM_BINS; j++) {
            dr[j] += dr_hist[i * NUM_BINS + j];
        }
    }

    int dd_t = 0;
    int dr_t = 0;
    int rr_t = 0;
    FILE *outfile;
    if ((outfile = fopen(args.output_name, "w")) == NULL) {
        fprintf(stderr, "Unable to open output file %s for writing, "
                "assuming stdout\n", args.output_name);
        outfile = stdout;
    }

    // print out final histograms + omega (while calculating omega)
    for (int i = 0; i < NUM_BINS; i++) {
        float w = (100.0 * dd_hist[i] - dr[i]) / rr[i] + 1.0;
        pb_SwitchToTimer(&timers, pb_TimerID_IO);
        fprintf(outfile, "%d: %f\n", i, w);
        pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
        dd_t += dd_hist[i];
        dr_t += dr[i];
        rr_t += rr[i];
    }

    if (outfile != stdout) {
        fclose(outfile);
    }

    // cleanup memory
    free(new_hists);
    free(h_x_data);

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    cudaFree(d_hists);
    cudaFree(d_x_data);

    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
    pb_PrintTimerSet(&timers);
}

