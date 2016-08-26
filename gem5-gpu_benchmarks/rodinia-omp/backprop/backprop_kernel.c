#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "backprop.h"

#ifdef GEM5_WORK
#include <stdint.h>
void m5_dumpreset_stats(uint64_t ns_delay, uint64_t ns_period);
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
#endif

////////////////////////////////////////////////////////////////////////////////

extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern int setup(int argc, char** argv);

extern float **alloc_2d_dbl(int m, int n);

extern float squash(float x);

#ifdef TIMING
#include <sys/time.h>
double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


void bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
  printf("Performing CPU computation\n");

#ifdef GEM5_WORK
  m5_work_begin(0, 0);
  m5_dumpreset_stats(0, 0);
#endif

#ifdef TIMING
  double start_time = gettime();
#endif

  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#ifdef TIMING
  double end_time = gettime();
  printf("ROI Runtime: %f\n", end_time - start_time);
#endif

#ifdef GEM5_WORK
  m5_dumpreset_stats(0, 0);
  m5_work_end(0, 0);
#endif

  // Print partial_sum to console
  // Only in CUDA?
  // printf("Partial Sums:\n");
  // for (int j = 1; j <= hid && j<100; j++) {
  //   for (int k = 0; k < num_blocks; k++) {
  //     printf("%f ", partial_sum[k * hid + j-1]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // Print input_weights_one_dim to console
  // Can't find corresponding thing to cuda
  // printf("Input Weights:\n");
  // int m = 0;
  // int k, j;
  // for (k = 0; k <= in && k<100; k++) {
  //   for (j = 0; j <= hid; j++) {
  //     printf("%f ", net->input_prev_weights[m++]);
  //   }
  // }
  // printf("\n");

  // Print net->input_units to console
//  printf("Net Inputs:\n");
//  int k;
//  for (k = 0; k < in && k<100; k++) {
//     printf("%f ", net->input_units[k]);
//  }
//  printf("\n");

}
