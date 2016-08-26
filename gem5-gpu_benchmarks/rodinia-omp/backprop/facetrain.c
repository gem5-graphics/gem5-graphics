#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"

extern char *strcpy();
extern void exit();

int layer_size = 0;

backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_kernel(net, &out_err, &hid_err);
  bpnn_free(net);
  printf("Training done\n");
}

int setup(argc, argv)
int argc;
char *argv[];
{
#ifndef OPEN
  if(argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
#else
  if(argc<2 || argc>3){
  fprintf(stderr, "usage: backprop <num of input elements> <num threads>\n");
#endif
  exit(0);
  }

  layer_size = atoi(argv[1]);
#ifdef OPEN
  num_threads = 1;
  if(argc>2)
      num_threads = atoi(argv[2]);
  printf("Number of OMP Threads: %d\n", num_threads);
#endif

  int seed;

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
