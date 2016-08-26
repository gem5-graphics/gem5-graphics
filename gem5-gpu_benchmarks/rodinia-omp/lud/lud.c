/*
 * =====================================================================================
 *
 *       Filename:  suite.c
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

#ifdef GEM5_WORK
#include <stdint.h>
void m5_dumpreset_stats(uint64_t ns_delay, uint64_t ns_period);
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
#endif

static int do_verify = 0;
int omp_num_threads = 1;

static struct option long_options[] = {
      /* name, has_arg, flag, val */
      {"input", 1, NULL, 'i'},
      {"size", 1, NULL, 's'},
      {"verify", 0, NULL, 'v'},
      {0,0,0,0}
};

extern void
lud_omp(float *m, int matrix_dim);

int
main ( int argc, char *argv[] )
{
  int matrix_dim = 32; /* default size */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *mm;
  stopwatch sw;

  while ((opt = getopt_long(argc, argv, "::vs:i:n:", 
                            long_options, &option_index)) != -1 ) {
      switch(opt){
        case 'i':
          input_file = optarg;
          break;
		case 'n':
          omp_num_threads = atoi(optarg);
          break;
        case 'v':
          do_verify = 1;
          break;
        case 's':
          matrix_dim = atoi(optarg);
          fprintf(stderr, "Currently not supported, use -i instead\n");
          fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n", argv[0]);
          exit(EXIT_FAILURE);
        case '?':
          fprintf(stderr, "invalid option\n");
          break;
        case ':':
          fprintf(stderr, "missing argument\n");
          break;
        default:
          fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n",
                  argv[0]);
          exit(EXIT_FAILURE);
      }
  }
  
  if ( (optind < argc) || (optind == 1)) {
      fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n", argv[0]);
      exit(EXIT_FAILURE);
  }

  if (input_file) {
      printf("Reading matrix from file %s\n", input_file);
      ret = create_matrix_from_file(&m, input_file, &matrix_dim);
      if (ret != RET_SUCCESS) {
          m = NULL;
          fprintf(stderr, "error create matrix from file %s\n", input_file);
          exit(EXIT_FAILURE);
      }
  } else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  } 

  if (do_verify){
    printf("Before LUD\n");
    print_matrix(m, matrix_dim);
    matrix_duplicate(m, &mm, matrix_dim);
  }

#ifdef GEM5_WORK
  m5_work_begin(0, 0);
  m5_dumpreset_stats(0, 0);
#endif

      stopwatch_start(&sw);
  lud_omp(m, matrix_dim);
      stopwatch_stop(&sw);
      printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));

#ifdef GEM5_WORK
  m5_dumpreset_stats(0, 0);
  m5_work_end(0, 0);
#endif

  if (do_verify){
    printf("After LUD\n");
    print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim); 
    free(mm);
  }

  free(m);

  return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
