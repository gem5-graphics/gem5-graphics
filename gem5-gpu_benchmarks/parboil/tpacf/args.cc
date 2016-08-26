/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include "args.h"

extern char *optarg;

void usage(char *name)
{
    printf("Usage: %s <-d data_file_name> <-r rnd_file_name> "
           "<-m rnd_count> <-p count> <-o file_name>\n", name);
    exit(0);
}

void parse_args(int argc, char **argv, options* args)
{
    int c;

    args->data_name = NULL;
    args->random_name = NULL;
    args->random_count = 0;
    args->npoints = 0;
    args->output_name = NULL;
    args->nbins = 0;

    while ((c = getopt(argc, argv, "b:d:n:r:p:o:i:")) != EOF) {
        switch (c) {
        case 'b':
            args->nbins = atoi(optarg);
            break;
        case 'd':
            args->data_name = optarg;
            break;
        case 'r':
            args->random_name = optarg;
            break;
        case 'n':
            args->random_count = atoi(optarg);
            break;
        case 'o':
            args->output_name = optarg;
            break;
        case 'p':
            args->npoints = atol(optarg);
            break;
        case 'i':
            printf("Ignoring PB parameter: %c\n", c);
            break;
        default:
            usage(argv[0]);
        }
    }
}
