
/* gli_misc.c - OpenGL image (GLI) miscellaneous routines */

/* Copyright NVIDIA Corporation, 2006. */

/* A lightweight generic image file loader for OpenGL programs. */

#include <stdlib.h>

#include "gli.h"

void
gliFree(gliGenericImage *image)
{
  if (image->pixels) {
    free(image->pixels);
  }
  if (image->cmap) {
    free(image->cmap);
  }
  free(image);
}

