
/* gli_merge.c - merge alpha channel into an in-memory image */

/* Copyright NVIDIA Corporation, 2000. */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "gli.h"

static const int __gliVerbose = 0;  // for debugging
static char __gliError[256];

int
gliMergeAlpha(gliGenericImage *image, gliGenericImage *alpha)
{
  int i, j;
  int w = image->width;
  int h = image->height;
  GLubyte *p;
  GLubyte *a;

  if (image->format == GL_COLOR_INDEX) {
    sprintf(__gliError,
      "gliMergeAlpha: source image not permiited to be paletted");
    if (__gliVerbose) {
      printf("%s\n", __gliError);
    }
    return 0;
  }
  if (w != alpha->width) {
    sprintf(__gliError,
      "gliMergeAlpha: image and alpha widths mismatch, %d!=%d",
      w, (int) alpha->width);
    if (__gliVerbose) {
      printf("%s\n", __gliError);
    }
    return 0;
  }
  if (h != alpha->height) {
    sprintf(__gliError,
      "gliMergeAlpha: image and alpha heights mismatch, %d!=%d",
      h, (int) alpha->height);
    if (__gliVerbose) {
      printf("%s\n", __gliError);
    }
    return 0;
  }
  if (alpha->components != 1) {
    sprintf(__gliError,
      "gliMergeAlpha: alpha image must have 1 component "
      "to merge alpha (instead of %d)",
      (int) alpha->components);
    if (__gliVerbose) {
      printf("%s\n", __gliError);
    }
    return 0;
  }

  if (image->components != 4 && image->components != 2) {
    GLubyte *np;
    
    assert(image->components == 3 || image->components == 1);
    if (__gliVerbose) {
      printf("gliMergeAlpha: adding alpha component to source image\n");
    }
    if (image->type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
      gliConvertImageToCoreFormat(image);
    }
    p = image->pixels;
    np = (GLubyte *) malloc (w * h * (image->components+1));
    if (np == NULL) {
      sprintf(__gliError, "gliMergeAlpha: malloc failed");
      if (__gliVerbose) {
        printf("%s\n", __gliError);
      }
      return 0;
    }
    switch (image->format) {
    case GL_RGB:
    case GL_BGR_EXT:
      assert(image->components == 3);
      for (j=0; j<h; j++) {
        for (i=0; i<w; i++) {
          np[(j*w + i)*4 + 0] = p[(j*w + i)*3 + 0];
          np[(j*w + i)*4 + 1] = p[(j*w + i)*3 + 1];
          np[(j*w + i)*4 + 2] = p[(j*w + i)*3 + 2];
        }
      }
      if (image->format == GL_RGB) {
        image->format = GL_RGBA;
      } else {
        image->format = GL_BGRA_EXT;
      }
      image->internalFormat = GL_RGBA8;
      break;
    case GL_ABGR_EXT:
      assert(image->components == 3);
      for (j=0; j<h; j++) {
        for (i=0; i<w; i++) {
          np[(j*w + i)*4 + 1] = p[(j*w + i)*3 + 0];
          np[(j*w + i)*4 + 2] = p[(j*w + i)*3 + 1];
          np[(j*w + i)*4 + 3] = p[(j*w + i)*3 + 2];
        }
      }
      image->format = GL_ABGR_EXT;
      image->internalFormat = GL_RGBA8;
      break;
    case GL_LUMINANCE:
    case GL_INTENSITY:
      assert(image->components == 1);
      for (j=0; j<h; j++) {
        for (i=0; i<w; i++) {
          np[(j*w + i)*2 + 0] = p[j*w + i];
        }
      }
      image->format = GL_LUMINANCE_ALPHA;
      image->internalFormat = GL_LUMINANCE8_ALPHA8;
      break;
    }
    free(p);
    image->pixels = np;
    image->components++;
  }

  p = image->pixels;
  a = alpha->pixels;

  switch (image->format) {
  case GL_LUMINANCE_ALPHA:
    for (j=0; j<h; j++) {
      for (i=0; i<w; i++) {
        p[(j*w + i)*2 + 1] = a[j*w + i];
      }
    }
    break;
  case GL_BGRA_EXT:
  case GL_RGBA:
    for (j=0; j<h; j++) {
      for (i=0; i<w; i++) {
        p[(j*w + i)*4 + 3] = a[j*w + i];
      }
    }
    break;
  case GL_ABGR_EXT:
    for (j=0; j<h; j++) {
      for (i=0; i<w; i++) {
        p[(j*w + i)*4 + 0] = a[j*w + i];
      }
    }
    break;
  default:
    sprintf(__gliError,
      "gliMergeAlpha: image format must be RBGA, BGRA, or ARGB "
      "(instead of 0x%x)",
      (unsigned int) image->format);
    if (__gliVerbose) {
      printf("%s\n", __gliError);
    }
    return 0;
  }
  return 1;
}
