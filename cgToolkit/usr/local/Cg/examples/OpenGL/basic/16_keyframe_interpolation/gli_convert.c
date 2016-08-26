
/* gli_convert.c - OpenGL image (GLI) file loader */

/* Copyright NVIDIA Corporation, 2000. */

/* Fromat conversion routines for lightweight generic image
   file loader for OpenGL programs. */

#include <assert.h>
#include <stdlib.h>

#include "gli.h"

void
gliConvertBGRtoRGB(gliGenericImage *image)
{
  const int components = 3;
  GLubyte pixel[3];
  int w, h;
  GLubyte *pixels;
  int i, j, c;
  
  w = image->width;
  h = image->height;
  
  assert(image->format == GL_BGR_EXT);
  image->format = GL_RGB;
  
  pixels = image->pixels;
  for (j=0; j<h; j++) {
    for (i=0; i<w; i++) {
      for (c=0; c<components; c++) {
        pixel[c] = pixels[(j*w+i)*components+c];
      }
      pixels[(j*w+i)*components+0] = pixel[2];
      pixels[(j*w+i)*components+1] = pixel[1];
      pixels[(j*w+i)*components+2] = pixel[0];
    }
  }
}

void
gliConvertBGRAtoRGBA(gliGenericImage *image)
{
  const int components = 4;
  GLubyte pixel[4];
  int w, h;
  GLubyte *pixels;
  GLubyte *spixels, spixel[2];
  int i, j, c;

  w = image->width;
  h = image->height;
  pixels = image->pixels;

  assert(image->format == GL_BGRA_EXT);
  
  switch (image->type) {
  case GL_UNSIGNED_BYTE:
    image->format = GL_RGBA;
    for (j=0; j<h; j++) {
      for (i=0; i<w; i++) {
        for (c=0; c<components; c++) {
          pixel[c] = pixels[(j*w+i)*components+c];
        }
        pixels[(j*w+i)*components+0] = pixel[2];
        pixels[(j*w+i)*components+1] = pixel[1];
        pixels[(j*w+i)*components+2] = pixel[0];
        pixels[(j*w+i)*components+3] = pixel[3];
      }
    }
    break;
  case GL_UNSIGNED_SHORT_1_5_5_5_REV:
    assert(image->type == GL_UNSIGNED_SHORT_1_5_5_5_REV);
    if (image->components == 4) {
      image->format = GL_RGBA;
      spixels = (GLubyte*) image->pixels;
      image->pixels = (GLubyte*) malloc(w * h * 4);
      pixels = image->pixels;
      for (j=0; j<h; j++) {
        for (i=0; i<w; i++) {
          GLubyte red, green, blue;
          
          spixel[0] = spixels[(j*w+i)*2+0];
          spixel[1] = spixels[(j*w+i)*2+1];

          red = (spixel[1] & 0x7c) >> 2;
          red = (red << 3) | ((red & 0x1) * 0x7);

          green = ((spixel[1] & 0x03) << 3) | ((spixel[0] & 0xe0) >> 5);
          green = (green << 3) | ((green & 0x1) * 0x7);

          blue = (spixel[0] & 0x1f) >> 0;
          blue = (blue << 3) | ((blue & 0x1) * 0x7);

          pixels[(j*w+i)*components+0] = red;
          pixels[(j*w+i)*components+1] = green;
          pixels[(j*w+i)*components+2] = blue;
          pixels[(j*w+i)*components+3] = 0xff;
        }
      }
    } else {
      assert(image->components == 3);
      image->format = GL_RGB;
      spixels = image->pixels;
      image->pixels = (GLubyte*) malloc(w * h * 3);
      pixels = image->pixels;
      for (j=0; j<h; j++) {
        for (i=0; i<w; i++) {
          GLubyte red, green, blue;

          spixel[0] = spixels[(j*w+i)*2+0];
          spixel[1] = spixels[(j*w+i)*2+1];

          red = (spixel[1] & 0x7c) >> 2;
          red = (red << 3) | ((red & 0x1) * 0x7);

          green = ((spixel[1] & 0x03) << 3) | ((spixel[0] & 0xe0) >> 5);
          green = (green << 3) | ((green & 0x1) * 0x7);

          blue = (spixel[0] & 0x1f) >> 0;
          blue = (blue << 3) | ((blue & 0x1) * 0x7);

          pixels[(j*w+i)*3+0] = red;
          pixels[(j*w+i)*3+1] = green;
          pixels[(j*w+i)*3+2] = blue;
        }
      }
    }
    image->type = GL_UNSIGNED_BYTE;
    free(spixels);
    break;
  default:
    assert(0);
    break;
  }
}

void
gliConvertABGRoRGBA(gliGenericImage *image)
{
  const int components = 4;
  GLubyte pixel[4];
  int w, h;
  GLubyte *pixels;
  int i, j, c;

  w = image->width;
  h = image->height;
  pixels = image->pixels;

  assert(image->format == GL_ABGR_EXT);
  image->format = GL_RGBA;

  for (j=0; j<h; j++) {
    for (i=0; i<w; i++) {
      for (c=0; c<components; c++) {
        pixel[c] = pixels[(j*w+i)*components+c];
      }
      pixels[(j*w+i)*components+0] = pixel[3];
      pixels[(j*w+i)*components+1] = pixel[2];
      pixels[(j*w+i)*components+2] = pixel[1];
      pixels[(j*w+i)*components+3] = pixel[0];
    }
  }
}

void
gliConvertImageToCoreFormat(gliGenericImage *image)
{
  switch (image->format) {
  case GL_BGR_EXT:
    gliConvertBGRtoRGB(image);
    break;
  case GL_BGRA_EXT:
    gliConvertBGRAtoRGBA(image);
    break;
  case GL_ABGR_EXT:
    gliConvertABGRoRGBA(image);
    break;
  default:
    /* Assume nothing needed. */
    break;
  }
}
