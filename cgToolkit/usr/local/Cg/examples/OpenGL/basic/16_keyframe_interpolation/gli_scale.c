
/* gli_convert.c - OpenGL image (GLI) file loader */

/* Copyright NVIDIA Corporation, 2000. */

#include <assert.h>
#include <stdlib.h>

#include "gli.h"

gliGenericImage *
gliScaleImage(gliGenericImage *image, int nw, int nh)
{
  gliGenericImage *newImage;
  GLubyte *newPixels;
  GLenum status;

  /* Cannot scale paletted images; application must gliDepalette first. */
  assert(image->cmap == NULL);
  assert(image->cmapEntries == 0);
  assert(image->cmapFormat == GL_NONE);

  newPixels = (GLubyte*) malloc(nw*nh*image->components);
  assert(newPixels);

  status = gluScaleImage(image->format, image->width, image->height, image->type,
    image->pixels, nw, nh, image->type, newPixels);
  /* gluScalImage may fail for formats introduced by extensions.
     For example, GL_BGR and GL_BGRA (introduced by the EXT_bgra
     extension though now part of OpenGL 1.2) are not recognized
     by some early GLU implementations.  In particular, the
     Macintosh GLU lacks BGR and BGRA support.  In such cases,
     gluScaleImage may fail with GLU_INVALID_ENUM and we may
     have to convert the image from BGR or BGRA to RGB or RGBA
     to ensure gluScaleImage can scale the image correctly. */
  if (status == GLU_INVALID_ENUM) {
    gliConvertImageToCoreFormat(image);
    status = gluScaleImage(image->format,
      image->width, image->height, image->type,
      image->pixels, nw, nh, image->type, newPixels);
  }
  assert(status == 0);

  newImage = (gliGenericImage*) malloc(sizeof(gliGenericImage));
  newImage->width = nw;
  newImage->height = nh;
  newImage->format = image->format;
  newImage->internalFormat = image->internalFormat;
  newImage->type = image->type;
  newImage->components = image->components;
  newImage->cmapEntries = 0;
  newImage->cmapFormat = GL_NONE;
  newImage->cmap = NULL;
  newImage->pixels = newPixels;

  return newImage;
}
