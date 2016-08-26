#ifndef __gli_h__
#define __gli_h__

/* gli.h - OpenGL image (GLI) file loader */

/* Copyright NVIDIA Corporation, 1999. */

/* A lightweight generic image file loader for OpenGL programs. */

#include <stdio.h>

#include <GL/glew.h>

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct {

  GLsizei  width;
  GLsizei  height;
  GLint    components;
  GLenum   type;
  GLenum   format;
  GLenum   internalFormat;

  GLsizei  cmapEntries;
  GLenum   cmapFormat;
  GLubyte *cmap;

  GLubyte *pixels;
  
} gliGenericImage;

extern gliGenericImage *gliReadTGA(FILE *fp, const char *name, int yFlip);
extern void gliFree(gliGenericImage *image);
extern gliGenericImage *gliScaleImage(gliGenericImage *image, int nw, int nh);
extern void gliConvertImageToCoreFormat(gliGenericImage *image);
extern int gliMergeAlpha(gliGenericImage *image, gliGenericImage *alpha);

#ifdef  __cplusplus
}
#endif

#endif /* __gli_h__ */
