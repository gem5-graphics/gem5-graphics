#ifndef __normalmap_h__
#define __normalmap_h__

/* normalmap.h - normal map construction routines */

/* Copyright NVIDIA Corporation, 2000. */

/* Structure to encode a normal like an 8-bit unsigned BGRA vector. */
typedef struct {
  /* The BGRA color component ordering is fastest for NVIDIA. */
  GLubyte nz, ny, nx;
  GLubyte mag;
} Normal;

extern Normal *convertHeightFieldToNormalMap(GLubyte *pixels,
                                             int w, int h,
					     int wr, int hr, float scale);
extern Normal *downSampleNormalMap(Normal *old, int w2, int h2, int w, int h);

#endif /* __normalmap_h__ */
