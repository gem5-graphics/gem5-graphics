#ifndef MATRIX_H
#define MATRIX_H

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for gluPerspective.                                          */

void
buildPinfMatrix
(
  double fieldOfView,  /* Input:  Field of view angle, in degrees, in y    */
  double aspectRatio,  /* Input:  Viewport aspect ratio: width/height      */
  double zNear,        /* Input:  Distance to the near clipping plane      */
  float  m[16]         /* Output: Row-major (C-style) 4x4 transform matrix */
);

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for gluPerspective.                                          */

void
buildPerspectiveMatrix
(
  double fieldOfView,  /* Input:  Field of view angle, in degrees, in y    */
  double aspectRatio,  /* Input:  Viewport aspect ratio: width/height      */
  double zNear,        /* Input:  Distance to the near clipping plane      */
  double zFar,         /* Input:  Distance to the far clipping plane       */
  float  m[16]         /* Output: Row-major (C-style) 4x4 transform matrix */
);

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for gluLookAt.                                               */

void
buildLookAtMatrix
(
  double eyex,         /* Input:  Position of the eye point.               */
  double eyey,
  double eyez,
  double centerx,      /* Input:  Position of the reference point.         */
  double centery,
  double centerz,
  double upx,          /* Input:  Direction of the up vector.              */
  double upy,
  double upz,
  float  m[16]         /* Output: Row-major (C-style) 4x4 transform matrix */
);

/* Simple 4x4 matrix by 4x4 matrix multiply.                               */

void
multMatrix
(
  float       dst [16], /* Output: src1 * src2, can be src1 or src2        */
  const float src1[16], /* Input:  First matrix                            */
  const float src2[16]  /* Input:  Second matrix                           */
);

#endif
