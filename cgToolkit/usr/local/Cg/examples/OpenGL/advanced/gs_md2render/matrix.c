#include "matrix.h"

#include <math.h>     /* for sqrt, sin, cos, and fabs */
#include <assert.h>   /* for assert */

static const double myPi = 3.14159265358979323846;

void
buildPinfMatrix
(
  double fieldOfView,
  double aspectRatio,
  double zNear,
  float m[16]
)
{
  double sine, cotangent;
  double radians = fieldOfView / 2.0 * myPi / 180.0;

  sine = sin(radians);
  /* Should be non-zero to avoid division by zero. */
  assert(sine);
  assert(aspectRatio);
  cotangent = cos(radians) / sine;

  m[0*4+0] = (float)(cotangent / aspectRatio);
  m[0*4+1] = 0.0f;
  m[0*4+2] = 0.0f;
  m[0*4+3] = 0.0f;

  m[1*4+0] = 0.0f;
  m[1*4+1] = (float) cotangent;
  m[1*4+2] = 0.0f;
  m[1*4+3] = 0.0f;

  m[2*4+0] = 0.0f;
  m[2*4+1] = 0.0f;
  m[2*4+2] = -0.999999f;
  m[2*4+3] = (float)(-2 * zNear);

  m[3*4+0] = 0.0f;
  m[3*4+1] = 0.0f;
  m[3*4+2] = -1.0f;
  m[3*4+3] = 0.0f;
}

void
buildPerspectiveMatrix
(
  double fieldOfView,
  double aspectRatio,
  double zNear,
  double zFar,
  float m[16]
)
{
  double sine, cotangent, deltaZ;
  double radians = fieldOfView / 2.0 * myPi / 180.0;

  deltaZ = zFar - zNear;
  sine = sin(radians);
  /* Should be non-zero to avoid division by zero. */
  assert(deltaZ);
  assert(sine);
  assert(aspectRatio);
  cotangent = cos(radians) / sine;

  m[0*4+0] = (float)(cotangent / aspectRatio);
  m[0*4+1] = 0.0f;
  m[0*4+2] = 0.0f;
  m[0*4+3] = 0.0f;

  m[1*4+0] = 0.0f;
  m[1*4+1] = (float) cotangent;
  m[1*4+2] = 0.0f;
  m[1*4+3] = 0.0f;

  m[2*4+0] = 0.0f;
  m[2*4+1] = 0.0f;
  m[2*4+2] = (float)( -(zFar + zNear) / deltaZ );
  m[2*4+3] = (float)( -2 * zNear * zFar / deltaZ );

  m[3*4+0] = 0.0f;
  m[3*4+1] = 0.0f;
  m[3*4+2] = -1.0f;
  m[3*4+3] = 0.0f;
}

void
buildLookAtMatrix
(
  double eyex,
  double eyey,
  double eyez,
  double centerx,
  double centery,
  double centerz,
  double upx,
  double upy,
  double upz,
  float m[16]
)
{
  double x[3], y[3], z[3], mag;

  /* Difference eye and center vectors to make Z vector. */
  z[0] = eyex - centerx;
  z[1] = eyey - centery;
  z[2] = eyez - centerz;
  /* Normalize Z. */
  mag = sqrt(z[0]*z[0] + z[1]*z[1] + z[2]*z[2]);
  if (mag) {
    z[0] /= mag;
    z[1] /= mag;
    z[2] /= mag;
  }

  /* Up vector makes Y vector. */
  y[0] = upx;
  y[1] = upy;
  y[2] = upz;

  /* X vector = Y cross Z. */
  x[0] =  y[1]*z[2] - y[2]*z[1];
  x[1] = -y[0]*z[2] + y[2]*z[0];
  x[2] =  y[0]*z[1] - y[1]*z[0];

  /* Recompute Y = Z cross X. */
  y[0] =  z[1]*x[2] - z[2]*x[1];
  y[1] = -z[0]*x[2] + z[2]*x[0];
  y[2] =  z[0]*x[1] - z[1]*x[0];

  /* Normalize X. */
  mag = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
  if (mag) {
    x[0] /= mag;
    x[1] /= mag;
    x[2] /= mag;
  }

  /* Normalize Y. */
  mag = sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2]);
  if (mag) {
    y[0] /= mag;
    y[1] /= mag;
    y[2] /= mag;
  }

  /* Build resulting view matrix. */
  m[0*4+0] = (float) x[0];  m[0*4+1] = (float) x[1];
  m[0*4+2] = (float) x[2];  m[0*4+3] = (float)(-x[0]*eyex + -x[1]*eyey + -x[2]*eyez);

  m[1*4+0] = (float) y[0];  m[1*4+1] = (float) y[1];
  m[1*4+2] = (float) y[2];  m[1*4+3] = (float)(-y[0]*eyex + -y[1]*eyey + -y[2]*eyez);

  m[2*4+0] = (float) z[0];  m[2*4+1] = (float) z[1];
  m[2*4+2] = (float) z[2];  m[2*4+3] = (float)(-z[0]*eyex + -z[1]*eyey + -z[2]*eyez);

  m[3*4+0] = 0.0f;   m[3*4+1] = 0.0f;  m[3*4+2] = 0.0f;  m[3*4+3] = 1.0f;
}

void
multMatrix
(
  float dst[16],
  const float src1[16],
  const float src2[16]
)
{
  float tmp[16];
  int i, j;

  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      tmp[i*4+j] = src1[i*4+0] * src2[0*4+j] +
                   src1[i*4+1] * src2[1*4+j] +
                   src1[i*4+2] * src2[2*4+j] +
                   src1[i*4+3] * src2[3*4+j];
    }
  }
  /* Copy result to dst (so dst can also be src1 or src2). */
  for (i=0; i<16; i++)
    dst[i] = tmp[i];
}
