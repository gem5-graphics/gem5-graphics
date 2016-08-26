
/* matrix.h - Various utility routines for making 4x4 matrices and using them. */

/* NOTE: All 4x4 matrices are stored as an array of 16 floats in row-major order (like C, not
   like FORTRAN).  If you pass these matrices to OpenGL functions,
   use glLoadTransposeMatrixf, etc. (rather than simply glLoadMatrixf). */

/* Philosophy:

   1)  Routines that make or invert a matrix use last parameter as matrix result.

   2)  Routines that multiply matrices or transform by matrices have result as first parameter.

   3)  The result matrix when inverting or multiplying matrices can safely be a source parameter too.

   */

#include <assert.h>
#include <math.h>

void makePerspectiveMatrix(double fieldOfView,
                           double aspectRatio,
                           double zNear, double zFar,
                           float m[16]);

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for gluLookAt. */
void makeLookAtMatrix(double eyex, double eyey, double eyez,
                      double centerx, double centery, double centerz,
                      double upx, double upy, double upz,
                      float m[16]);

/* Simple 4x4 matrix by 4x4 matrix multiply. */
void multMatrix(float dst[16],
                const float src1[16], const float src2[16]);

/* Normalize a 3-component vector. */
void normalizeDirection(float v[3]);

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for glRotatef. */
void makeRotateMatrix(float angle,
                      float ax, float ay, float az,
                      float m[16]);

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for glTranslatef. */
void makeTranslateMatrix(float x, float y, float z, float m[16]);

/* Invert a row-major (C-style) 4x4 matrix. */
void invertMatrix(float out[16], const float m[16]);

/* Transpose a 4x4 matrix. */
void transposeMatrix(float dst[16], const float mat[16]);

/* Simple 4x4 matrix by 4-component column vector multiply and perform perspective divide. */
void transformPosition(float dst[4],
                       const float mat[16], const float vec[4]);

/* Simple 4x4 matrix by 4-component column vector multiply. */
void transformVector(float dst[4],
                     const float mat[16], const float vec[4]);

/* Simple upper-left 3x3 of a 4x4 matrix by 3-component column vector multiply. */
void transformDirection(float dst[3],
                        const float mat[16],
                        const float vec[3]);

void printMatrix(const char *name, const float mat[16]);
void printVector(const char *name, const float vec[4]);
void printDirection(const char *name, const float vec[4]);
