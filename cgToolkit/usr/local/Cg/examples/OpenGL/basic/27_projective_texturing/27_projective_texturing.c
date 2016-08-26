
/* 27_projective_texturing.c - OpenGL-based projective texturing example
   using Cg program from Chapter 9 of "The Cg Tutorial" (Addison-Wesley,
   ISBN 0321194969). */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   1.5 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sqrt, sin, cos, and fabs */
#include <assert.h>   /* for assert */

#include <GL/glew.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#ifdef _WIN32
#include <GL/wglew.h>
#else
#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#else
#include <GL/glxew.h>
#endif
#endif

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgFragmentProfile;
static CGprogram   myCgVertexProgram,
                   myCgFragmentProgram;
static CGparameter myCgVertexParam_modelViewProj,
                   myCgVertexParam_Kd,
                   myCgVertexParam_lightPosition,
                   myCgVertexParam_textureMatrix;

static const char *myProgramName = "27_projective_texturing",
                  *myVertexProgramFileName = "C9E5v_projTex.cg",
/* Page 254 */    *myVertexProgramName = "C9E5v_projTex",
                  *myFragmentProgramFileName = "C9E6f_projTex.cg",
/* Page 254 */    *myFragmentProgramName = "C9E6f_projTex";

static float eyeHeight = 5.0f;    /* Vertical height of light. */
static float eyeAngle  = 0.53f;   /* Angle in radians eye rotates around scene. */
static float lightAngle = -0.4;   /* Angle light rotates around scene. */
static float lightHeight = 2.0f;  /* Vertical height of light. */
static float projectionMatrix[16];

static const float Kd[3] = { 1, 1, 1 };  /* White. */

static void checkForCgError(const char *situation)
{
  CGerror error;
  const char *string = cgGetLastErrorString(&error);

  if (error != CG_NO_ERROR) {
    printf("%s: %s: %s\n",
      myProgramName, situation, string);
    if (error == CG_COMPILER_ERROR) {
      printf("%s\n", cgGetLastListing(myCgContext));
    }
    exit(1);
  }
}

/* Forward declared GLUT callbacks registered by main. */
static void reshape(int width, int height);
static void display(void);
static void keyboard(unsigned char c, int x, int y);
static void menu(int item);
static void mouse(int button, int state, int x, int y);
static void motion(int x, int y);
static void setupDemonSampler(void);
static void requestSynchronizedSwapBuffers(void);

int main(int argc, char **argv)
{
  glutInitWindowSize(600, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_1) {
    fprintf(stderr, "%s: failed to initialize GLEW, OpenGL 1.1 required.\n", myProgramName);    
    exit(1);
  }

  requestSynchronizedSwapBuffers();
  glClearColor(0.1, 0.1, 0.1, 0);  /* Gray background. */
  glEnable(GL_DEPTH_TEST);         /* Hidden surface removal. */

  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgGLSetDebugMode(CG_FALSE);
  /* The example uses just one texture unit but let the Cg runtime manage
     binding this sampler. */
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);

  myCgVertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
  cgGLSetOptimalOptions(myCgVertexProfile);
  checkForCgError("selecting vertex profile");

  myCgVertexProgram =
    cgCreateProgramFromFile(
      myCgContext,              /* Cg runtime context */
      CG_SOURCE,                /* Program in human-readable form */
      myVertexProgramFileName,  /* Name of file containing program */
      myCgVertexProfile,        /* Profile: OpenGL ARB vertex program */
      myVertexProgramName,      /* Entry function name */
      NULL);                    /* No extra compiler options */
  checkForCgError("creating vertex program from file");
  cgGLLoadProgram(myCgVertexProgram);
  checkForCgError("loading vertex program");

#define GET_VERTEX_PARAM(name) \
  myCgVertexParam_##name = \
    cgGetNamedParameter(myCgVertexProgram, #name); \
  checkForCgError("could not get " #name " parameter");

  GET_VERTEX_PARAM(modelViewProj);
  GET_VERTEX_PARAM(Kd);
  GET_VERTEX_PARAM(lightPosition);
  GET_VERTEX_PARAM(textureMatrix);

  cgSetParameter3fv(myCgVertexParam_Kd, Kd);

  myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  cgGLSetOptimalOptions(myCgFragmentProfile);
  checkForCgError("selecting fragment profile");

  myCgFragmentProgram =
    cgCreateProgramFromFile(
      myCgContext,              /* Cg runtime context */
      CG_SOURCE,                /* Program in human-readable form */
      myFragmentProgramFileName,
      myCgFragmentProfile,      /* Profile: latest fragment profile */
      myFragmentProgramName,    /* Entry function name */
      NULL); /* No extra compiler options */
  checkForCgError("creating fragment program from string");
  cgGLLoadProgram(myCgFragmentProgram);
  checkForCgError("loading fragment program");

  setupDemonSampler();

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAddMenuEntry("[r] Toggle repeat/clamp projection", 'r');
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

static const GLubyte
myDemonTextureImage[3*(128*128)] = {
/* RGB8 image data for a mipmapped 128x128 demon texture */
#include "demon_image.h"
};

static void setupDemonSampler(void)
{
  GLuint texobj = 666;
  CGparameter sampler;

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); /* Tightly packed texture data. */

  glBindTexture(GL_TEXTURE_2D, texobj);
  /* Load demon decal texture with mipmaps. */
  gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB8,
    128, 128, GL_RGB, GL_UNSIGNED_BYTE, myDemonTextureImage);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
    GL_LINEAR_MIPMAP_LINEAR);

  /* Assumes OpenGL 1.3 or ARB_texture_border_clamp */
#ifndef GL_CLAMP_TO_BORDER
#define GL_CLAMP_TO_BORDER                0x812D
#endif

  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, Kd);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

  sampler = cgGetNamedParameter(myCgFragmentProgram, "projectiveMap");
  checkForCgError("getting projectiveMap sampler parameter");
  cgGLSetTextureParameter(sampler, texobj);
}

/* Forward declared routine used by reshape callback. */
static void buildPerspectiveMatrix(double fieldOfView,
                                   double aspectRatio,
                                   double zMin, double zMax,
                                   float m[16]);

static void reshape(int width, int height)
{
  double aspectRatio = (float) width / (float) height;
  double fieldOfView = 40.0; /* Degrees */

  /* Build eye's projection matrix once per window resize. */
  buildPerspectiveMatrix(fieldOfView, aspectRatio,
                         1.0, 50.0,  /* Znear and Zfar */
                         projectionMatrix);
  /* Light drawn with fixed-function transformation so also load same
     projection matrix in fixed-function projection matrix. */
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(fieldOfView, aspectRatio,
                 1.0, 50.0);  /* Znear and Zfar */
  glMatrixMode(GL_MODELVIEW);
  /* Set viewport to window dimensions. */
  glViewport(0, 0, width, height);
}

static const double myPi = 3.14159265358979323846;

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for gluPerspective. */
static void buildPerspectiveMatrix(double fieldOfView,
                                   double aspectRatio,
                                   double zNear, double zFar,
                                   float m[16])
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
  
  m[0*4+0] = cotangent / aspectRatio;
  m[0*4+1] = 0.0;
  m[0*4+2] = 0.0;
  m[0*4+3] = 0.0;
  
  m[1*4+0] = 0.0;
  m[1*4+1] = cotangent;
  m[1*4+2] = 0.0;
  m[1*4+3] = 0.0;
  
  m[2*4+0] = 0.0;
  m[2*4+1] = 0.0;
  m[2*4+2] = -(zFar + zNear) / deltaZ;
  m[2*4+3] = -2 * zNear * zFar / deltaZ;
  
  m[3*4+0] = 0.0;
  m[3*4+1] = 0.0;
  m[3*4+2] = -1;
  m[3*4+3] = 0;
}

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for gluLookAt. */
static void buildLookAtMatrix(double eyex, double eyey, double eyez,
                              double centerx, double centery, double centerz,
                              double upx, double upy, double upz,
                              float m[16])
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
  m[0*4+0] = x[0];  m[0*4+1] = x[1];
  m[0*4+2] = x[2];  m[0*4+3] = -x[0]*eyex + -x[1]*eyey + -x[2]*eyez;

  m[1*4+0] = y[0];  m[1*4+1] = y[1];
  m[1*4+2] = y[2];  m[1*4+3] = -y[0]*eyex + -y[1]*eyey + -y[2]*eyez;

  m[2*4+0] = z[0];  m[2*4+1] = z[1];
  m[2*4+2] = z[2];  m[2*4+3] = -z[0]*eyex + -z[1]*eyey + -z[2]*eyez;

  m[3*4+0] = 0.0;   m[3*4+1] = 0.0;  m[3*4+2] = 0.0;  m[3*4+3] = 1.0;
}

static void normalizeVector(float v[3])
{
  float mag;

  mag = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  if (mag) {
    float oneOverMag = 1.0 / mag;

    v[0] *= oneOverMag;
    v[1] *= oneOverMag;
    v[2] *= oneOverMag;
  }
}

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for glRotatef. */
static void makeRotateMatrix(float angle,
                             float ax, float ay, float az,
                             float m[16])
{
  float radians, sine, cosine, ab, bc, ca, tx, ty, tz;
  float axis[3];

  axis[0] = ax;
  axis[1] = ay;
  axis[2] = az;
  normalizeVector(axis);

  radians = angle * myPi / 180.0;
  sine = sin(radians);
  cosine = cos(radians);
  ab = axis[0] * axis[1] * (1 - cosine);
  bc = axis[1] * axis[2] * (1 - cosine);
  ca = axis[2] * axis[0] * (1 - cosine);
  tx = axis[0] * axis[0];
  ty = axis[1] * axis[1];
  tz = axis[2] * axis[2];

  m[0]  = tx + cosine * (1 - tx);
  m[1]  = ab + axis[2] * sine;
  m[2]  = ca - axis[1] * sine;
  m[3]  = 0.0f;
  m[4]  = ab - axis[2] * sine;
  m[5]  = ty + cosine * (1 - ty);
  m[6]  = bc + axis[0] * sine;
  m[7]  = 0.0f;
  m[8]  = ca + axis[1] * sine;
  m[9]  = bc - axis[0] * sine;
  m[10] = tz + cosine * (1 - tz);
  m[11] = 0;
  m[12] = 0;
  m[13] = 0;
  m[14] = 0;
  m[15] = 1;
}

static void makeClipToTextureMatrix(float m[16])
{
  m[0]  = 0.5f;  m[1]  = 0;     m[2]  = 0;     m[3]  = 0.5f;
  m[4]  = 0;     m[5]  = 0.5f;  m[6]  = 0;     m[7]  = 0.5f;
  m[8]  = 0;     m[9]  = 0;     m[10] = 0.5f;  m[11] = 0.5f;
  m[12] = 0;     m[13] = 0;     m[14] = 0;     m[15] = 1;
}

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for glTranslatef. */
static void makeTranslateMatrix(float x, float y, float z, float m[16])
{
  m[0]  = 1;  m[1]  = 0;  m[2]  = 0;  m[3]  = x;
  m[4]  = 0;  m[5]  = 1;  m[6]  = 0;  m[7]  = y;
  m[8]  = 0;  m[9]  = 0;  m[10] = 1;  m[11] = z;
  m[12] = 0;  m[13] = 0;  m[14] = 0;  m[15] = 1;
}

/* Simple 4x4 matrix by 4x4 matrix multiply. */
static void multMatrix(float dst[16],
                       const float src1[16], const float src2[16])
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

/* Invert a row-major (C-style) 4x4 matrix. */
static void invertMatrix(float *out, const float *m)
{
/* Assumes matrices are ROW major. */
#define SWAP_ROWS(a, b) { GLdouble *_tmp = a; (a)=(b); (b)=_tmp; }
#define MAT(m,r,c) (m)[(r)*4+(c)]

  double wtmp[4][8];
  double m0, m1, m2, m3, s;
  double *r0, *r1, *r2, *r3;

  r0 = wtmp[0], r1 = wtmp[1], r2 = wtmp[2], r3 = wtmp[3];

  r0[0] = MAT(m,0,0), r0[1] = MAT(m,0,1),
  r0[2] = MAT(m,0,2), r0[3] = MAT(m,0,3),
  r0[4] = 1.0, r0[5] = r0[6] = r0[7] = 0.0,

  r1[0] = MAT(m,1,0), r1[1] = MAT(m,1,1),
  r1[2] = MAT(m,1,2), r1[3] = MAT(m,1,3),
  r1[5] = 1.0, r1[4] = r1[6] = r1[7] = 0.0,

  r2[0] = MAT(m,2,0), r2[1] = MAT(m,2,1),
  r2[2] = MAT(m,2,2), r2[3] = MAT(m,2,3),
  r2[6] = 1.0, r2[4] = r2[5] = r2[7] = 0.0,

  r3[0] = MAT(m,3,0), r3[1] = MAT(m,3,1),
  r3[2] = MAT(m,3,2), r3[3] = MAT(m,3,3),
  r3[7] = 1.0, r3[4] = r3[5] = r3[6] = 0.0;

  /* Choose myPivot, or die. */
  if (fabs(r3[0])>fabs(r2[0])) SWAP_ROWS(r3, r2);
  if (fabs(r2[0])>fabs(r1[0])) SWAP_ROWS(r2, r1);
  if (fabs(r1[0])>fabs(r0[0])) SWAP_ROWS(r1, r0);
  if (0.0 == r0[0]) {
    assert(!"could not invert matrix");
  }

  /* Eliminate first variable. */
  m1 = r1[0]/r0[0]; m2 = r2[0]/r0[0]; m3 = r3[0]/r0[0];
  s = r0[1]; r1[1] -= m1 * s; r2[1] -= m2 * s; r3[1] -= m3 * s;
  s = r0[2]; r1[2] -= m1 * s; r2[2] -= m2 * s; r3[2] -= m3 * s;
  s = r0[3]; r1[3] -= m1 * s; r2[3] -= m2 * s; r3[3] -= m3 * s;
  s = r0[4];
  if (s != 0.0) { r1[4] -= m1 * s; r2[4] -= m2 * s; r3[4] -= m3 * s; }
  s = r0[5];
  if (s != 0.0) { r1[5] -= m1 * s; r2[5] -= m2 * s; r3[5] -= m3 * s; }
  s = r0[6];
  if (s != 0.0) { r1[6] -= m1 * s; r2[6] -= m2 * s; r3[6] -= m3 * s; }
  s = r0[7];
  if (s != 0.0) { r1[7] -= m1 * s; r2[7] -= m2 * s; r3[7] -= m3 * s; }

  /* Choose myPivot, or die. */
  if (fabs(r3[1])>fabs(r2[1])) SWAP_ROWS(r3, r2);
  if (fabs(r2[1])>fabs(r1[1])) SWAP_ROWS(r2, r1);
  if (0.0 == r1[1]) {
    assert(!"could not invert matrix");
  }

  /* Eliminate second variable. */
  m2 = r2[1]/r1[1]; m3 = r3[1]/r1[1];
  r2[2] -= m2 * r1[2]; r3[2] -= m3 * r1[2];
  r2[3] -= m2 * r1[3]; r3[3] -= m3 * r1[3];
  s = r1[4]; if (0.0 != s) { r2[4] -= m2 * s; r3[4] -= m3 * s; }
  s = r1[5]; if (0.0 != s) { r2[5] -= m2 * s; r3[5] -= m3 * s; }
  s = r1[6]; if (0.0 != s) { r2[6] -= m2 * s; r3[6] -= m3 * s; }
  s = r1[7]; if (0.0 != s) { r2[7] -= m2 * s; r3[7] -= m3 * s; }

  /* Choose myPivot, or die. */
  if (fabs(r3[2])>fabs(r2[2])) SWAP_ROWS(r3, r2);
  if (0.0 == r2[2]) {
    assert(!"could not invert matrix");
  }

  /* Eliminate third variable. */
  m3 = r3[2]/r2[2];
  r3[3] -= m3 * r2[3], r3[4] -= m3 * r2[4],
  r3[5] -= m3 * r2[5], r3[6] -= m3 * r2[6],
  r3[7] -= m3 * r2[7];

  /* Last check. */
  if (0.0 == r3[3]) {
    assert(!"could not invert matrix");
  }

  s = 1.0/r3[3];              /* Now back substitute row 3. */
  r3[4] *= s; r3[5] *= s; r3[6] *= s; r3[7] *= s;

  m2 = r2[3];                 /* Now back substitute row 2. */
  s  = 1.0/r2[2];
  r2[4] = s * (r2[4] - r3[4] * m2), r2[5] = s * (r2[5] - r3[5] * m2),
  r2[6] = s * (r2[6] - r3[6] * m2), r2[7] = s * (r2[7] - r3[7] * m2);
  m1 = r1[3];
  r1[4] -= r3[4] * m1, r1[5] -= r3[5] * m1,
  r1[6] -= r3[6] * m1, r1[7] -= r3[7] * m1;
  m0 = r0[3];
  r0[4] -= r3[4] * m0, r0[5] -= r3[5] * m0,
  r0[6] -= r3[6] * m0, r0[7] -= r3[7] * m0;

  m1 = r1[2];                 /* Now back substitute row 1. */
  s  = 1.0/r1[1];
  r1[4] = s * (r1[4] - r2[4] * m1), r1[5] = s * (r1[5] - r2[5] * m1),
  r1[6] = s * (r1[6] - r2[6] * m1), r1[7] = s * (r1[7] - r2[7] * m1);
  m0 = r0[2];
  r0[4] -= r2[4] * m0, r0[5] -= r2[5] * m0,
  r0[6] -= r2[6] * m0, r0[7] -= r2[7] * m0;

  m0 = r0[1];                 /* Now back substitute row 0. */
  s  = 1.0/r0[0];
  r0[4] = s * (r0[4] - r1[4] * m0), r0[5] = s * (r0[5] - r1[5] * m0),
  r0[6] = s * (r0[6] - r1[6] * m0), r0[7] = s * (r0[7] - r1[7] * m0);

  MAT(out,0,0) = r0[4]; MAT(out,0,1) = r0[5],
  MAT(out,0,2) = r0[6]; MAT(out,0,3) = r0[7],
  MAT(out,1,0) = r1[4]; MAT(out,1,1) = r1[5],
  MAT(out,1,2) = r1[6]; MAT(out,1,3) = r1[7],
  MAT(out,2,0) = r2[4]; MAT(out,2,1) = r2[5],
  MAT(out,2,2) = r2[6]; MAT(out,2,3) = r2[7],
  MAT(out,3,0) = r3[4]; MAT(out,3,1) = r3[5],
  MAT(out,3,2) = r3[6]; MAT(out,3,3) = r3[7]; 

#undef MAT
#undef SWAP_ROWS
}

/* Simple 4x4 matrix by 4-component column vector multiply. */
static void transform(float dst[4],
                      const float mat[16], const float vec[4])
{
  double tmp[4], invW;
  int i;

  for (i=0; i<4; i++) {
    tmp[i] = mat[i*4+0] * vec[0] +
             mat[i*4+1] * vec[1] +
             mat[i*4+2] * vec[2] +
             mat[i*4+3] * vec[3];
  }
  invW = 1 / tmp[3];
  /* Apply perspective divide and copy to dst (so dst can vec). */
  for (i=0; i<3; i++)
    dst[i] = tmp[i] * invW;
  dst[3] = 1;
}

void
loadMVP(const float modelView[16])
{
  float transpose[16];
  int i, j;

  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      transpose[i*4+j] = modelView[j*4+i];
    }
  }
  glLoadMatrixf(transpose);
}

static void buildTextureMatrix(const float viewMatrix[16],
                               const float modelMatrix[16],
                               float textureMatrix[16])
{
  static float eyeToClipMatrix[16];
  float modelViewMatrix[16];
  static int needsInit = 1;

  if (needsInit) {
    const float fieldOfView = 50.0f;
    const float aspectRatio = 1;
    float textureProjectionMatrix[16];
    float clipToTextureMatrix[16];

    /* Build texture projection matrix once. */
    buildPerspectiveMatrix(fieldOfView, aspectRatio,
                           0.25, 20.0,  /* Znear and Zfar */
                           textureProjectionMatrix);

    makeClipToTextureMatrix(clipToTextureMatrix);

    /* eyeToClip = clipToTexture * textureProjection */
    multMatrix(eyeToClipMatrix,
      clipToTextureMatrix, textureProjectionMatrix);
    needsInit = 1;
  }

  /* modelView = view * model */
  multMatrix(modelViewMatrix, viewMatrix, modelMatrix);
  /* texture = eyeToClip * modelView */
  multMatrix(textureMatrix, eyeToClipMatrix, modelViewMatrix);
}

static void display(void)
{
  /* World-space positions for light and eye. */
  const float eyePosition[4] = { 10*sin(eyeAngle), 
                                 eyeHeight,
                                 10*cos(eyeAngle), 1 };
  const float lightPosition[4] = { 4.5*sin(lightAngle), 
                                   lightHeight,
                                   4.5*cos(lightAngle), 1 };

  static const float center[3] = { 0, 0, 0 };  /* eye and light point at the origin */
  static const float up[3] = { 0, 1, 0 };      /* up is positive Y direction */

  float translateMatrix[16], rotateMatrix[16],
        modelMatrix[16], invModelMatrix[16], eyeViewMatrix[16],
        modelViewMatrix[16], modelViewProjMatrix[16], textureMatrix[16],
        lightViewMatrix[16];
  float objSpaceLightPosition[4];

  buildLookAtMatrix(eyePosition[0], eyePosition[1], eyePosition[2],
                    center[0], center[1], center[2],
                    up[0], up[1], up[2],
                    eyeViewMatrix);

  buildLookAtMatrix(lightPosition[0], lightPosition[1], lightPosition[2],
                    center[0], center[1], center[2],
                    up[0], -up[1], up[2],  /* Flip up for projected texture's view */
                    lightViewMatrix);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  cgGLBindProgram(myCgVertexProgram);
  checkForCgError("binding vertex program");

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  cgGLBindProgram(myCgFragmentProgram);
  checkForCgError("binding fragment program");

  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  /*** Render solid sphere ***/

  /* modelView = rotateMatrix * translateMatrix */
  makeRotateMatrix(70, 1, 1, 1, rotateMatrix);
  makeTranslateMatrix(2, 0, 0, translateMatrix);
  multMatrix(modelMatrix, translateMatrix, rotateMatrix);

  /* invModelMatrix = inverse(modelMatrix) */
  invertMatrix(invModelMatrix, modelMatrix);

  /* Transform world-space light positions to sphere's object-space. */
  transform(objSpaceLightPosition, invModelMatrix, lightPosition);
  cgSetParameter3fv(myCgVertexParam_lightPosition, objSpaceLightPosition);

  /* modelViewMatrix = eyeViewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, eyeViewMatrix, modelMatrix);

  /* modelViewProj = projectionMatrix * modelViewMatrix */
  multMatrix(modelViewProjMatrix, projectionMatrix, modelViewMatrix);

  buildTextureMatrix(lightViewMatrix, modelMatrix, /*out*/textureMatrix);

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgSetMatrixParameterfr(myCgVertexParam_textureMatrix, textureMatrix);
  cgUpdateProgramParameters(myCgVertexProgram);
  cgUpdateProgramParameters(myCgFragmentProgram);
  glutSolidSphere(2.0, 40, 40);

  /*** Render solid cube ***/

  /* modelView = eyeViewMatrix * translateMatrix */
  makeTranslateMatrix(-2,  1.0, 0, translateMatrix);
  makeRotateMatrix(90, 1, 0, 0, rotateMatrix);
  multMatrix(modelMatrix, translateMatrix, rotateMatrix);

  /* invModelMatrix = inverse(modelMatrix) */
  invertMatrix(invModelMatrix, modelMatrix);

  /* Transform world-space light positions to sphere's object-space. */
  transform(objSpaceLightPosition, invModelMatrix, lightPosition);
  cgSetParameter3fv(myCgVertexParam_lightPosition, objSpaceLightPosition);

  /* modelViewMatrix = eyeViewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, eyeViewMatrix, modelMatrix);

  /* modelViewProj = projectionMatrix * modelViewMatrix */
  multMatrix(modelViewProjMatrix, projectionMatrix, modelViewMatrix);

  buildTextureMatrix(lightViewMatrix, modelMatrix, /*out*/textureMatrix);

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgSetMatrixParameterfr(myCgVertexParam_textureMatrix, textureMatrix);
  cgUpdateProgramParameters(myCgVertexProgram);
  cgUpdateProgramParameters(myCgFragmentProgram);
  glutSolidCube(2.5);

  /*** Render floor ***/

  /* modelViewProj = projection * view */
  multMatrix(modelViewProjMatrix, projectionMatrix, eyeViewMatrix);

  /* Identity modeling. */
  makeTranslateMatrix(0, 0, 0, modelMatrix);
  buildTextureMatrix(lightViewMatrix, modelMatrix, /*out*/textureMatrix);

  cgSetParameter3fv(myCgVertexParam_lightPosition, lightPosition);
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgSetMatrixParameterfr(myCgVertexParam_textureMatrix, textureMatrix);

  cgUpdateProgramParameters(myCgVertexProgram);
  cgUpdateProgramParameters(myCgFragmentProgram);

  glEnable(GL_CULL_FACE);

  glBegin(GL_QUADS);
    glNormal3f(0, 1, 0);
    glVertex3f( 12, -2, -12);
    glVertex3f(-12, -2, -12);
    glVertex3f(-12, -2,  12);
    glVertex3f( 12, -2,  12);

    glNormal3f(0, 0, 1);
    glVertex3f(-12, -2, -12);
    glVertex3f( 12, -2, -12);
    glVertex3f( 12, 10, -12);
    glVertex3f(-12, 10, -12);

    glNormal3f(0, 0, -1);
    glVertex3f( 12, -2,  12);
    glVertex3f(-12, -2,  12);
    glVertex3f(-12, 10,  12);
    glVertex3f( 12, 10,  12);

    glNormal3f(0, -1, 0);
    glVertex3f(-12, 10, -12);
    glVertex3f( 12, 10, -12);
    glVertex3f( 12, 10,  12);
    glVertex3f(-12, 10,  12);

    glNormal3f(1, 0, 0);
    glVertex3f(-12, -2,  12);
    glVertex3f(-12, -2, -12);
    glVertex3f(-12, 10, -12);
    glVertex3f(-12, 10,  12);

    glNormal3f(-1, 0, 0);
    glVertex3f(12, -2, -12);
    glVertex3f(12, -2,  12);
    glVertex3f(12, 10,  12);
    glVertex3f(12, 10, -12);
  glEnd();

  glDisable(GL_CULL_FACE);

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  /*** Render light as white cone ***/

  /* modelView = translateMatrix */
  makeTranslateMatrix(lightPosition[0], lightPosition[1], lightPosition[2],
    modelMatrix);

  /* modelViewMatrix = eyeViewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, eyeViewMatrix, modelMatrix);

  glPushMatrix();
    /* glLoadMatrixf expects a column-major matrix but Cg matrices are
       row-major (matching C/C++ arrays) so used loadMVP to transpose
       the Cg version. */
    loadMVP(modelViewMatrix);
    glColor3f(1,1,1); /* make sure current color is white */
    glutSolidSphere(0.15, 30, 30);
  glPopMatrix();

  glutSwapBuffers();
}

static void idle(void)
{
  lightAngle += 0.008;  /* Add a small angle (in radians). */
  if (lightAngle > 2*myPi) {
    lightAngle -= 2*myPi;
  }
  glutPostRedisplay();
}


static void keyboard(unsigned char c, int x, int y)
{
  static int animating = 0,
             repeat = 0;

  switch (c) {
  case ' ':
    animating = !animating; /* Toggle */
    if (animating) {
      glutIdleFunc(idle);
    } else {
      glutIdleFunc(NULL);
    }    
    break;
  case 'r':
    repeat = !repeat;
    if (repeat) {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    } else {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    }
    glutPostRedisplay();
    break;
  case 27:  /* Esc key */
    /* Demonstrate proper deallocation of Cg runtime data structures.
       Not strictly necessary if we are simply going to exit. */
    cgDestroyProgram(myCgVertexProgram);
    cgDestroyContext(myCgContext);
    exit(0);
    break;
  }
}

static void menu(int item)
{
  /* Pass menu item character code to keyboard callback. */
  keyboard((unsigned char)item, 0, 0);
}

/* Use a motion and mouse GLUT callback to allow the viewer and light to
   rotate around the monkey head and move the viewer up and down. */

static int beginx, beginy;
static int moving = 0;
static int movingLight = 0;
static int xLightBegin, yLightBegin;

void
motion(int x, int y)
{
  const float heightMax = 10,
              heightMin = -1.5;

  if (moving) {
    eyeAngle += 0.005*(beginx - x);
    eyeHeight += 0.03*(y - beginy);
    if (eyeHeight > heightMax) {
      eyeHeight = heightMax;
    }
    if (eyeHeight < heightMin) {
      eyeHeight = heightMin;
    }
    beginx = x;
    beginy = y;
    glutPostRedisplay();
  }
  if (movingLight) {
    lightAngle += 0.005*(x - xLightBegin);
    lightHeight += 0.03*(yLightBegin - y);
    xLightBegin = x;
    yLightBegin = y;
    glutPostRedisplay();
  }
}

void
mouse(int button, int state, int x, int y)
{
  const int spinButton = GLUT_LEFT_BUTTON,
            lightButton = GLUT_MIDDLE_BUTTON;

  if (button == spinButton && state == GLUT_DOWN) {
    moving = 1;
    beginx = x;
    beginy = y;
  }
  if (button == spinButton && state == GLUT_UP) {
    moving = 0;
  }
  if (button == lightButton && state == GLUT_DOWN) {
    movingLight = 1;
    xLightBegin = x;
    yLightBegin = y;
  }
  if (button == lightButton && state == GLUT_UP) {
    movingLight = 0;
  }
}

/* Platform-specific code to request synchronized buffer swaps. */

static void requestSynchronizedSwapBuffers(void)
{
#if defined(__APPLE__)
#ifdef CGL_VERSION_1_2
  const GLint sync = 1;
#else
  const long sync = 1;
#endif
  CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &sync);

#elif defined(_WIN32)
  if (wglSwapIntervalEXT) {
    wglSwapIntervalEXT(1);
  }
#else
  if (glXSwapIntervalSGI) {
    glXSwapIntervalSGI(1);
  }
#endif
}
