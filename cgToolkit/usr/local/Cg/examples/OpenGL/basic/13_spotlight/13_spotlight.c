
/* 13_spotlight.c - OpenGL-based spotlight attenuation example
   using Cg program from Chapter 5 of "The Cg Tutorial" (Addison-Wesley,
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
                   myCgFragmentParam_globalAmbient,
                   /* One set of light parameters */
                   myCgFragmentParam_lightColor[1],
                   myCgFragmentParam_lightPosition[1],
                   myCgFragmentParam_lightConstantAtten[1],
                   myCgFragmentParam_lightLinearAtten[1],
                   myCgFragmentParam_lightQuadraticAtten[1],
                   myCgFragmentParam_lightDirection[1],
                   myCgFragmentParam_lightCosInnerCone[1],
                   myCgFragmentParam_lightCosOuterCone[1],
                   myCgFragmentParam_eyePosition,
                   myCgFragmentParam_material_Ke,
                   myCgFragmentParam_material_Ka,
                   myCgFragmentParam_material_Kd,
                   myCgFragmentParam_material_Ks,
                   myCgFragmentParam_material_shininess;

static const char *myProgramName = "13_spotlight",
                  *myVertexProgramFileName = "C5E2v_fragmentLighting.cg",
/* Page 124 */    *myVertexProgramName = "C5E2v_fragmentLighting",
                  *myFragmentProgramFileName = "C5E10_spotAttenLighting.cg",
/* Page 136 */    *myFragmentProgramName = "oneLight";

static float myLightAngle = -0.4;   /* Angle light rotates around scene. */
static float myProjectionMatrix[16];
static float myGlobalAmbient[3] = { 0.4, 0.4, 0.4 };  /* Dim */
static float myLightColor[3] = { 1, 1, 1 };  /* White */

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
static void requestSynchronizedSwapBuffers(void);

int main(int argc, char **argv)
{
  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutReshapeFunc(reshape);

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

#define GET_FRAGMENT_PARAM(name) \
  myCgFragmentParam_##name = \
    cgGetNamedParameter(myCgFragmentProgram, #name); \
  checkForCgError("could not get " #name " parameter");

  GET_FRAGMENT_PARAM(globalAmbient);
  GET_FRAGMENT_PARAM(eyePosition);

#define GET_FRAGMENT_PARAM2(varname, cgname) \
  myCgFragmentParam_##varname = \
    cgGetNamedParameter(myCgFragmentProgram, cgname); \
  checkForCgError("could not get " cgname " parameter");

  GET_FRAGMENT_PARAM2(lightPosition[0],       "lights[0].position");
  GET_FRAGMENT_PARAM2(lightColor[0],          "lights[0].color");
  GET_FRAGMENT_PARAM2(lightConstantAtten[0],  "lights[0].kC");
  GET_FRAGMENT_PARAM2(lightLinearAtten[0],    "lights[0].kL");
  GET_FRAGMENT_PARAM2(lightQuadraticAtten[0], "lights[0].kQ");
  GET_FRAGMENT_PARAM2(lightDirection[0],      "lights[0].direction");
  GET_FRAGMENT_PARAM2(lightCosInnerCone[0],   "lights[0].cosInnerCone");
  GET_FRAGMENT_PARAM2(lightCosOuterCone[0],   "lights[0].cosOuterCone");

  GET_FRAGMENT_PARAM2(material_Ke, "material.Ke");
  GET_FRAGMENT_PARAM2(material_Ka, "material.Ka");
  GET_FRAGMENT_PARAM2(material_Kd, "material.Kd");
  GET_FRAGMENT_PARAM2(material_Ks, "material.Ks");
  GET_FRAGMENT_PARAM2(material_shininess, "material.shininess");

  /* Set light source color parameters once. */
  cgSetParameter3fv(myCgFragmentParam_globalAmbient, myGlobalAmbient);
  cgSetParameter3fv(myCgFragmentParam_lightColor[0], myLightColor);

  cgSetParameter1f(myCgFragmentParam_lightConstantAtten[0], 1);
  cgSetParameter1f(myCgFragmentParam_lightLinearAtten[0], 0.0);
  cgSetParameter1f(myCgFragmentParam_lightQuadraticAtten[0], 0.0001);

  cgSetParameter1f(myCgFragmentParam_lightCosInnerCone[0], 0.95);
  cgSetParameter1f(myCgFragmentParam_lightCosOuterCone[0], 0.85);

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
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

  /* Build projection matrix once. */
  buildPerspectiveMatrix(fieldOfView, aspectRatio,
                         1.0, 100.0,  /* Znear and Zfar */
                         myProjectionMatrix);
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

/* Simple 4x4 matrix by 4-component column vector multiply. */
static void transformDirection(float dst[3],
                               const float mat[16],
                               const float vec[3])
{
  double tmp[3];
  int i;

  for (i=0; i<3; i++) {
    tmp[i] = mat[i*4+0] * vec[0] +
             mat[i*4+1] * vec[1] +
             mat[i*4+2] * vec[2];
  }
  for (i=0; i<3; i++)
    dst[i] = tmp[i];
}

static void setBrassMaterial(void)
{
  const float brassEmissive[3] = {0.0,  0.0,  0.0},
              brassAmbient[3]  = {0.33, 0.22, 0.03},
              brassDiffuse[3]  = {0.78, 0.57, 0.11},
              brassSpecular[3] = {0.99, 0.91, 0.81},
              brassShininess = 27.8;

  cgSetParameter3fv(myCgFragmentParam_material_Ke, brassEmissive);
  cgSetParameter3fv(myCgFragmentParam_material_Ka, brassAmbient);
  cgSetParameter3fv(myCgFragmentParam_material_Kd, brassDiffuse);
  cgSetParameter3fv(myCgFragmentParam_material_Ks, brassSpecular);
  cgSetParameter1f(myCgFragmentParam_material_shininess, brassShininess);
}

static void setRedPlasticMaterial(void)
{
  const float redPlasticEmissive[3] = {0.0,  0.0,  0.0},
              redPlasticAmbient[3]  = {0.0, 0.0, 0.0},
              redPlasticDiffuse[3]  = {0.5, 0.0, 0.0},
              redPlasticSpecular[3] = {0.7, 0.6, 0.6},
              redPlasticShininess = 32.0;

  cgSetParameter3fv(myCgFragmentParam_material_Ke, redPlasticEmissive);
  checkForCgError("setting Ke parameter");
  cgSetParameter3fv(myCgFragmentParam_material_Ka, redPlasticAmbient);
  checkForCgError("setting Ka parameter");
  cgSetParameter3fv(myCgFragmentParam_material_Kd, redPlasticDiffuse);
  checkForCgError("setting Kd parameter");
  cgSetParameter3fv(myCgFragmentParam_material_Ks, redPlasticSpecular);
  checkForCgError("setting Ks parameter");
  cgSetParameter1f(myCgFragmentParam_material_shininess, redPlasticShininess);
  checkForCgError("setting shininess parameter");
}

static void setGreenEmeraldMaterial(void)
{
  const float greenEmeraldEmissive[3] = {0.0,  0.0,  0.0},
              greenEmeraldAmbient[3]  = {0.0215, 0.1745, 0.0215},
              greenEmeraldDiffuse[3]  = {0.07568, 0.61424, 0.07568},
              greenEmeraldSpecular[3] = {0.633, 0.727811, 0.633},
              greenEmeraldShininess = 76.8;

  cgSetParameter3fv(myCgFragmentParam_material_Ke, greenEmeraldEmissive);
  checkForCgError("setting Ke parameter");
  cgSetParameter3fv(myCgFragmentParam_material_Ka, greenEmeraldAmbient);
  checkForCgError("setting Ka parameter");
  cgSetParameter3fv(myCgFragmentParam_material_Kd, greenEmeraldDiffuse);
  checkForCgError("setting Kd parameter");
  cgSetParameter3fv(myCgFragmentParam_material_Ks, greenEmeraldSpecular);
  checkForCgError("setting Ks parameter");
  cgSetParameter1f(myCgFragmentParam_material_shininess, greenEmeraldShininess);
  checkForCgError("setting shininess parameter");
}

static void setEmissiveLightColorOnly(void)
{
  const float zero[3] = {0.0,  0.0,  0.0};

  cgSetParameter3fv(myCgFragmentParam_material_Ke, myLightColor);
  checkForCgError("setting Ke parameter");
  cgSetParameter3fv(myCgFragmentParam_material_Ka, zero);
  checkForCgError("setting Ka parameter");
  cgSetParameter3fv(myCgFragmentParam_material_Kd, zero);
  checkForCgError("setting Kd parameter");
  cgSetParameter3fv(myCgFragmentParam_material_Ks, zero);
  checkForCgError("setting Ks parameter");
  cgSetParameter1f(myCgFragmentParam_material_shininess, 0);
  checkForCgError("setting shininess parameter");
}

static void display(void)
{
  /* World-space positions for light and eye. */
  const float eyePosition[4] = { 0, 0, 13, 1 };
  const float lightPosition[4] = { 5*sin(myLightAngle), 
                                   1.5,
                                   5*cos(myLightAngle), 1 };
  const float lightDirection[3] = { -lightPosition[0],
                                    -lightPosition[1],
                                    -lightPosition[2] };

  float translateMatrix[16], rotateMatrix[16],
        modelMatrix[16], invModelMatrix[16], viewMatrix[16],
        modelViewMatrix[16], modelViewProjMatrix[16];
  float objSpaceEyePosition[4], objSpaceLightPosition[4],
        objSpaceLightDirection[3];

  buildLookAtMatrix(eyePosition[0], eyePosition[1], eyePosition[2],
                    0, 0, 0,
                    0, 1, 0,
                    viewMatrix);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  cgGLBindProgram(myCgVertexProgram);
  checkForCgError("binding vertex program");

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  cgGLBindProgram(myCgFragmentProgram);
  checkForCgError("binding fragment program");

  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  /*** Render brass solid sphere ***/

  setBrassMaterial();

  /* modelView = rotateMatrix * translateMatrix */
  makeRotateMatrix(70, 1, 1, 1, rotateMatrix);
  makeTranslateMatrix(2, 0, 0, translateMatrix);
  multMatrix(modelMatrix, translateMatrix, rotateMatrix);

  /* invModelMatrix = inverse(modelMatrix) */
  invertMatrix(invModelMatrix, modelMatrix);

  /* Transform world-space eye and light positions to sphere's object-space. */
  transform(objSpaceEyePosition, invModelMatrix, eyePosition);
  cgSetParameter3fv(myCgFragmentParam_eyePosition, objSpaceEyePosition);
  transform(objSpaceLightPosition, invModelMatrix, lightPosition);
  cgSetParameter3fv(myCgFragmentParam_lightPosition[0], objSpaceLightPosition);
  transformDirection(objSpaceLightDirection, invModelMatrix, lightDirection);
  normalizeVector(objSpaceLightDirection);
  cgSetParameter3fv(myCgFragmentParam_lightDirection[0], objSpaceLightDirection);

  /* modelViewMatrix = viewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, viewMatrix, modelMatrix);

  /* modelViewProj = projectionMatrix * modelViewMatrix */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, modelViewMatrix);

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgUpdateProgramParameters(myCgVertexProgram);
  cgUpdateProgramParameters(myCgFragmentProgram);
  glutSolidSphere(2.0, 40, 40);

  /*** Render red plastic solid cone ***/

  setRedPlasticMaterial();

  /* modelView = viewMatrix * translateMatrix */
  makeTranslateMatrix(-2, -1.5, 0, translateMatrix);
  makeRotateMatrix(90, 1, 0, 0, rotateMatrix);
  multMatrix(modelMatrix, translateMatrix, rotateMatrix);

  /* invModelMatrix = inverse(modelMatrix) */
  invertMatrix(invModelMatrix, modelMatrix);

  /* Transform world-space eye and light positions to sphere's object-space. */
  transform(objSpaceEyePosition, invModelMatrix, eyePosition);
  cgSetParameter3fv(myCgFragmentParam_eyePosition, objSpaceEyePosition);
  transform(objSpaceLightPosition, invModelMatrix, lightPosition);
  cgSetParameter3fv(myCgFragmentParam_lightPosition[0], objSpaceLightPosition);
  transformDirection(objSpaceLightDirection, invModelMatrix, lightDirection);
  normalizeVector(objSpaceLightDirection);
  cgSetParameter3fv(myCgFragmentParam_lightDirection[0], objSpaceLightDirection);

  /* modelViewMatrix = viewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, viewMatrix, modelMatrix);

  /* modelViewProj = projectionMatrix * modelViewMatrix */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, modelViewMatrix);

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgUpdateProgramParameters(myCgVertexProgram);
  cgUpdateProgramParameters(myCgFragmentProgram);
  glutSolidCone(1.5, 3.5, 30, 30);

  /*** Render green emerald floor ***/

  setGreenEmeraldMaterial();

  /* modelViewProj = projection * view */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, viewMatrix);

  cgSetParameter3fv(myCgFragmentParam_eyePosition, eyePosition);
  cgSetParameter3fv(myCgFragmentParam_lightPosition[0], lightPosition);
  objSpaceLightDirection[0] = lightDirection[0];
  objSpaceLightDirection[1] = lightDirection[1];
  objSpaceLightDirection[2] = lightDirection[2];
  normalizeVector(objSpaceLightDirection);
  cgSetParameter3fv(myCgFragmentParam_lightDirection[0], objSpaceLightDirection);
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);

  cgUpdateProgramParameters(myCgVertexProgram);
  cgUpdateProgramParameters(myCgFragmentProgram);

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

  /*** Render light as emissive white cone ***/

  /* modelView = translateMatrix */
  makeTranslateMatrix(lightPosition[0], lightPosition[1], lightPosition[2],
    modelMatrix);

  /* modelViewMatrix = viewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, viewMatrix, modelMatrix);

  /* modelViewProj = projectionMatrix * modelViewMatrix */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, modelViewMatrix);

  setEmissiveLightColorOnly();
  /* Avoid degenerate lightPosition. */
  cgSetParameter3f(myCgFragmentParam_lightPosition[0], 0,0,0);

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgUpdateProgramParameters(myCgVertexProgram);
  cgUpdateProgramParameters(myCgFragmentProgram);
  glutSolidCone(0.15, 0.95, 30, 30);

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  glutSwapBuffers();
}

static void idle(void)
{
  myLightAngle += 0.008;  /* Add a small angle (in radians). */
  if (myLightAngle > 2*myPi) {
    myLightAngle -= 2*myPi;
  }
  glutPostRedisplay();
}


static void keyboard(unsigned char c, int x, int y)
{
  static int animating = 0;

  switch (c) {
  case ' ':
    animating = !animating; /* Toggle */
    if (animating) {
      glutIdleFunc(idle);
    } else {
      glutIdleFunc(NULL);
    }    
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
