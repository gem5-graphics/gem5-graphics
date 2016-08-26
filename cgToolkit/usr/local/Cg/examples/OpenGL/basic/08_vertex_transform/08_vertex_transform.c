
/* 08_vertex_transform.c - OpenGL-based vertex transform example
   using Cg program from Chapter 4 of "The Cg Tutorial" (Addison-Wesley,
   ISBN 0321194969). */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   1.5 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sqrt, sin, and cos */
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
                   myCgFragmentParam_c;

static const char *myProgramName = "08_vertex_transform",
                  *myVertexProgramFileName = "C4E1v_transform.cg",
/* Page 97 */     *myVertexProgramName = "C4E1v_transform";

static float myEyeAngle = 0;   /* Angle eye rotates around scene. */
static float myProjectionMatrix[16];

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
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
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

  myCgVertexParam_modelViewProj =
    cgGetNamedParameter(myCgVertexProgram, "modelViewProj");
  checkForCgError("could not get modelViewProj parameter");

  myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  cgGLSetOptimalOptions(myCgFragmentProfile);
  checkForCgError("selecting fragment profile");

  /* Specify fragment program with a string. */
  myCgFragmentProgram =
    cgCreateProgram(
      myCgContext,              /* Cg runtime context */
      CG_SOURCE,                /* Program in human-readable form */
      "float4 main(uniform float4 c) : COLOR { return c; }",
      myCgFragmentProfile,      /* Profile: latest fragment profile */
      "main",                   /* Entry function name */
      NULL); /* No extra compiler options */
  checkForCgError("creating fragment program from string");
  cgGLLoadProgram(myCgFragmentProgram);
  checkForCgError("loading fragment program");

  myCgFragmentParam_c =
    cgGetNamedParameter(myCgFragmentProgram, "c");
  checkForCgError("could not get c parameter");

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

/* Forward declared routine used by reshape callback. */
static void buildPerspectiveMatrix(double, double,
                                   double, double, float m[16]);

static void reshape(int width, int height)
{
  double aspectRatio = (float) width / (float) height;
  double fieldOfView = 40.0; /* Degrees */

  /* Build projection matrix once. */
  buildPerspectiveMatrix(fieldOfView, aspectRatio,
                         1.0, 20.0,  /* Znear and Zfar */
                         myProjectionMatrix);
  glViewport(0, 0, width, height);
}

static const double myPi = 3.14159265358979323846;

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

static void makeRotateMatrix(float angle,
                             float ax, float ay, float az,
                             float m[16])
{
  float radians, sine, cosine, ab, bc, ca, tx, ty, tz;
  float axis[3];
  float mag;

  axis[0] = ax;
  axis[1] = ay;
  axis[2] = az;
  mag = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
  if (mag) {
    axis[0] /= mag;
    axis[1] /= mag;
    axis[2] /= mag;
  }

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

static void display(void)
{
  float translateMatrix[16], rotateMatrix[16],
        modelMatrix[16], viewMatrix[16],
        modelViewMatrix[16], modelViewProjMatrix[16];

  buildLookAtMatrix(13*sin(myEyeAngle), 0, 13*cos(myEyeAngle),  /* eye position */
                    0, 0, 0, /* view center */
                    0, 1, 0, /* up vector */
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

  /*** Render green wireframe sphere ***/

  /* modelView = rotateMatrix * translateMatrix */
  makeRotateMatrix(70, 1, 1, 1, rotateMatrix);
  makeTranslateMatrix(2, 0, 0, translateMatrix);
  multMatrix(modelMatrix, translateMatrix, rotateMatrix);

  /* modelViewMatrix = viewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, viewMatrix, modelMatrix);

  /* modelViewProj = projectionMatrix * modelViewMatrix */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, modelViewMatrix);

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgSetParameter4f(myCgFragmentParam_c, 0.1,0.7,0.1,1);  /* Green */
  cgUpdateProgramParameters(myCgVertexProgram);
  cgUpdateProgramParameters(myCgFragmentProgram);
  glutWireSphere(2.0, 30, 30);

  /*** Render red wireframe cone ***/

  /* modelView = viewMatrix * translateMatrix */
  makeTranslateMatrix(-2, -1.5, 0, translateMatrix);
  makeRotateMatrix(90, 1, 0, 0, rotateMatrix);
  multMatrix(modelMatrix, translateMatrix, rotateMatrix);

  /* modelViewMatrix = viewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, viewMatrix, modelMatrix);

  /* modelViewProj = projectionMatrix * modelViewMatrix */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, modelViewMatrix);

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgSetParameter4f(myCgFragmentParam_c, 0.8,0.1,0.1,1);  /* Red */
  cgUpdateProgramParameters(myCgVertexProgram);
  cgUpdateProgramParameters(myCgFragmentProgram);
  glutWireCone(1.5, 3.5, 20, 20);

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  glutSwapBuffers();
}

static void idle(void)
{
  myEyeAngle += 0.008;  /* Add a small angle (in radians). */
  if (myEyeAngle > 2*myPi) {
    myEyeAngle -= 2*myPi;
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
