
/* 16_keyframe_interpolation.c - OpenGL-based keyframe interpolation example
   using Cg program from Chapter 6 of "The Cg Tutorial" (Addison-Wesley,
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

#include "loadtex.h"
#include "md2.h"
#include "md2render.h"

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgFragmentProfile;
static CGprogram   myCgVertexProgram[2],
                   myCgFragmentProgram[2];
static CGparameter myCgVertexParam_modelViewProj[2],
                   myCgVertexParam_keyFrameBlend[2],
                   myCgVertexParam_light_eyePosition,
                   myCgVertexParam_light_lightPosition,
                   myCgFragmentParam_decal;

const char *myProgramName = "16_keyframe_interpolation";

static const char *myVertexProgramFileName[2] = { "C6E3v_keyFrame.cg",
                                                  "C6E4v_litKeyFrame.cg" },
/* Page 159 */    *myVertexProgramName[2] = { "C6E3v_keyFrame",
/* Page 161-2 */                              "C6E4v_litKeyFrame" },
                  *myFragmentProgramFileName[2] = { "texmodulate.cg",
                                                    "colorinterp.cg" },
                  *myFragmentProgramName[2] = { "texmodulate",
                                                "colorinterp" };

static float myLightAngle = 0.78f;  /* Angle in radians light rotates around knight. */
static float myLightHeight = 12.0f; /* Vertical height of light. */
static float myEyeAngle = 0.53f;    /* Angle in radians eye rotates around knight. */

static float myProjectionMatrix[16];
static float mySpecularExponent = 8.0f;
static float myAmbient = 0.2f;
static float myLightColor[3] = { 1, 1, 1 };  /* White */

static int myVertexProgramIndex = 0,
           myFragmentProgramIndex = 0;

static Md2Model *myKnightModel;
static MD2render *myMD2render;
static float myFrameKnob = 0;

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
static void visibility(int state);
static void display(void);
static void keyboard(unsigned char c, int x, int y);
static void menu(int item);
static void mouse(int button, int state, int x, int y);
static void motion(int x, int y);
static void requestSynchronizedSwapBuffers(void);

int main(int argc, char **argv)
{
  CGparameter param;
  gliGenericImage *decalImage;
  int i;

  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);

  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutVisibilityFunc(visibility);
  glutKeyboardFunc(keyboard);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_5) {
    fprintf(stderr, "%s: failed to initialize GLEW, OpenGL 1.5 required.\n", myProgramName);    
    exit(1);
  }

  myKnightModel = md2ReadModel("knight.md2");
  if (0 == myKnightModel) {
    fprintf(stderr, "%s: count not load knight.md2\n", myProgramName);
    exit(1);
  }
  myMD2render = createMD2render(myKnightModel);
  decalImage = readImage("knight.tga");
  decalImage = loadTextureDecal(decalImage, 1);
  gliFree(decalImage);

  requestSynchronizedSwapBuffers();
  glClearColor(0.3, 0.3, 0.1, 0);  /* Gray background. */
  glEnable(GL_DEPTH_TEST);         /* Hidden surface removal. */
  glEnable(GL_CULL_FACE);          /* Backface culling. */
  glLineWidth(3.0f);

  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgGLSetDebugMode(CG_FALSE);

  /** Create vertex programs **/
  myCgVertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
  cgGLSetOptimalOptions(myCgVertexProfile);
  checkForCgError("selecting vertex profile");

  for (i=0; i<2; i++) {
    myCgVertexProgram[i] =
      cgCreateProgramFromFile(
        myCgContext,                 /* Cg runtime context */
        CG_SOURCE,                   /* Program in human-readable form */
        myVertexProgramFileName[i],  /* Name of file containing program */
        myCgVertexProfile,           /* Profile: OpenGL ARB vertex program */
        myVertexProgramName[i],      /* Entry function name */
        NULL);                       /* No extra compiler options */
    checkForCgError("creating vertex program from file");
    cgGLLoadProgram(myCgVertexProgram[i]);
    checkForCgError("loading vertex program");
  }

#define GET_VERTEX_PARAM_I(name,i) \
  myCgVertexParam_##name[i] = \
    cgGetNamedParameter(myCgVertexProgram[i], #name); \
  checkForCgError("could not get " #name " parameter");

  GET_VERTEX_PARAM_I(modelViewProj, 0);
  GET_VERTEX_PARAM_I(modelViewProj, 1);
  GET_VERTEX_PARAM_I(keyFrameBlend, 0);
  GET_VERTEX_PARAM_I(keyFrameBlend, 1);

  myCgVertexParam_light_eyePosition =
    cgGetNamedParameter(myCgVertexProgram[1], "light.eyePosition");
  checkForCgError("could not get light.eyePosition parameter");

  myCgVertexParam_light_lightPosition =
    cgGetNamedParameter(myCgVertexProgram[1], "light.lightPosition");
  checkForCgError("could not get light.lightPosition parameter");

  /* Set light source color parameters once. */
  param = cgGetNamedParameter(myCgVertexProgram[1], "light.lightColor");
  checkForCgError("could not get light.lightColor parameter");
  cgSetParameter4fv(param, myLightColor);

  param = cgGetNamedParameter(myCgVertexProgram[1], "light.specularExponent");
  checkForCgError("could not get light.specularExponent parameter");
  cgSetParameter1f(param, mySpecularExponent);

  param = cgGetNamedParameter(myCgVertexProgram[1], "light.ambient");
  checkForCgError("could not get light.ambient parameter");
  cgSetParameter1f(param, myAmbient);

  /** Creat fragment programs **/
  myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  cgGLSetOptimalOptions(myCgFragmentProfile);
  checkForCgError("selecting fragment profile");

  for (i=0; i<2; i++) {
    myCgFragmentProgram[i] =
      cgCreateProgramFromFile(
        myCgContext,               /* Cg runtime context */
        CG_SOURCE,                 /* Program in human-readable form */
        myFragmentProgramFileName[i],
        myCgFragmentProfile,       /* Profile: latest fragment profile */
        myFragmentProgramName[i],  /* Entry function name */
        NULL);                     /* No extra compiler options */
    checkForCgError("creating fragment program from string");
    cgGLLoadProgram(myCgFragmentProgram[i]);
    checkForCgError("loading fragment program");
  }

  myCgFragmentParam_decal =
    cgGetNamedParameter(myCgFragmentProgram[0], "decal");
  checkForCgError("could not get decal parameter");

  param = cgGetNamedParameter(myCgFragmentProgram[0], "scaleFactor");
  checkForCgError("could not get scaleFactor parameter");
  cgSetParameter2f(param,
    1.0f/myKnightModel->header.skinWidth, 1.0f/myKnightModel->header.skinHeight);

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAddMenuEntry("[w] Toggle wireframe", 'w');
  glutAddMenuEntry("[v] Toggle vertex program", 'f');
  glutAddMenuEntry("[f] Toggle fragment program", 'v');
  glutAddMenuEntry("[Enter] Toggle lighting/texture", 13);
  glutAddMenuEntry("[Esc] Quit", 27);
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
                         10.0, 200.0,  /* Znear and Zfar */
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

float addDelta(float frameKnob, float delta, int numFrames)
{
  frameKnob += delta;
  while (frameKnob >= numFrames) {
      frameKnob -= numFrames;
  }
  if (frameKnob < 0) {
    frameKnob = 0;  /* Just to be sure myFrameKnob is never negative. */
  }
  return frameKnob;
}

static void display(void)
{
  /* World-space positions for light and eye. */
  const float eyeRadius = 85,
              lightRadius = 40;
  const float eyePosition[3] = { cos(myEyeAngle)*eyeRadius, 0, sin(myEyeAngle)*eyeRadius };
  const float lightPosition[3] = { lightRadius*sin(myLightAngle), 
                                   myLightHeight,
                                   lightRadius*cos(myLightAngle) };

  const int frameA = floor(myFrameKnob),
            frameB = addDelta(myFrameKnob, 1, myKnightModel->header.numFrames);

  float viewMatrix[16], modelViewProjMatrix[16];

  buildLookAtMatrix(eyePosition[0], eyePosition[1], eyePosition[2],
                    0, 0, 0,
                    0, 1, 0,
                    viewMatrix);
  /* modelViewProj = projectionMatrix * viewMatrix (model is identity) */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, viewMatrix);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  cgGLBindProgram(myCgVertexProgram[myVertexProgramIndex]);
  checkForCgError("binding vertex program");

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  cgGLBindProgram(myCgFragmentProgram[myFragmentProgramIndex]);
  checkForCgError("binding fragment program");

  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj[myVertexProgramIndex], modelViewProjMatrix);
  cgSetParameter1f(myCgVertexParam_keyFrameBlend[myVertexProgramIndex], myFrameKnob-floor(myFrameKnob));
  /* Set eye and light positions if lighting. */
  if (myVertexProgramIndex == 1) {
    cgSetParameter3fv(myCgVertexParam_light_lightPosition, lightPosition);
    cgSetParameter3fv(myCgVertexParam_light_eyePosition, eyePosition);
  }

  drawMD2render(myMD2render, frameA, frameB);

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  /* If using lighting vertex program, render light position as yellow sphere. */
  if (myVertexProgramIndex == 1) {
    glPushMatrix();
      /* glLoadMatrixf expects a column-major matrix but Cg matrices are
         row-major (matching C/C++ arrays) so used glLoadTransposeMatrixf
         which OpenGL 1.3 introduced. */
      glLoadTransposeMatrixf(modelViewProjMatrix);
      glTranslatef(lightPosition[0], lightPosition[1], lightPosition[2]);
      glColor3f(1,1,0); /* yellow */
      glutSolidSphere(1, 10, 10);  /* sphere to represent light position */
      glColor3f(1,1,1); /* reset back to white */
    glPopMatrix();
  }

  glutSwapBuffers();
}

static int myLastElapsedTime;

static void idle(void)
{
  const float millisecondsPerSecond = 1000.0f;
  const float keyFramesPerSecond = 3.0f;
  int now = glutGet(GLUT_ELAPSED_TIME);
  float delta = (now - myLastElapsedTime) / millisecondsPerSecond;

  myLastElapsedTime = now;  /* This time become "prior time". */

  delta *= keyFramesPerSecond;
  myFrameKnob = addDelta(myFrameKnob, delta, myKnightModel->header.numFrames);
  glutPostRedisplay();
}

static int myAnimating = 1;

static void visibility(int state)
{
  if (state == GLUT_VISIBLE && myAnimating) {
    myLastElapsedTime = glutGet(GLUT_ELAPSED_TIME);
    glutIdleFunc(idle);
  } else {
    glutIdleFunc(NULL);
  }
}

static void keyboard(unsigned char c, int x, int y)
{
  static int wireframe = 0;

  switch (c) {
  case ' ':
    myAnimating = !myAnimating; /* Toggle */
    if (myAnimating) {
      myLastElapsedTime = glutGet(GLUT_ELAPSED_TIME);
      glutIdleFunc(idle);
    } else {
      glutIdleFunc(NULL);
    }    
    break;
  case 13:
    myVertexProgramIndex = 1-myVertexProgramIndex;
    myFragmentProgramIndex = myVertexProgramIndex;
    break;
  case 'f':
    myFragmentProgramIndex = 1-myFragmentProgramIndex;
    break;
  case 'v':
    myVertexProgramIndex = 1-myVertexProgramIndex;
    break;
  case 'w':
    wireframe = !wireframe;
    if (wireframe) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    break;
  case 27:  /* Esc key */
    cgDestroyContext(myCgContext);
    exit(0);
    break;
  default:
    return;
  }
  glutPostRedisplay();
}

static void menu(int item)
{
  /* Pass menu item character code to keyboard callback. */
  keyboard((unsigned char)item, 0, 0);
}

int beginx, beginy;
int moving = 0;
int movingLight = 0;
int xLightBegin, yLightBegin;

void
motion(int x, int y)
{
  if (moving) {
    myEyeAngle += 0.005*(x - beginx);
    beginx = x;
    beginy = y;
    glutPostRedisplay();
  }
  if (movingLight) {
    myLightAngle += 0.005*(x - xLightBegin);
    myLightHeight += 0.1*(yLightBegin - y);
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
