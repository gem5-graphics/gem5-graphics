
/* gs_quadnormal.c - a CgFX 2.0 demo */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>   /* for exit */
#ifdef _WIN32
#include <windows.h>  /* for QueryPerformanceCounter */
#endif
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <Cg/cg.h>    /* core Cg runtime API */
#include <Cg/cgGL.h>  /* 3D API specific Cg runtime API for OpenGL */

#include "fast_teapot.h"
#include "request_vsync.h"

const char *programName = "gs_quadnormal"; /* Program name for messages. */

/* Cg global variables */
CGcontext   myCgContext;
CGeffect    myCgEffect;
CGtechnique myCgTechnique;
CGparameter myCgEyePositionParam,
            myCgLightPositionParam,
            myCgModelViewProjParam,
            myCgModelViewParam,
            myCgInverseModelViewParam;

static int enableSync = 1;  /* Sync buffer swaps to monitor refresh rate. */

static void handleFPS(void);
static void display(void);
static void reshape(int width, int height);
static void keyboard(unsigned char c, int x, int y);
static void initCg();
static void initTechniqueMenu();
static void initOpenGL();

int main(int argc, char **argv)
{
  int i;

  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(640, 480);
  glutInit(&argc, argv);

  for (i=1; i<argc; i++) {
    if (!strcmp("-nosync", argv[i])) {
      enableSync = 0;
    }
  }

  glutCreateWindow(programName);

  requestSynchronizedSwapBuffers(enableSync);
  initCg();
  initTechniqueMenu();
  initOpenGL();

  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMainLoop();
  return 0;
}

static void checkForCgError(CGerror error, const char *situation)
{
  if (error != CG_NO_ERROR) {
    printf("%s: %s: %s\n",
      programName, situation, cgGetErrorString(error));
#ifdef _WIN32
    MessageBox(0, "Cg compile error, see console window", programName, 0);
#endif
    exit(1);
  }
}

static void initCg(void)
{
  myCgContext = cgCreateContext();
  cgGLSetDebugMode( CG_FALSE );
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);
  cgGLRegisterStates(myCgContext);
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
  checkForCgError(cgGetError(), "establishing Cg context");

  myCgEffect = cgCreateEffectFromFile(myCgContext, "gs_quadnormal.cgfx", NULL);
  if (!myCgEffect) {
      printf("%s\n", cgGetLastListing(myCgContext));
  }
  checkForCgError(cgGetError(), "creating gs_quadnormal.cgfx effect");

  myCgTechnique = cgGetFirstTechnique(myCgEffect);
  while (myCgTechnique && cgValidateTechnique(myCgTechnique) == CG_FALSE) {
    fprintf(stderr, "%s: Technique %s did not validate.  Skipping.\n",
      programName, cgGetTechniqueName(myCgTechnique));
    myCgTechnique = cgGetNextTechnique(myCgTechnique);
  }
  if (myCgTechnique) {
    fprintf(stderr, "%s: Use technique %s.\n",
      programName, cgGetTechniqueName(myCgTechnique));
  } else {
    fprintf(stderr, "%s: No valid technique\n",
      programName);
    exit(1);
  }

  myCgModelViewProjParam =
    cgGetEffectParameterBySemantic(myCgEffect, "ModelViewProjection");
  if (!myCgModelViewProjParam) {
    fprintf(stderr,
      "%s: must find parameter with ModelViewProjection semantic\n",
      programName);
    exit(1);
  }
  myCgModelViewParam =
    cgGetEffectParameterBySemantic(myCgEffect, "ModelView");
  if (!myCgModelViewParam) {
    fprintf(stderr,
      "%s: must find parameter with ModelView semantic\n",
      programName);
    exit(1);
  }
  myCgInverseModelViewParam =
    cgGetEffectParameterBySemantic(myCgEffect, "InverseModelView");
  if (!myCgInverseModelViewParam) {
    fprintf(stderr,
      "%s: must find parameter with InverseModelView semantic\n",
      programName);
    exit(1);
  }
  myCgEyePositionParam =
    cgGetNamedEffectParameter(myCgEffect, "EyePosition");
  if (!myCgEyePositionParam) {
    fprintf(stderr, "%s: must find parameter named EyePosition\n",
      programName);
    exit(1);
  }
  myCgLightPositionParam =
    cgGetNamedEffectParameter(myCgEffect, "LightPosition");
  if (!myCgLightPositionParam) {
    fprintf(stderr, "%s: must find parameter named LightPosition\n",
      programName);
    exit(1);
  }
}

CGtechnique validTechnique[20];
#define MAX_TECHNIQUES sizeof(validTechnique)/sizeof(validTechnique[0])
int numTechniques = 0;

void selectTechnique(int item)
{
  if (item < numTechniques) {
    myCgTechnique = validTechnique[item];
  } else {
    // Try the item as a keyboard callback
    keyboard((unsigned char)item, 0, 0);
  }
  glutPostRedisplay();
}

void initTechniqueMenu(void)
{
  CGtechnique technique;
  int entry = 0;

  glutCreateMenu(selectTechnique);
  technique = cgGetFirstTechnique(myCgEffect);
  while (technique && entry < MAX_TECHNIQUES) {
    if (cgValidateTechnique(technique)) {
      validTechnique[entry] = technique;
      glutAddMenuEntry(cgGetTechniqueName(technique), entry);
      entry++;
    } else {
      printf("%s: could not validate technique %s\n",
        programName, cgGetTechniqueName(technique));
    }
    technique = cgGetNextTechnique(technique);
  }
  numTechniques = entry;
  glutAddMenuEntry("[t] Toggle fine vs. coarse tessellation", 't');
  glutAddMenuEntry("[f] Toggle frame rate", 'f');
  glutAddMenuEntry("[ ] Toggle animation", ' ');
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}

static void initOpenGL(void)
{
  glClearColor(0.1, 0.3, 0.6, 0.0);  /* Blue background */
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  /* Evaluator enables for fast teapots */
  glEnable(GL_MAP2_VERTEX_3);
  glEnable(GL_AUTO_NORMAL);
}

static void reshape(int width, int height)
{
  float aspectRatio = (float) width / (float) height;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(
    60.0,        /* Field of view in degree */
    aspectRatio, /* Aspect ratio */ 
    0.1,         /* Z near */
    100.0);      /* Z far */
  glMatrixMode(GL_MODELVIEW);

  glViewport(0, 0, width, height);
}

/* Draw a flat 2D patch that can be "rolled & bent" into a 3D torus by
   a vertex program. */
static void drawFlatPatch(float rows, float columns)
{
  float m = 1.0f/columns,
        n = 1.0f/rows;
  int i, j;

  for (i=0; i<columns; i++) {
    glBegin(GL_QUAD_STRIP);
    for (j=0; j<=rows; j++) {
      glVertex2f(i*m, j*n);
      glVertex2f((i+1)*m, j*n);
    }
    glVertex2f(i*m, 0);
    glVertex2f((i+1)*m, 0);
    glEnd();
  }
}

/* Initial scene state */
static float myEyeAngle = 0;
static const float myLightPosition[3] = { -8, 0, 15 };
static int myTeapotTessFactor = 2;
static int mySides = 8, myRings = 14;

static void display(void)
{
  const float eyeRadius = 18.0,
              eyeElevationRange = 8.0;
  const float teapotScale = -2.9;
  float eyePosition[3];
  CGpass pass;
  int drawTorus = 0;
  const char *n = cgGetTechniqueName(myCgTechnique);

  if (strstr(n, "Torus")) {
    drawTorus = 1;
  }

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  eyePosition[0] = eyeRadius * sin(myEyeAngle);
  eyePosition[1] = eyeElevationRange * sin(myEyeAngle);
  eyePosition[2] = eyeRadius * cos(myEyeAngle);

  glLoadIdentity();
  gluLookAt(
    eyePosition[0], eyePosition[1], eyePosition[2], 
    0.0 ,0.0,  0.0,   /* XYZ view center */
    0.0, 1.0,  0.0);  /* Up is in positive Y direction */

  if (drawTorus) {
    // No setup required for torus
  } else {
    /* Teapot */
    glRotatef(90.0, 1.0, 1.0, 0.0);
    glScalef(teapotScale, teapotScale, teapotScale);
    glTranslatef(0.0, 0.0, -1.5);
  }

  /* Set Cg parameters for the technique's effect. */
  cgGLSetStateMatrixParameter(myCgModelViewProjParam,
    CG_GL_MODELVIEW_PROJECTION_MATRIX,
    CG_GL_MATRIX_IDENTITY);
  cgGLSetStateMatrixParameter(myCgModelViewParam,
    CG_GL_MODELVIEW_MATRIX,
    CG_GL_MATRIX_IDENTITY);
  cgGLSetStateMatrixParameter(myCgInverseModelViewParam,
    CG_GL_MODELVIEW_MATRIX,
    CG_GL_MATRIX_INVERSE);
  cgSetParameter3fv(myCgEyePositionParam, eyePosition);
  cgSetParameter3fv(myCgLightPositionParam, myLightPosition);

  /* Iterate through rendering passes for technique (even
     though bumpdemo.cgfx has just one pass). */
  pass = cgGetFirstPass(myCgTechnique);
  while (pass) {
    cgSetPassState(pass);
    if (drawTorus) {
      drawFlatPatch(mySides, myRings);
    } else {
      fastTeapot(myTeapotTessFactor);
    }
    cgResetPassState(pass);
    pass = cgGetNextPass(pass);
  }

  handleFPS();
  glutSwapBuffers();
}

static int myDrawFPS = 1;

static void drawFPS(double fpsRate)
{
  GLubyte dummy;
  char buffer[200], *c;

  glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
      glLoadIdentity();
      glOrtho(0, 1, 1, 0, -1, 1);
      glDisable(GL_DEPTH_TEST);
      glColor3f(1,1,1);
      glRasterPos2f(1,1);
      glBitmap(0, 0, 0, 0, -10*9, 15, &dummy);
      sprintf(buffer, "fps %0.1f", fpsRate);
      for (c = buffer; *c != '\0'; c++)
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);
      glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

#ifndef _WIN32
#include <sys/time.h> /* for gettimeofday and struct timeval */
#endif

void
handleFPS(void)
{
  static int frameCount = 0;     /* Number of frames for timing */
  static double lastFpsRate = 0;
#ifdef _WIN32
  /* Use Win32 performance counter for high-accuracy timing. */
  static __int64 freq = 0;
  static __int64 lastCount = 0;  /* Timer count for last fps update */
  __int64 newCount;

  if (!freq) {
    QueryPerformanceFrequency((LARGE_INTEGER*) &freq);
  }

  /* Update the frames per second count if we have gone past at least
     a second since the last update. */

  QueryPerformanceCounter((LARGE_INTEGER*) &newCount);
  frameCount++;
  if (((newCount - lastCount) > freq) && drawFPS) {
    double fpsRate;

    fpsRate = (double) (freq * (__int64) frameCount)  / (double) (newCount - lastCount);
    lastCount = newCount;
    frameCount = 0;
    lastFpsRate = fpsRate;
  }
#else
  /* Use BSD 4.2 gettimeofday system call for high-accuracy timing. */
  static struct timeval last_tp = { 0, 0 };
  struct timeval new_tp;
  double secs;
  
  gettimeofday(&new_tp, NULL);
  secs = (new_tp.tv_sec - last_tp.tv_sec) + (new_tp.tv_usec - last_tp.tv_usec)/1000000.0;
  if (secs >= 1.0) {
    lastFpsRate = frameCount / secs;
    last_tp = new_tp;
    frameCount = 0;
  }
  frameCount++;
#endif
  if (myDrawFPS) {
    drawFPS(lastFpsRate);
  }
}

static void advanceAnimation(void)
{
  myEyeAngle += 0.05f;
  if (myEyeAngle > 2*3.14159)
    myEyeAngle -= 2*3.14159;
  glutPostRedisplay();
}

static void keyboard(unsigned char c, int x, int y)
{
  static int animating = 0;

  switch (c) {
  case ' ':
    animating = !animating; /* Toggle */
    glutIdleFunc(animating ? advanceAnimation : NULL);
    break;
  case 't':
    myTeapotTessFactor = (7 - myTeapotTessFactor);
    mySides = 24 - mySides;
    myRings = 40 - myRings;
    glutPostRedisplay();
    break;
  case 'f':
    myDrawFPS = !myDrawFPS;
    glutPostRedisplay();
    break;
  case 27:  /* Esc key */
    /* Demonstrate proper deallocation of Cg runtime data structures.
       Not strictly necessary if we are simply going to exit. */
    cgDestroyEffect(myCgEffect);
    cgDestroyContext(myCgContext);
    exit(0);
    break;
  }
}
