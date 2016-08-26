
/* gs_interp_quad.c - a CgFX 2.0 demo of a quadrilateral barycentric
   interpolation scheme using mean value coordinates for better
   attribute interpolation over quadrilateral and proper rasterization
   "bow-tie" quadrilaterals. */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>   /* for exit */

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

#ifdef _WIN32
#include <windows.h>  /* for QueryPerformanceCounter */
#endif

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed? */
#include <Cg/cgGL.h>  /* 3D API specific Cg runtime API for OpenGL */

#include "request_vsync.h"

const char *programName = "gs_interp_quad"; /* Program name for messages. */

/* Cg global variables */
CGcontext   myCgContext;
CGeffect    myCgEffect;

static int enableSync = 1;  /* Sync buffer swaps to monitor refresh rate. */

static void handleFPS(void);
static void display(void);
static void reshape(int width, int height);
static void keyboard(unsigned char c, int x, int y);
static void initCg();
static void initMenu();
static void initOpenGL();

int main(int argc, char **argv)
{
  int i;

  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(600, 300);
  glutInit(&argc, argv);

  for (i=1; i<argc; i++) {
    if (!strcmp("-nosync", argv[i])) {
      enableSync = 0;
    }
  }

  glutCreateWindow(programName);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_1) {
    fprintf(stderr, "%s: Failed to initialize GLEW. OpenGL 1.1 required.\n", programName);    
    exit(1);
  }

  requestSynchronizedSwapBuffers(enableSync);
  initCg();
  initMenu();
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
  CGtechnique technique;
  int problems = 0;

  myCgContext = cgCreateContext();
  cgGLSetDebugMode( CG_FALSE );
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);
  cgGLRegisterStates(myCgContext);
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
  checkForCgError(cgGetError(), "establishing Cg context");

  myCgEffect = cgCreateEffectFromFile(myCgContext, "gs_interp_quad.cgfx", NULL);
  if (!myCgEffect) {
      printf("%s\n", cgGetLastListing(myCgContext));
  }
  checkForCgError(cgGetError(), "creating gs_interp_quad.cgfx effect");

  technique = cgGetFirstTechnique(myCgEffect);
  if (technique && cgValidateTechnique(technique) == CG_FALSE) {
    fprintf(stderr, "%s: Technique %s did not validate.\n",
      programName, cgGetTechniqueName(technique));
    problems++;
  }
  technique = cgGetNextTechnique(technique);
  if (technique && cgValidateTechnique(technique) == CG_FALSE) {
    fprintf(stderr, "%s: Technique %s did not validate.\n",
      programName, cgGetTechniqueName(technique));
    problems++;
  }
  if (problems) {
    fprintf(stderr, "%s: is your OpenGL implementation incapable of the geometry profile (gp4gp)?\n",
      programName);
    exit(0);
  }
}

static void menu(int item)
{
  /* Pass menu selections to keyboard routine. */
  keyboard((unsigned char)item, 0, 0);
}

void initMenu(void)
{
  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Toggle animation", ' ');
  glutAddMenuEntry("[f] Toggle frame rate", 'f');
  glutAddMenuEntry("[c] Toggle conventional quad rendering", 'c');
  glutAddMenuEntry("[v] Toggle view rotation", 'v');
  glutAddMenuEntry("[o] Toggle ortho/perspective view", 'o');
  glutAddMenuEntry("[r] Reset", 'r');
  glutAddMenuEntry("Quit", 27);
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}

static void initOpenGL(void)
{
  glClearColor(0.1, 0.3, 0.6, 0.0);  /* Blue background */
  glEnable(GL_DEPTH_TEST);
}

GLsizei viewWidth, viewHeight;
GLsizei windowWidth, windowHeight;

static void reshape(int width, int height)
{
  windowWidth = width;
  windowHeight = height;
  viewWidth = width/2;
  viewHeight = height;
}

/* Animation parameters. */
float spinAngle = 0.0;
float viewAngle = 0;
float viewRotateRate = 0;

static void perespectiveView(void)
{
  float viewWidthf, viewHeightf;
  float aspectRatio;
  const float eyeRadius = 1.8;
  float eyePosition[3];

  viewWidthf = viewWidth/2.0f;
  viewHeightf = viewHeight/2.0f;

  aspectRatio = viewWidthf / viewHeightf;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(
    60.0,        /* Field of view in degree */
    aspectRatio, /* Aspect ratio */ 
    0.1,         /* Z near */
    100.0);      /* Z far */

  eyePosition[0] = eyeRadius * sin(viewAngle);
  eyePosition[1] = 0;
  eyePosition[2] = eyeRadius * cos(viewAngle);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(
    eyePosition[0], eyePosition[1], eyePosition[2], 
    0.0 ,0.0,  0.0,   /* XYZ view center */
    0.0, 1.0,  0.0);  /* Up is in positive Y direction */
}

static const float pi  = 3.14159265358979323846,
                   pi2 = 2 * 3.14159265358979323846;

/* Draw the vertices of a single quadrilateral.  Twist the right 
   two vertices based on spinAngle. */
static void quadVertices(void)
{
  glColor3f(1,0,0);  /* red */
  glVertex3f(-0.8, -0.8, 0);
  glColor3f(0,1,0);  /* green */
  glVertex3f( 0.8, 0.8*cos(spinAngle + pi), 0.8*sin(spinAngle + pi));
  glColor3f(1,0,0);  /* red */
  glVertex3f( 0.8,  0.8*cos(spinAngle), 0.8*sin(spinAngle));
  glColor3f(0,0,1);  /* blue */
  glVertex3f(-0.8,  0.8, 0);
}

/* Draw a flat 2D patch that can be "rolled & bent" into a 3D torus by
   a vertex program. */
static void drawQuad(void)
{
  glBegin(GL_LINES_ADJACENCY_EXT);
  quadVertices();
  glEnd();
}

static void drawShadedQuad(CGtechnique technique)
{
  CGpass pass = cgGetFirstPass(technique);

  /* Iterate through rendering passes for technique (even
     though gs_interp_quad.cgfx has just one pass). */
  while (pass) {
    cgSetPassState(pass);
    drawQuad();
    cgResetPassState(pass);
    pass = cgGetNextPass(pass);
  }
}

static void drawLabel(float x, float y, const char *label)
{
  GLubyte dummy;
  const char *c;

  glRasterPos2f(0,0);
  glBitmap(0, 0, 0, 0, x, -y, &dummy);
  for (c = label; *c != '\0'; c++) {
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);
  }
}

static void drawLabels(const char *label1, const char *label2)
{
  glViewport(0,0, windowWidth, windowHeight);
  glPushMatrix();
  {
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    {
      glLoadIdentity();
      glOrtho(0, 1, 1, 0, -1, 1);
      glDisable(GL_DEPTH_TEST);

      glColor3f(1,1,1);
      drawLabel(20, 20, label1);
      drawLabel(20+viewWidth, 20, label2);

      /* Draw vertical yellow line to divide the two views. */
      glColor3f(1,1,0);
      glBegin(GL_LINES);
      glVertex2f(0.5, 0);
      glVertex2f(0.5, 1);
      glEnd();

      glEnable(GL_DEPTH_TEST);
    }
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
  }
  glPopMatrix();
}

/* Fixed-function GL_QUADS rendering if true; otherwise use Cg technique. */
int conventionalFixedFunction = 0;
/* Orthographic view if true, otherwise perspective. */
int orthoMode = 0;

static void display(void)
{
  CGtechnique technique1 = cgGetFirstTechnique(myCgEffect),
              technique2 = cgGetNextTechnique(technique1);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (orthoMode) {
    /* Configure orthographic view. */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef(viewAngle * 360/pi2, 0, 1, 0);
  } else {
    /* Configure a perspective view. */
    perespectiveView();
  }

  glViewport(0, 0, viewWidth, viewHeight);
  if (conventionalFixedFunction) {
    glBegin(GL_QUADS);
    quadVertices();
    glEnd();
  } else {
    drawShadedQuad(technique1);
  }

  glViewport(viewWidth, 0, windowWidth-viewWidth, viewHeight);
  drawShadedQuad(technique2);

  drawLabels(conventionalFixedFunction ? "Fixed-function GL_QUADS" :
                                         cgGetTechniqueName(technique1),
             cgGetTechniqueName(technique2));

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
  spinAngle += 0.1;
  if (spinAngle > pi2) {
    spinAngle -= pi2;
  }
  viewAngle += viewRotateRate;
  if (viewAngle > pi2) {
    viewAngle -= pi2;
  }
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
  case 'v':
    if (viewRotateRate) {
      viewRotateRate = 0;
    } else {
      viewRotateRate = 0.04;
    }
    break;
  case 'c':
    conventionalFixedFunction = !conventionalFixedFunction;
    glutPostRedisplay();
    break;
  case 'o':
    orthoMode = !orthoMode;
    glutPostRedisplay();
    break;
  case 'r':
    spinAngle = 0;
    viewAngle = 0;
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
