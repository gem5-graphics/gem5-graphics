
/* cg_boxfilter - a fast box filter implemented with Cg  */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <GL/glew.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed? */
#include <Cg/cgGL.h>  /* 3D API specific Cg runtime API for OpenGL */

static const char *myProgramName = "cg_boxfilter";  /* Program name for messages. */

#define MAX_TECHNIQUES 20
int numTechniques;
const char *techniqueName[MAX_TECHNIQUES];

/* Cg global variables */
CGcontext   myCgContext;
CGeffect    myCgEffect;
CGtechnique myCgTechnique, technique[MAX_TECHNIQUES];

CGparameter myCgSourceWidth,
            myCgSourceHeight,
            myCgDestWidth,
            myCgDestHeight,
            myCgWindowSize,
            myCgImageWindowOffset,
            myCgSourceImage;

int mode = 0;

int imageX = 40, imageY = 20;
int imageW = 600, imageH = 450;
//int imageW = 10, imageH = 10;

static const GLubyte
myDemonTextureImage[3*(128*128)] = {
/* RGB8 image data for 128x128 demon texture */
#include "demon_image.h"
};

extern GLubyte myVistaTextureImage[3*(1024*1024)];

static void reshape(int width, int height);
static void display(void);
static void keyboard(unsigned char c, int x, int y);
static void mouse(int button, int state, int x, int y);
static void motion(int x, int y);
static void initCg();
static void initOpenGL();

int main(int argc, char **argv)
{
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(900, 600);
  glutInit(&argc, argv);
  glutCreateWindow(myProgramName);
  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_1) {
    fprintf(stderr, "%s: Failed to initialize GLEW. OpenGL 1.1 required.\n", myProgramName);    
    exit(1);
  }

  initCg();
  initOpenGL();

  glutMainLoop();
  return 0;
}

static void checkForCgError(const char *situation)
{
  char buffer[4096];
  CGerror error;
  const char *string = cgGetLastErrorString(&error);
  
  if (error != CG_NO_ERROR) {
    if (error == CG_COMPILER_ERROR) {
      sprintf(buffer,
              "Program: %s\n"
              "Situation: %s\n"
              "Error: %s\n\n"
              "Cg compiler output...\n",
              myProgramName, situation, string);
#ifdef _WIN32
      OutputDebugStringA(buffer);
      OutputDebugStringA(cgGetLastListing(myCgContext));
      sprintf(buffer,
              "Program: %s\n"
              "Situation: %s\n"
              "Error: %s\n\n"
              "Check debug output for Cg compiler output...",
              myProgramName, situation, string);
      MessageBoxA(0, buffer,
                  "Cg compilation error", MB_OK | MB_ICONSTOP | MB_TASKMODAL);
#else
      printf("%s", buffer);
      printf("%s\n", cgGetLastListing(myCgContext));
#endif
    } else {
      sprintf(buffer,
              "Program: %s\n"
              "Situation: %s\n"
              "Error: %s",
              myProgramName, situation, string);
#ifdef _WIN32
      MessageBoxA(0, buffer,
                  "Cg runtime error", MB_OK | MB_ICONSTOP | MB_TASKMODAL);
#else
      printf("%s\n", buffer);
#endif
    }
    exit(1);
  }
}

int setTechnique(void)
{
  CGbool valid;

  printf("%s: try to validate \"%s\" technique\n",
    myProgramName, techniqueName[mode]);
  
  valid = cgValidateTechnique(technique[mode]);

  if (valid) {
    glutSetWindowTitle(techniqueName[mode]);
    myCgTechnique = technique[mode];
  } else {
    /* Clear error cgValidateTechnique might have set. */
    cgGetError();
    printf("%s: could not validate \"%s\" technique\n",
      myProgramName, techniqueName[mode]);
  }
  return valid;
}

int sourceW, sourceH;

static void initCg(void)
{
  CGtechnique t;

  myCgContext = cgCreateContext();
  checkForCgError("establishing Cg context");

  cgGLRegisterStates(myCgContext);
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);

  myCgEffect = cgCreateEffectFromFile(myCgContext, "boxfilter.cgfx", NULL);
  if (!myCgEffect) {
      printf("%s\n", cgGetLastListing(myCgContext));
  }
  checkForCgError("creating boxfilter.cgfx effect");

  numTechniques = 0;
  t = cgGetFirstTechnique(myCgEffect);
  while (t) {
    technique[numTechniques] = t;
    techniqueName[numTechniques] = cgGetTechniqueName(t);
    printf("%s: technique \"%s\"\n", myProgramName, techniqueName[numTechniques]);
    numTechniques++;
    t = cgGetNextTechnique(t);
  }

  /* Find a valid technique. */
  while (!setTechnique() && ++mode < numTechniques);
  if (mode >= numTechniques) {
    printf("%s: no valid techniques found!\n", myProgramName);
    exit(0);
  }

  myCgSourceWidth       = cgGetNamedEffectParameter(myCgEffect, "sourceWidth");
  myCgSourceHeight      = cgGetNamedEffectParameter(myCgEffect, "sourceHeight");
  myCgDestWidth         = cgGetNamedEffectParameter(myCgEffect, "destWidth");
  myCgDestHeight        = cgGetNamedEffectParameter(myCgEffect, "destHeight");
  myCgWindowSize        = cgGetNamedEffectParameter(myCgEffect, "windowSize");
  myCgImageWindowOffset = cgGetNamedEffectParameter(myCgEffect, "imageWindowOffset");
  myCgSourceImage       = cgGetNamedEffectParameter(myCgEffect, "sourceImage");

  checkForCgError("get parameters");
}

/* OpenGL texture object (TO) handles. */
enum {
  TO_SOURCE_IMAGE = 1,
};

static void initOpenGL(void)
{
  glClearColor(0.1, 0.3, 0.6, 0.0);  /* Blue background */
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); /* Tightly packed texture data. */

  glBindTexture(GL_TEXTURE_2D, TO_SOURCE_IMAGE);
  sourceW = 128;
  sourceH = 128;
  glTexImage2D(GL_TEXTURE_2D, 0,
    GL_RGB8, 128, 128, 0, GL_RGB, GL_UNSIGNED_BYTE, myDemonTextureImage);

  cgSetParameter1f(myCgSourceWidth, sourceW);
  cgSetParameter1f(myCgSourceHeight, sourceH);
  cgGLSetTextureParameter(myCgSourceImage, TO_SOURCE_IMAGE);
  cgSetSamplerState(myCgSourceImage);

  checkForCgError("setting source image texture");

  cgSetParameter1f(myCgSourceWidth, sourceW);
  cgSetParameter1f(myCgSourceHeight, sourceH);
  cgSetParameter1f(myCgDestWidth, imageW);
  cgSetParameter1f(myCgDestHeight, imageH);
}

int myWindowWidth, myWindowHeight;

static void reshape(int width, int height)
{
  myWindowWidth = width;
  myWindowHeight = height;

  glViewport(0, 0, width, height);

  cgSetParameter2f(myCgWindowSize, myWindowWidth, myWindowHeight);
}

static int draw(int count)
{
  CGpass pass;
  int i;

  cgSetParameter1f(myCgDestWidth, imageW);
  cgSetParameter1f(myCgDestHeight, imageH);
  cgSetParameter2f(myCgImageWindowOffset, imageX, imageY);

  /* Iterate through rendering passes for technique (even
     though bumpdemo.cgfx has just one pass). */
  pass = cgGetFirstPass(myCgTechnique);
  while (pass) {
    cgSetPassState(pass);

    for (i=0; i<count; i++) {
      glRectf(0,0,1,1);
    }

    cgResetPassState(pass);
    pass = cgGetNextPass(pass);
  }
  return count;
}

static void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  draw(1);

  glutSwapBuffers();
}

int sizing = 0;
int downX, downY;

static void mouse(int button, int state, int x, int y)
{
  if (button == GLUT_LEFT_BUTTON) {
    if (state == GLUT_DOWN) {
      sizing = 1;
      downX = x;
      downY = y;
      imageW = 1;
      imageH = 1;
      imageX = x;
      imageY = myWindowHeight - y - imageH;
      glutPostRedisplay();
    } else {
      sizing = 0;
    }
  }
}

static void motion(int x, int y)
{
  if (sizing) {
    imageW = x - downX;
    imageH = y - downY;
    imageY = myWindowHeight - downY - imageH;
    glutPostRedisplay();
  }
}

int loops = 2;

#define MIN_MEASUREMENT_LOOPS 5
#define MAX_MEASUREMENT_LOOPS 1000000

// in milliseconds
#define MIN_CALIBRATION_TIME 100
#define MIN_MEASUREMENT_TIME 2000

#define MEASUREMENT_START                                           \
    /* loops==1 means: redraw only once */                          \
    doLoops = loops==1 ? 1 : MIN_MEASUREMENT_LOOPS;                 \
    do {                                                            \
      start = glutGet(GLUT_ELAPSED_TIME);                           \
                                                                    \
      for (i=0; i<doLoops; i++) {

#define MEASUREMENT_END                                             \
      }                                                             \
                                                                    \
      glFinish();                                                   \
      glReadPixels(0, 0, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, pixel);    \
      end = glutGet(GLUT_ELAPSED_TIME);                             \
      /* loops==1 means: redraw only once */                        \
    } while ((loops > 1) && !IsMeasurementDone(start, end, &doLoops));

#define MEASURE(INIT, TASK)                                         \
    /* loops==1 means: redraw only once */                          \
    doLoops = loops==1 ? 1 : MIN_MEASUREMENT_LOOPS;                 \
    do {                                                            \
{ INIT }                                                            \
      start = glutGet(GLUT_ELAPSED_TIME);                           \
                                                                    \
      for (i=0; i<doLoops; i++) {                                   \
{ TASK }                                                            \
      }                                                             \
                                                                    \
      glFinish();                                                   \
      glReadPixels(0, 0, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, pixel);    \
      end = glutGet(GLUT_ELAPSED_TIME);                             \
      /* loops==1 means: redraw only once */                        \
    } while ((loops > 1) && !IsMeasurementDone(start, end, &doLoops)); 

// note: this function must not alter *pDoLoops when it returns 1,
//       because doLoops is used in perf calculation later on
static int IsMeasurementDone(int start, int end, int *pDoLoops)
{
  if (end - start >= MIN_MEASUREMENT_TIME) {
    return 1; // done
  }
  // Dlist runs in trivial reject cases vary pretty much in perf!
  // Therefore use an upper limit do prevent runs which nearly hang
  // even if measurement time is 0
  if (*pDoLoops >= MAX_MEASUREMENT_LOOPS) {
    return 1; // done
  }
  if (end - start < MIN_CALIBRATION_TIME) {
    // repeat calibration
    *pDoLoops *= 3;
  } else {
    // project for 110% of MIN_MEASUREMENT_TIME to account for varying runs
    *pDoLoops = (*pDoLoops * MIN_MEASUREMENT_TIME * 11 / (10 * (end - start)));
  }
  // limit number of loopthrus
  if (*pDoLoops > MAX_MEASUREMENT_LOOPS) {
    *pDoLoops = MAX_MEASUREMENT_LOOPS;
  }
  return 0;
}

void
benchmark(void)
{
  int start, end;
  double secs;
  double framesPerSec;
  double rescalesPerSec;
  double sourcePixelsPerSec;
  double rescales;
  double frames;
  int doLoops;
  GLubyte pixel[3];
  int i;
  double rescaledPixelsPerSec;

  printf("Benchmarking %dx%d-to-%dx%d %s rescaling rate...\n", sourceW, sourceH, imageW, imageH, techniqueName[mode]);

  MEASURE(rescales = 0;,
          rescales += draw(4););

  secs = (end-start) / 1000.0;
  frames = doLoops;
  framesPerSec = frames / secs;
  rescalesPerSec = rescales / secs;
  rescaledPixelsPerSec = rescales * imageW * imageH / secs / 1000000.0;
  sourcePixelsPerSec = rescales * sourceW * sourceH / secs / 1000000.0;

  printf("  %8.2f sec benchmark run\n", secs);
  //printf("  %8.2f frames/sec\n", framesPerSec);
  printf("  %8.2f rescales/sec\n", rescalesPerSec);
  printf("  %8.2f mega rescaled pixels/sec\n", rescaledPixelsPerSec);
  printf("  %8.2f mega sourced pixels/sec\n", sourcePixelsPerSec);
  //printf("  %8.2f useful rescales/frame\n", rescales/frames);
}

static void keyboard(unsigned char c, int x, int y)
{
  switch (c) {
  case 27:  /* Esc key */
    cgDestroyEffect(myCgEffect);
    cgDestroyContext(myCgContext);
    exit(0);
    break;
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    if (c - '1' < numTechniques) {
      mode = c - '1';
      setTechnique();
    } else {
      printf("%s: only %d techniques\n", myProgramName, numTechniques);
    }
    break;
  case 'j':
    imageY += 1;
    break;
  case 'k':
    imageY -= 1;
    break;
  case 'h':
    imageX -= 1;
    break;
  case 'l':
    imageX += 1;
    break;
  case 'a':
    imageW += 1;
    imageH += 1;
    break;
  case 'z':
    imageW -= 1;
    imageH -= 1;
    break;
  case 'b':
    benchmark();
    break;
  case ' ':
    mode = (mode+1) % numTechniques;
    setTechnique();
    break;
  }
  glutPostRedisplay();
}
