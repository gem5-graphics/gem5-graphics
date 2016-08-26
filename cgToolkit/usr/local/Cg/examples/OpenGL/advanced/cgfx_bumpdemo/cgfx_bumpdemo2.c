
/* cgfx_bumpdemo2.c - a CgFX 1.4 demo */

#include <stdio.h>
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

const char *programName = "cgfx_bumpdemo2"; /* Program name for messages. */

/* Cg global variables */
CGcontext   myCgContext;
CGeffect    myCgEffect;
CGtechnique myCgTechnique;
CGparameter myCgNormalMapParam,
            myCgNormalizeCubeParam,
            myCgEyePositionParam,
            myCgLightPositionParam,
            myCgModelViewProjParam;

static void handleFPS(void);
static void display(void);
static void reshape(int width, int height);
static void keyboard(unsigned char c, int x, int y);
static void initCg();
static void initTechniqueMenu();
static void initOpenGL();

int main(int argc, char **argv)
{
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(640, 480);
  glutInit(&argc, argv);
  glutCreateWindow(programName);

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

  myCgEffect = cgCreateEffectFromFile(myCgContext, "bumpdemo2.cgfx", NULL);
  if (!myCgEffect) {
      printf("%s\n", cgGetLastListing(myCgContext));
  }
  checkForCgError(cgGetError(), "creating bumpdemo2.cgfx effect");

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

void selectTechnique(int item)
{
  CGtechnique newTechnique = validTechnique[item];

  if (cgValidateTechnique(newTechnique)) {
    myCgTechnique = newTechnique;
    glutPostRedisplay();
  } else {
    fprintf(stderr, "%s: Technique %s did not validate.  Skipping.\n",
      programName, cgGetTechniqueName(newTechnique));
  } 
}

void initTechniqueMenu(void)
{
  CGtechnique technique;
  int entry = 0;

  glutCreateMenu(selectTechnique);
  technique = cgGetFirstTechnique(myCgEffect);
  while (technique && entry < MAX_TECHNIQUES) {
    validTechnique[entry] = technique;
    glutAddMenuEntry(cgGetTechniqueName(technique), entry);
    entry++;
    technique = cgGetNextTechnique(technique);
  }
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}

/* OpenGL texture object (TO) handles. */
enum {
  TO_NORMALIZE_VECTOR_CUBE_MAP = 1,
  TO_NORMAL_MAP = 2,
};

static const GLubyte
myBrickNormalMapImage[3*(128*128+64*64+32*32+16*16+8*8+4*4+2*2+1*1)] = {
/* RGB8 image data for a mipmapped 128x128 normal map for a brick pattern */
#include "brick_image.h"
};

static const GLubyte
myNormalizeVectorCubeMapImage[6*3*32*32] = {
/* RGB8 image data for a normalization vector cube map with 32x32 faces */
#include "normcm_image.h"
};

static CGparameter useSamplerParameter(CGeffect effect,
                                       const char *paramName, GLuint texobj)
{
  CGparameter param;

  param = cgGetNamedEffectParameter(effect, paramName);
  if (!param) {
    fprintf(stderr, "%s: expected effect parameter named %s\n",
      programName, paramName);
    exit(1);
  }
  cgGLSetTextureParameter(param, texobj);
  cgSetSamplerState(param);
  return param;
}

static void initOpenGL(void)
{
  unsigned int size, level, face;
  const GLubyte *image;

  glClearColor(0.1, 0.3, 0.6, 0.0);  /* Blue background */
  glEnable(GL_DEPTH_TEST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); /* Tightly packed texture data. */

  glBindTexture(GL_TEXTURE_2D, TO_NORMAL_MAP);
  /* Load each mipmap level of range-compressed 128x128 brick normal
     map texture. */
  for (size = 128, level = 0, image = myBrickNormalMapImage;
       size > 0;
       image += 3*size*size, size /= 2, level++) {
    glTexImage2D(GL_TEXTURE_2D, level,
      GL_RGB8, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
  }

/* OpenGL tokens for cube maps missing from Windows version of <GL/gl.h> */
#define GL_TEXTURE_CUBE_MAP                 0x8513
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X      0x8515

  myCgNormalMapParam = useSamplerParameter(myCgEffect, "normalMap",
                                           TO_NORMAL_MAP);

  glBindTexture(GL_TEXTURE_CUBE_MAP, TO_NORMALIZE_VECTOR_CUBE_MAP);
  /* Load each 32x32 face (without mipmaps) of range-compressed "normalize
     vector" cube map. */
  for (face = 0, image = myNormalizeVectorCubeMapImage;
       face < 6;
       face++, image += 3*32*32) {
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0,
      GL_RGB8, 32, 32, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
  }

  myCgNormalizeCubeParam = useSamplerParameter(myCgEffect, "normalizeCube",
                                               TO_NORMALIZE_VECTOR_CUBE_MAP);
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

static void display(void)
{
  const int sides = 20, rings = 40;
  const float eyeRadius = 18.0,
              eyeElevationRange = 8.0;
  float eyePosition[3];
  CGpass pass;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  eyePosition[0] = eyeRadius * sin(myEyeAngle);
  eyePosition[1] = eyeElevationRange * sin(myEyeAngle);
  eyePosition[2] = eyeRadius * cos(myEyeAngle);

  glLoadIdentity();
  gluLookAt(
    eyePosition[0], eyePosition[1], eyePosition[2], 
    0.0 ,0.0,  0.0,   /* XYZ view center */
    0.0, 1.0,  0.0);  /* Up is in positive Y direction */

  /* Set Cg parameters for the technique's effect. */
  cgGLSetStateMatrixParameter(myCgModelViewProjParam,
    CG_GL_MODELVIEW_PROJECTION_MATRIX,
    CG_GL_MATRIX_IDENTITY);
  cgSetParameter3fv(myCgEyePositionParam, eyePosition);
  cgSetParameter3fv(myCgLightPositionParam, myLightPosition);

  /* Iterate through rendering passes for technique (even
     though bumpdemo.cgfx has just one pass). */
  pass = cgGetFirstPass(myCgTechnique);
  while (pass) {
    cgSetPassState(pass);
    drawFlatPatch(sides, rings);
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

static int myLastElapsedTime;

static void advanceAnimation(void)
{
  const float millisecondsPerSecond = 1000.0f;
  const float radiansPerSecond = 2.5f;
  int now = glutGet(GLUT_ELAPSED_TIME);
  float deltaSeconds = (now - myLastElapsedTime) / millisecondsPerSecond;

  myLastElapsedTime = now;  /* This time become "prior time". */

  myEyeAngle += deltaSeconds * radiansPerSecond;
  if (myEyeAngle > 2*3.14159)
    myEyeAngle -= 2*3.14159f;
}

static void idle(void)
{
  advanceAnimation();
  glutPostRedisplay();
}

static void keyboard(unsigned char c, int x, int y)
{
  static int animating = 0;

  switch (c) {
  case ' ':
    animating = !animating; /* Toggle */
    if (animating) {
      myLastElapsedTime = glutGet(GLUT_ELAPSED_TIME);
      glutIdleFunc(idle);
    } else {
      glutIdleFunc(NULL);
    }  
    break;
  case 'f':
    myDrawFPS = !myDrawFPS;
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
