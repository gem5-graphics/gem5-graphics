
/* 24_bump_map_torus.c - OpenGL-based bump mapping example using Cg programs
   from Chapter 8 of "The Cg Tutorial" (Addison-Wesley, ISBN 0321194969). */

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
static CGparameter myCgVertexParam_lightPosition,
                   myCgVertexParam_eyePosition,
                   myCgVertexParam_modelViewProj,
                   myCgVertexParam_torusInfo,
                   myCgFragmentParam_ambient,
                   myCgFragmentParam_LMd,
                   myCgFragmentParam_LMs,
                   myCgFragmentParam_normalMap,
                   myCgFragmentParam_normalizeCube,
                   myCgFragmentParam_normalizeCube2;

static const char *myProgramName = "24_bump_map_torus",
                  *myVertexProgramFileName = "C8E6v_torus.cg",
/* page 223 */    *myVertexProgramName = "C8E6v_torus",
                  *myFragmentProgramFileName = "C8E4f_specSurf.cg",
/* page 209 */    *myFragmentProgramName = "C8E4f_specSurf";

static float myEyeAngle = 0,
             myAmbient[4] = { 0.3f, 0.3f, 0.3f, 0.3f }, /* Dull white */
             myLMd[4] = { 0.9f, 0.6f, 0.3f, 1.0f },     /* Gold */
             myLMs[4] = { 1.0f, 1.0f, 1.0f, 1.0f };     /* Bright white */

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
static void display(void);
static void keyboard(unsigned char c, int x, int y);
static void menu(int item);
static void requestSynchronizedSwapBuffers(void);

int main(int argc, char **argv)
{
  const GLubyte *image;
  unsigned int size, level;
  int face;

  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_3) {
    fprintf(stderr, "%s: failed to initialize GLEW, OpenGL 1.3 required.\n", myProgramName);    
    exit(1);
  }

  requestSynchronizedSwapBuffers();
  glClearColor(0.1, 0.3, 0.6, 0.0);  /* Blue background */
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(
    60.0,   /* Field of view in degree */
    1.0,    /* Aspect ratio */ 
    0.1,    /* Z near */
    100.0); /* Z far */
  glMatrixMode(GL_MODELVIEW);
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
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
    GL_LINEAR_MIPMAP_LINEAR);

  glBindTexture(GL_TEXTURE_CUBE_MAP, TO_NORMALIZE_VECTOR_CUBE_MAP);
  /* Load each 32x32 face (without mipmaps) of range-compressed "normalize
     vector" cube map. */
  for (face = 0, image = myNormalizeVectorCubeMapImage;
       face < 6;
       face++, image += 3*32*32) {
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0,
      GL_RGB8, 32, 32, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
  }
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

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

  myCgVertexParam_lightPosition =
    cgGetNamedParameter(myCgVertexProgram, "lightPosition");
  checkForCgError("could not get lightPosition parameter");

  myCgVertexParam_eyePosition =
    cgGetNamedParameter(myCgVertexProgram, "eyePosition");
  checkForCgError("could not get eyePosition parameter");

  myCgVertexParam_modelViewProj =
    cgGetNamedParameter(myCgVertexProgram, "modelViewProj");
  checkForCgError("could not get modelViewProj parameter");

  myCgVertexParam_torusInfo =
    cgGetNamedParameter(myCgVertexProgram, "torusInfo");
  checkForCgError("could not get torusInfo parameter");

  myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  cgGLSetOptimalOptions(myCgFragmentProfile);
  checkForCgError("selecting fragment profile");

  myCgFragmentProgram =
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myFragmentProgramFileName,  /* Name of file containing program */
      myCgFragmentProfile,        /* Profile: OpenGL ARB vertex program */
      myFragmentProgramName,      /* Entry function name */
      NULL);                      /* No extra compiler options */
  checkForCgError("creating fragment program from file");
  cgGLLoadProgram(myCgFragmentProgram);
  checkForCgError("loading fragment program");

  myCgFragmentParam_ambient =
    cgGetNamedParameter(myCgFragmentProgram, "ambient");
  checkForCgError("getting ambient parameter");

  myCgFragmentParam_LMd =
    cgGetNamedParameter(myCgFragmentProgram, "LMd");
  checkForCgError("getting LMd parameter");

  myCgFragmentParam_LMs =
    cgGetNamedParameter(myCgFragmentProgram, "LMs");
  checkForCgError("getting LMs parameter");

  myCgFragmentParam_normalMap =
    cgGetNamedParameter(myCgFragmentProgram, "normalMap");
  checkForCgError("getting normalMap parameter");

  myCgFragmentParam_normalizeCube =
    cgGetNamedParameter(myCgFragmentProgram, "normalizeCube");
  checkForCgError("getting normalizeCube parameter");

  myCgFragmentParam_normalizeCube2 =
    cgGetNamedParameter(myCgFragmentProgram, "normalizeCube2");
  checkForCgError("getting normalizeCube2 parameter");

  cgGLSetTextureParameter(myCgFragmentParam_normalMap,
    TO_NORMAL_MAP);
  checkForCgError("setting normal map 2D texture");

  cgGLSetTextureParameter(myCgFragmentParam_normalizeCube,
    TO_NORMALIZE_VECTOR_CUBE_MAP);
  checkForCgError("setting 1st normalize vector cube map");

  cgGLSetTextureParameter(myCgFragmentParam_normalizeCube2,
    TO_NORMALIZE_VECTOR_CUBE_MAP);
  checkForCgError("setting 2nd normalize vector cube map");

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

/* Draw a flat 2D patch that can be "rolled & bent" into a 3D torus by
   a vertex program. */
void
drawFlatPatch(float rows, float columns)
{
  const float m = 1.0f/columns;
  const float n = 1.0f/rows;
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

static void display(void)
{
  const float outerRadius = 6, innerRadius = 2;
  const int sides = 20, rings = 40;
  const float eyeRadius = 18.0;
  const float eyeElevationRange = 8.0;
  float eyePosition[3];

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  eyePosition[0] = eyeRadius * sin(myEyeAngle);
  eyePosition[1] = eyeElevationRange * sin(myEyeAngle);
  eyePosition[2] = eyeRadius * cos(myEyeAngle);

  glLoadIdentity();
  gluLookAt(
    eyePosition[0], eyePosition[1], eyePosition[2], 
    0.0 ,0.0,  0.0,   /* XYZ view center */
    0.0, 1.0,  0.0);  /* Up is in positive Y direction */

  cgGLBindProgram(myCgVertexProgram);
  checkForCgError("binding vertex program");

  cgGLSetStateMatrixParameter(myCgVertexParam_modelViewProj,
                              CG_GL_MODELVIEW_PROJECTION_MATRIX,
                              CG_GL_MATRIX_IDENTITY);
  checkForCgError("setting modelview-projection matrix");
  cgSetParameter3f(myCgVertexParam_lightPosition, -8, 0, 15);
  checkForCgError("setting light position");
  cgSetParameter3fv(myCgVertexParam_eyePosition, eyePosition);
  checkForCgError("setting eye position");
  cgSetParameter2f(myCgVertexParam_torusInfo, outerRadius, innerRadius);
  checkForCgError("setting torus information");

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  cgGLBindProgram(myCgFragmentProgram);
  checkForCgError("binding fragment program");

  cgSetParameter4fv(myCgFragmentParam_ambient, myAmbient);
  checkForCgError("setting ambient");
  cgSetParameter4fv(myCgFragmentParam_LMd, myLMd);
  checkForCgError("setting diffuse material");
  cgSetParameter4fv(myCgFragmentParam_LMs, myLMs);
  checkForCgError("setting specular material");

  cgGLEnableTextureParameter(myCgFragmentParam_normalMap);
  checkForCgError("enable texture normal map");
  cgGLEnableTextureParameter(myCgFragmentParam_normalizeCube);
  checkForCgError("enable 1st normalize vector cube map");
  cgGLEnableTextureParameter(myCgFragmentParam_normalizeCube2);
  checkForCgError("enable 2nd normalize vector cube map");

  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  cgUpdateProgramParameters(myCgVertexProgram);
  cgUpdateProgramParameters(myCgFragmentProgram);

  drawFlatPatch(sides, rings);

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  glutSwapBuffers();
}

static void idle(void)
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
