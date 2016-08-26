
/* combine_program.c - demonstration of cgCombinePrograms to combine
   vertex and fragment programs; based on 24_bump_map_torus example */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   1.5 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <string.h>   /* for strcmp */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sqrt, sin, and cos */
#include <assert.h>   /* for assert */
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>

/* An OpenGL 1.2 define */
#define GL_CLAMP_TO_EDGE                    0x812F

/* A few OpenGL 1.3 defines */
#define GL_TEXTURE_CUBE_MAP                 0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP         0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X      0x8515

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgFragmentProfile;
static CGprogram   myCgComboProgram;  // Just one program handle!
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

static const char *myProgramName = "combine_program",
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

int main(int argc, char **argv)
{
  int useGLSL = 0;
  const GLubyte *image;
  unsigned int size, level;
  int face;
  CGprogram programList[2];

  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInit(&argc, argv);

  for (argv++; *argv; argv++) {
    if (!strcmp(*argv, "-glsl")) {
      useGLSL = 1;
      printf("%s: GLSL profiles requested\n", myProgramName);
    }
  }

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);

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
  cgGLSetDebugMode( CG_FALSE );
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);
  checkForCgError("creating context");

  if (useGLSL) {
    myCgVertexProfile = CG_PROFILE_GLSLV;
  } else {
    myCgVertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
  }
  cgGLSetOptimalOptions(myCgVertexProfile);
  checkForCgError("selecting vertex profile");

  programList[0] =
    cgCreateProgramFromFile(
      myCgContext,              /* Cg runtime context */
      CG_SOURCE,                /* Program in human-readable form */
      myVertexProgramFileName,  /* Name of file containing program */
      myCgVertexProfile,        /* Profile: OpenGL ARB vertex program */
      myVertexProgramName,      /* Entry function name */
      NULL);                    /* No extra compiler options */
  checkForCgError("creating vertex program from file");

  if (useGLSL) {
    myCgFragmentProfile = CG_PROFILE_GLSLF;
  } else {
    myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  }
  cgGLSetOptimalOptions(myCgFragmentProfile);
  checkForCgError("selecting fragment profile");

  programList[1] =
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myFragmentProgramFileName,  /* Name of file containing program */
      myCgFragmentProfile,        /* Profile: OpenGL ARB fragment program */
      myFragmentProgramName,      /* Entry function name */
      NULL);                      /* No extra compiler options */
  checkForCgError("creating fragment program from file");

  /* Combine vertex and fragment programs */
  myCgComboProgram = cgCombinePrograms(2, programList);
  checkForCgError("combining programs");
  assert(2 == cgGetNumProgramDomains(myCgComboProgram));

  cgDestroyProgram(programList[0]);
  cgDestroyProgram(programList[1]);
  checkForCgError("destroying original programs after combining");

  cgGLLoadProgram(myCgComboProgram);
  checkForCgError("loading combo program");

  myCgVertexParam_lightPosition =
    cgGetNamedParameter(myCgComboProgram, "lightPosition");
  checkForCgError("could not get lightPosition parameter");

  myCgVertexParam_eyePosition =
    cgGetNamedParameter(myCgComboProgram, "eyePosition");
  checkForCgError("could not get eyePosition parameter");

  myCgVertexParam_modelViewProj =
    cgGetNamedParameter(myCgComboProgram, "modelViewProj");
  checkForCgError("could not get modelViewProj parameter");

  myCgVertexParam_torusInfo =
    cgGetNamedParameter(myCgComboProgram, "torusInfo");
  checkForCgError("could not get torusInfo parameter");

  myCgFragmentParam_ambient =
    cgGetNamedParameter(myCgComboProgram, "ambient");
  checkForCgError("getting ambient parameter");

  myCgFragmentParam_LMd =
    cgGetNamedParameter(myCgComboProgram, "LMd");
  checkForCgError("getting LMd parameter");

  myCgFragmentParam_LMs =
    cgGetNamedParameter(myCgComboProgram, "LMs");
  checkForCgError("getting LMs parameter");

  myCgFragmentParam_normalMap =
    cgGetNamedParameter(myCgComboProgram, "normalMap");
  checkForCgError("getting normalMap parameter");

  myCgFragmentParam_normalizeCube =
    cgGetNamedParameter(myCgComboProgram, "normalizeCube");
  checkForCgError("getting normalizeCube parameter");

  myCgFragmentParam_normalizeCube2 =
    cgGetNamedParameter(myCgComboProgram, "normalizeCube2");
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

  cgGLBindProgram(myCgComboProgram);
  checkForCgError("binding combined program");

  cgGLSetStateMatrixParameter(myCgVertexParam_modelViewProj,
                              CG_GL_MODELVIEW_PROJECTION_MATRIX,
                              CG_GL_MATRIX_IDENTITY);
  checkForCgError("setting modelview-projection matrix");
  cgGLSetParameter3f(myCgVertexParam_lightPosition, -8, 0, 15);
  checkForCgError("setting light position");
  cgGLSetParameter3fv(myCgVertexParam_eyePosition, eyePosition);
  checkForCgError("setting eye position");
  cgGLSetParameter2f(myCgVertexParam_torusInfo, outerRadius, innerRadius);
  checkForCgError("setting torus information");

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  cgGLSetParameter4fv(myCgFragmentParam_ambient, myAmbient);
  checkForCgError("setting ambient");
  cgGLSetParameter4fv(myCgFragmentParam_LMd, myLMd);
  checkForCgError("setting diffuse material");
  cgGLSetParameter4fv(myCgFragmentParam_LMs, myLMs);
  checkForCgError("setting specular material");

  cgGLEnableTextureParameter(myCgFragmentParam_normalMap);
  checkForCgError("enable texture normal map");
  cgGLEnableTextureParameter(myCgFragmentParam_normalizeCube);
  checkForCgError("enable 1st normalize vector cube map");
  cgGLEnableTextureParameter(myCgFragmentParam_normalizeCube2);
  checkForCgError("enable 2nd normalize vector cube map");

  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  cgUpdateProgramParameters(myCgComboProgram);
  drawFlatPatch(sides, rings);

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  glutSwapBuffers();
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
  case 27:  /* Esc key */
    /* Demonstrate proper deallocation of Cg runtime data structures.
       Not strictly necessary if we are simply going to exit. */
    cgDestroyProgram(myCgComboProgram);
    exit(0);
    break;
  }
}

static void menu(int item)
{
  /* Pass menu item character code to keyboard callback. */
  keyboard((unsigned char)item, 0, 0);
}
