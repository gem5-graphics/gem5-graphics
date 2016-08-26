
/* cgfx_latest.c - a Cg 2.2 demo demonstrating the "latest" profile
   string and cgSetStateLatestProfile usage.  Command line options are
   used to programmatically over-ride the "latest" profile behavior. */

/* What to try:
   1) run "cgfx_latest -arb"; then hit 'd' in window and notice ARB
      assembly code is the output compiled code.
   2) run "cgfx_latest -nv30"; then hit 'd' in window and notice GeForce 5
      OpenGL assembly code is the output compiled code.
*/

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>    /* for exit */
#include <string.h>    /* for strcmp */
#if __APPLE__
#include <GLUT/glut.h> /* OpenGL Utility Toolkit (GLUT) */
#else
#include <GL/glut.h>   /* OpenGL Utility Toolkit (GLUT) */
#endif
#include <Cg/cg.h>     /* Cg Core API: Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>   /* Cg OpenGL API (part of Cg Toolkit) */

static const char *myProgramName = "cgfx_latest"; /* Program name for messages. */

/* Cg global variables */
CGcontext   myCgContext;
CGeffect    myCgEffect;
CGtechnique myCgTechnique;
CGparameter myCgEyePositionParam,
            myCgLightPositionParam,
            myCgModelViewProjParam;

/* Forward declare helper functions and callbacks registered by main. */
static void processCommandLineForLatestProfiles(int argc, char **argv);
static void checkForCgError(const char *situation);
static void display(void);
static void reshape(int width, int height);
static void keyboard(unsigned char c, int x, int y);
static void initCg();
static void initOpenGL();

int main(int argc, char **argv)
{
  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgGLSetDebugMode( CG_FALSE );
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);

  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(640, 480);
  glutInit(&argc, argv);
  processCommandLineForLatestProfiles(argc, argv);

  glutCreateWindow("cgfx_latest (OpenGL)");
  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);

  initCg();
  initOpenGL();

  glutMainLoop();
  return 0;
}

static CGprofile latestVertexProfile = 0;
static CGprofile latestFragmentProfile = 0;

static void processCommandLineForLatestProfiles(int argc, char **argv)
{
  int i;

  for (i=1; i<argc; i++) {
    if (!strcmp("-nv30", argv[i])) {
      latestVertexProfile = CG_PROFILE_VP30;
      latestFragmentProfile = CG_PROFILE_FP30;
    } else
    if (!strcmp("-nv40", argv[i])) {
      latestVertexProfile = CG_PROFILE_VP40;
      latestFragmentProfile = CG_PROFILE_FP40;
    } else
    if (!strcmp("-arb", argv[i])) {
      latestVertexProfile = CG_PROFILE_ARBVP1;
      latestFragmentProfile = CG_PROFILE_ARBFP1;
    } else
    if (!strcmp("-gp4", argv[i])) {
      latestVertexProfile = CG_PROFILE_GP4VP;
      latestFragmentProfile = CG_PROFILE_GP4FP;
    } else {
      fprintf(stderr, "%s: Unknown option %s.\n",
        myProgramName, argv[i]);
      fprintf(stderr, "Valid options:\n"
                      "  -nv30 :: GeForce 5 series functionality\n"
                      "  -nv40 :: GeForce 6 & 7 series functionality\n"
                      "  -gp4  :: GeForce 8 & 9 series functionality\n"
                      "  -arb  :: Multi-vendor functionality\n");
      exit(1);
    }
  }
}

/* If requested, over-ride the "latest" profile for the vertex and
   fragment state assignments. */
static void registerLatestProfiles(void)
{
  CGstate vpState, vsState, fpState, psState;

  /* To be comprehensive, change both the OpenGL-style "VertexProgram"
     and Direct3D-style "VertexShader" state names. */
  vpState = cgGetNamedState(myCgContext, "VertexProgram");
  vsState = cgGetNamedState(myCgContext, "VertexShader");

  assert(CG_PROGRAM_TYPE == cgGetStateType(vpState));
  assert(CG_PROGRAM_TYPE == cgGetStateType(vsState));

  if (latestVertexProfile) {
    cgSetStateLatestProfile(vpState, latestVertexProfile);
    cgSetStateLatestProfile(vsState, latestVertexProfile);
  }

  printf("VertexProgram latest profile = %s\n", cgGetProfileString(cgGetStateLatestProfile(vpState)));
  printf("VertexShader latest profile = %s\n", cgGetProfileString(cgGetStateLatestProfile(vsState)));

  /* To be comprehensive, change both the OpenGL-style "FragmentProgram"
     and Direct3D-style "FragmentShader" state names. */
  fpState = cgGetNamedState(myCgContext, "FragmentProgram");
  psState = cgGetNamedState(myCgContext, "PixelShader");

  assert(CG_PROGRAM_TYPE == cgGetStateType(fpState));
  assert(CG_PROGRAM_TYPE == cgGetStateType(psState));

  if (latestFragmentProfile) {
    cgSetStateLatestProfile(fpState, latestFragmentProfile);
    cgSetStateLatestProfile(psState, latestFragmentProfile);
  }

  printf("FragmentProgram latest profile = %s\n", cgGetProfileString(cgGetStateLatestProfile(fpState)));
  printf("PixelShader latest profile = %s\n", cgGetProfileString(cgGetStateLatestProfile(psState)));
}

static void checkForCgError(const char *situation)
{
  CGerror error;
  const char *string = cgGetLastErrorString(&error);
  
  if (error != CG_NO_ERROR) {
    if (error == CG_COMPILER_ERROR) {
      fprintf(stderr,
             "Program: %s\n"
             "Situation: %s\n"
             "Error: %s\n\n"
             "Cg compiler output...\n%s",
             myProgramName, situation, string,
             cgGetLastListing(myCgContext));
    } else {
      fprintf(stderr,
              "Program: %s\n"
              "Situation: %s\n"
              "Error: %s",
              myProgramName, situation, string);
    }
    exit(1);
  }
}

static void initCg(void)
{
  cgGLRegisterStates(myCgContext);
  checkForCgError("registering standard CgFX states");
  registerLatestProfiles();
  checkForCgError("registering command line latest profile setting for CgFX program states");
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
  checkForCgError("manage texture parameters");

  myCgEffect = cgCreateEffectFromFile(myCgContext, "latest.cgfx", NULL);
  checkForCgError("creating bumpdemo.cgfx effect");
  assert(myCgEffect);

  myCgTechnique = cgGetFirstTechnique(myCgEffect);
  while (myCgTechnique && cgValidateTechnique(myCgTechnique) == CG_FALSE) {
    fprintf(stderr, "%s: Technique %s did not validate.  Skipping.\n",
      myProgramName, cgGetTechniqueName(myCgTechnique));
    myCgTechnique = cgGetNextTechnique(myCgTechnique);
  }
  if (myCgTechnique) {
    fprintf(stderr, "%s: Use technique %s.\n",
      myProgramName, cgGetTechniqueName(myCgTechnique));
  } else {
    fprintf(stderr, "%s: No valid technique\n",
      myProgramName);
    exit(1);
  }

  myCgModelViewProjParam =
    cgGetEffectParameterBySemantic(myCgEffect, "ModelViewProjection");
  if (!myCgModelViewProjParam) {
    fprintf(stderr,
      "%s: must find parameter with ModelViewProjection semantic\n",
      myProgramName);
    exit(1);
  }
  myCgEyePositionParam =
    cgGetNamedEffectParameter(myCgEffect, "EyePosition");
  if (!myCgEyePositionParam) {
    fprintf(stderr, "%s: must find parameter named EyePosition\n",
      myProgramName);
    exit(1);
  }
  myCgLightPositionParam =
    cgGetNamedEffectParameter(myCgEffect, "LightPosition");
  if (!myCgLightPositionParam) {
    fprintf(stderr, "%s: must find parameter named LightPosition\n",
      myProgramName);
    exit(1);
  }
}

static const GLubyte
myBrickNormalMapImage[3*(128*128+64*64+32*32+16*16+8*8+4*4+2*2+1*1)] = {
/* RGB8 image data for a mipmapped 128x128 normal map for a brick pattern */
#include "brick_image.h"
};

static void useSamplerParameter(CGeffect effect,
                                const char *paramName, GLuint texobj)
{
  CGparameter param = cgGetNamedEffectParameter(effect, paramName);

  if (!param) {
    fprintf(stderr, "%s: expected effect parameter named %s\n",
      myProgramName, paramName);
    exit(1);
  }
  cgGLSetTextureParameter(param, texobj);
  cgSetSamplerState(param);
}

/* OpenGL texture object (TO) handles. */
enum {
  TO_BOGUS = 0,
  TO_NORMAL_MAP = 1,
};

static void initOpenGL(void)
{
  unsigned int size, level;
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

  useSamplerParameter(myCgEffect, "normalMap",
                      TO_NORMAL_MAP);
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

const int myTorusSides = 20,
          myTorusRings = 40;

/* Initial scene state */
static int myAnimating = 0;
static float myEyeAngle = 0;
static const float myLightPosition[3] = { -8, 0, 15 };

static void display(void)
{
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
    drawFlatPatch(myTorusSides, myTorusRings);
    cgResetPassState(pass);
    pass = cgGetNextPass(pass);
  }

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

static void dumpPrograms(void)
{
  /* Iterate through rendering passes for technique (even
     though bumpdemo.cgfx has just one pass). */
  CGpass pass = cgGetFirstPass(myCgTechnique);
  while (pass) {
    CGstateassignment sa = cgGetFirstStateAssignment(pass);
    while (sa) {
      CGprogram program = cgGetProgramStateAssignmentValue(sa);
      if (program) {
        const char *compiledCode = cgGetProgramString(program, CG_COMPILED_PROGRAM);

        if (compiledCode) {
          printf("^^^^^^^^^^^^\n%s\nvvvvvvvvvvvv\n", compiledCode);
        }
      }
      sa = cgGetNextStateAssignment(sa);
    }
    pass = cgGetNextPass(pass);
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
  case 'd':
    dumpPrograms();
    break;
  case 'W':
    wireframe = !wireframe;
    if (wireframe) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
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
