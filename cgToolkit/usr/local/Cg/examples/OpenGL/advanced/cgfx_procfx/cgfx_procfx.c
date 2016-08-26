
/* cgfx_procfx.c - a CgFX 1.5 procedural effect demo */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>   /* for exit */
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <Cg/cg.h>
#include <Cg/cgGL.h>

#include "cgfx_dump.h"

/* Cg global variables */
CGcontext   myCgContext;
CGeffect    myCgEffect;
CGtechnique myCgTechnique;
CGparameter myCgNormalMapParam,
            myCgNormalizeCubeParam,
            myCgEyePositionParam,
            myCgLightPositionParam,
            myCgModelViewProjParam;

static const char *myProgramName = "cgfx_procfx", /* Program name for messages. */
                  *myVertexProgramFileName = "C8E6v_torus.cg",
                  *myVertexmyProgramName = "C8E6v_torus",
                  *myFragmentProgramFileName = "C8E4f_specSurf.cg",
                  *myFragmentmyProgramName = "C8E4f_specSurf";

/* Initial scene state */
static float myEyeAngle = 0;
static float myLightPosition[3] = { -8, 0, 15 },
             myAmbient = { 0.3f }, /* Dull white */
             myLMd[4] = { 0.9f, 0.6f, 0.3f, 1.0f },     /* Gold */
             myLMs[4] = { 1.0f, 1.0f, 1.0f, 1.0f };     /* Bright white */

static void display(void);
static void reshape(int width, int height);
static void keyboard(unsigned char c, int x, int y);
static void initCg();
static void initOpenGL();

int main(int argc, char **argv)
{
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(640, 480);
  glutInit(&argc, argv);
  glutCreateWindow(myProgramName);

  initCg();
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
      myProgramName, situation, cgGetErrorString(error));
    if (error == CG_COMPILER_ERROR) {
      printf("%s\n", cgGetLastListing(myCgContext));
    }
    exit(1);
  }
}

static void checkForValidCgHandle(CGhandle handle, const char *situation)
{
  if (handle == 0) {
    printf("%s: INVALID handle: %s\n",
      myProgramName, situation);
    exit(1);
  }
}

static void checkForSuccess(CGbool success, const char *situation)
{
  if (success == CG_FALSE) {
    printf("%s: FAILED: %s\n",
      myProgramName, situation);
    exit(1);
  }
}

static void connectEffectParameterToProgramParameter(CGparameter effectParameter,
                                                     CGprogram program,
                                                     const char *programParameterString)
{
  CGparameter programParameter = cgGetNamedParameter(program, programParameterString);

  cgConnectParameter(effectParameter, programParameter);
  checkForCgError(cgGetError(), "connect effect program to program parameter");
}

static void setProgramStateAssignment(CGpass pass, const char *stateName, CGprogram program)
{
  CGstate state;
  CGstateassignment sa;
  int ok;

  state = cgGetNamedState(myCgContext, stateName);
  sa = cgCreateStateAssignment(pass, state);
  ok = cgSetProgramStateAssignment(sa, program);
  checkForSuccess(ok, "set program state assignment");
}

static void setBoolSamplerStateAssignment(CGparameter param, const char *stateName, int value)
{
  CGstate state;
  CGstateassignment sa;
  int ok;

  state = cgGetNamedSamplerState(myCgContext, stateName);
  sa = cgCreateSamplerStateAssignment(param, state);
  ok = cgSetBoolStateAssignment(sa, value);
  checkForSuccess(ok, "set bool sampler state assignment");
}

static void setIntSamplerStateAssignment(CGparameter param, const char *stateName, int value)
{
  CGstate state;
  CGstateassignment sa;
  int ok;

  state = cgGetNamedSamplerState(myCgContext, stateName);
  sa = cgCreateSamplerStateAssignment(param, state);
  ok = cgSetIntStateAssignment(sa, value);
  checkForSuccess(ok, "set int sampler state assignment");
}

static void setTextureSamplerStateAssignment(CGparameter param, const char *stateName, CGparameter value)
{
  CGstate state;
  CGstateassignment sa;
  int ok;

  state = cgGetNamedSamplerState(myCgContext, stateName);
  sa = cgCreateSamplerStateAssignment(param, state);
  ok = cgSetTextureStateAssignment(sa, value);
  checkForSuccess(ok, "set int sampler state assignment");
}

static void buildCgEffect(void)
{
  CGparameter texParam;
  CGparameter param;
  CGpass pass;
  CGprofile vertexProfile, fragmentProfile;
  CGprogram vertexProgram, fragmentProgram;
  CGbool ok;

  // Create a context, register OpenGL states, and manage textures.
  myCgContext = cgCreateContext();
  cgGLSetDebugMode( CG_FALSE );
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);
  cgGLRegisterStates(myCgContext);
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
  checkForCgError(cgGetError(), "establishing Cg context");

  // Create an empty effect.
  myCgEffect = cgCreateEffect(myCgContext, NULL, NULL);
  checkForCgError(cgGetError(), "creating empty effect");

  // Add an empty technique and pass to the empty effect.
  myCgTechnique = cgCreateTechnique(myCgEffect, "bumpdemo");
  checkForValidCgHandle(myCgTechnique, "creating technique");
  pass = cgCreatePass(myCgTechnique, "single");
  checkForValidCgHandle(pass, "creating pass");

  // Pick the best vertex profile.
  vertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
  cgGLSetOptimalOptions(vertexProfile);
  checkForCgError(cgGetError(), "selecting vertex profile");

  // Load vertex program from file into context.
  vertexProgram =
    cgCreateProgramFromFile(
      myCgContext,              /* Cg runtime context */
      CG_SOURCE,                /* Program in human-readable form */
      myVertexProgramFileName,  /* Name of file containing program */
      vertexProfile,            /* Profile: OpenGL ARB vertex program */
      myVertexmyProgramName,    /* Entry function name */
      NULL);                    /* No extra compiler options */
  checkForCgError(cgGetError(), "creating vertex program");

  // Create and set "VertexProgram" state assignment to the program.
  setProgramStateAssignment(pass, "VertexProgram", vertexProgram);

  // Pick the best fragment profile.
  fragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  cgGLSetOptimalOptions(fragmentProfile);
  checkForCgError(cgGetError(), "selecting fragment profile");

  // Load fragment program from file into context.
  fragmentProgram =
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myFragmentProgramFileName,  /* Name of file containing program */
      fragmentProfile,            /* Profile: OpenGL ARB fragment program */
      myFragmentmyProgramName,    /* Entry function name */
      NULL);                      /* No extra compiler options */
  checkForCgError(cgGetError(), "creating fragment program");

  // Create and set "FragmentProgram" state assignment to the program.
  setProgramStateAssignment(pass, "FragmentProgram", fragmentProgram);

  // Validate the one-pass technique.
  ok = cgValidateTechnique(myCgTechnique);
  checkForSuccess(ok, "validating technique");

  // Create, set, and connect vertex program effect parameters.

  param = cgCreateEffectParameter(myCgEffect, "ModelViewProj", CG_FLOAT4x4);
  cgSetParameterSemantic(param, "ModelViewProjection");
  connectEffectParameterToProgramParameter(param, vertexProgram, "modelViewProj");

  param = cgCreateEffectParameter(myCgEffect, "LightPosition", CG_FLOAT3);
  cgSetParameter3f(param, -8, 0, 15);
  checkForCgError(cgGetError(), "setting light position");
  connectEffectParameterToProgramParameter(param, vertexProgram, "lightPosition");

  param = cgCreateEffectParameter(myCgEffect, "EyePosition", CG_FLOAT3);
  cgSetParameter3f(param, 0, 0, 18);
  checkForCgError(cgGetError(), "setting eye position");
  connectEffectParameterToProgramParameter(param, vertexProgram, "eyePosition");

  param = cgCreateEffectParameter(myCgEffect, "TorusInfo", CG_FLOAT2);
  cgSetParameter2f(param, 6, 2);
  checkForCgError(cgGetError(), "setting torus info");
  connectEffectParameterToProgramParameter(param, vertexProgram, "torusInfo");

  // Create, set, and connect fragment program effect parameters.

  param = cgCreateEffectParameter(myCgEffect, "Ambient", CG_FLOAT);
  cgSetParameter1f(param, myAmbient);
  checkForCgError(cgGetError(), "setting ambient");
  connectEffectParameterToProgramParameter(param, fragmentProgram, "ambient");

  param = cgCreateEffectParameter(myCgEffect, "DiffuseMaterialTimesLightColor", CG_FLOAT4);
  cgSetParameter4fv(param, myLMd);
  checkForCgError(cgGetError(), "setting diffuse material times light color");
  connectEffectParameterToProgramParameter(param, fragmentProgram, "LMd");

  param = cgCreateEffectParameter(myCgEffect, "SpecularMaterialTimesLightColor", CG_FLOAT4);
  cgSetParameter4fv(param, myLMs);
  checkForCgError(cgGetError(), "setting specular material times light color");
  connectEffectParameterToProgramParameter(param, fragmentProgram, "LMs");

  param = cgCreateEffectParameter(myCgEffect, "normalMap", CG_SAMPLER2D);
  setBoolSamplerStateAssignment(param, "generateMipMap", GL_TRUE);
  setIntSamplerStateAssignment(param, "minFilter", GL_LINEAR_MIPMAP_LINEAR);
  setIntSamplerStateAssignment(param, "magFilter", GL_LINEAR);
  connectEffectParameterToProgramParameter(param, fragmentProgram, "normalMap");

/* An OpenGL 1.2 define */
#define GL_CLAMP_TO_EDGE                    0x812F

  texParam = cgCreateEffectParameter(myCgEffect, "normalizeCubeTexture", CG_TEXTURE);

  param = cgCreateEffectParameter(myCgEffect, "normalizeCube", CG_SAMPLERCUBE);
  setTextureSamplerStateAssignment(param, "Texture", texParam);
  setIntSamplerStateAssignment(param, "minFilter", GL_LINEAR);
  setIntSamplerStateAssignment(param, "magFilter", GL_LINEAR);
  setIntSamplerStateAssignment(param, "wrapS", GL_CLAMP_TO_EDGE);
  setIntSamplerStateAssignment(param, "wrapT", GL_CLAMP_TO_EDGE);
  connectEffectParameterToProgramParameter(param, fragmentProgram, "normalizeCube");
  connectEffectParameterToProgramParameter(param, fragmentProgram, "normalizeCube2");
}

static void initCg(void)
{
  // Use new Cg 1.5 API to build an effect procedurally 
  // (rather than simply loading it from a file).
  buildCgEffect();
  
  dumpCgContext(myCgContext, CG_FALSE);

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
      myProgramName, paramName);
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
    cgDestroyEffect(myCgEffect);
    cgDestroyContext(myCgContext);
    exit(0);
    break;
  }
}
