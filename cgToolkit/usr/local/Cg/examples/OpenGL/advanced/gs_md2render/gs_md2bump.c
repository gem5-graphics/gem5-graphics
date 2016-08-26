
/* gs_md2bump.c - Cg 2.0 geometry program example demonstrating
   per-triangle computation of a surface-local coordinate system
   for bump-mapping based on the texture space of Quake2 MD2 models. */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   2.0 or higher).  Requires a GPU with geometry program support such
   as GeForce 8800. */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <string.h>   /* for strcmp */
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

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed? */
#include <Cg/cgGL.h>  /* 3D API specific Cg runtime API for OpenGL */

#include "matrix.h"
#include "loadtex.h"
#include "md2.h"
#include "md2render.h"
#include "request_vsync.h"

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgGeometryProfile,
                   myCgFragmentProfile;
static CGprogram   myCgVertexProgram,
                   myCgGeometryProgram,
                   myCgFragmentProgram;
static CGparameter myCgVertexParam_modelViewProj,
                   myCgVertexParam_keyFrameBlend,
                   myCgVertexParam_eyePosition,
                   myCgVertexParam_lightPosition,
                   myCgFragmentParam_decalGlossMap,
                   myCgFragmentParam_normalMap;

const char *myProgramName = "gs_md2bump";

static const char *myVertexProgramFileName   = "gs_md2bump.cg",
                  *myVertexProgramName       = "md2bump_vertex",
                  *myGeometryProgramFileName = "gs_md2bump.cg",
                  *myGeometryProgramName     = "md2bump_geometry",
                  *myFragmentProgramFileName = "gs_md2bump.cg",
                  *myFragmentProgramName     = "md2bump_fragment";

static float myLightAngle = 1.3f;   /* Angle in radians light rotates around knight. */
static float myLightHeight = 5.0f;  /* Vertical height of light. */
static float myEyeAngle = 0.095f;    /* Angle in radians eye rotates around knight. */

static float myProjectionMatrix[16];

static int enableSync = 1;  /* Sync buffer swaps to monitor refresh rate. */

static Md2Model *myModel;
static MD2render *myMD2render;
static float myFrameKnob = 0;

/* Texture object names */
enum {
  TO_BOGUS = 0,
  TO_DECAL,
  TO_NORMAL_MAP,
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
static void reshape(int width, int height);
static void visibility(int state);
static void display(void);
static void keyboard(unsigned char c, int x, int y);
static void menu(int item);
static void mouse(int button, int state, int x, int y);
static void motion(int x, int y);

typedef struct {
  const char *md2FileName,
             *decalFileName,
             *glossFileName,
             *heightFileName;
} ModelFiles;

const ModelFiles myKnight = { "knight.md2",
                              "knight_decal.tga",
                              "knight_gloss.tga",
                              "knight_height.tga" };

static void loadModel(const ModelFiles *modelFiles);

int main(int argc, char **argv)
{
  CGparameter param;
  int i;

  glutInitWindowSize(600, 600);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);

  glutInit(&argc, argv);

  for (i=1; i<argc; i++) {
    if (!strcmp("-nosync", argv[i])) {
      enableSync = 0;
    }
  }

  /* Register GLUT window callbacks. */
  glutCreateWindow("gs_md2bump - Cg 2.0 geometry program example of bump mapping");
  glutDisplayFunc(display);
  glutVisibilityFunc(visibility);
  glutKeyboardFunc(keyboard);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_5) {
    fprintf(stderr, "%s: Failed to initialize GLEW. OpenGL 1.5 required.\n", myProgramName);    
    exit(1);
  }

  loadModel(&myKnight);

  requestSynchronizedSwapBuffers(enableSync);
  glClearColor(0.3, 0.3, 0.1, 0);  /* Dark amber background. */
  glEnable(GL_DEPTH_TEST);         /* Hidden surface removal. */
  glEnable(GL_CULL_FACE);          /* Backface culling. */
  glLineWidth(3.0f);               /* Wide lines for wireframe mode */

  /* This program needs geometry profile support */
  if (cgGLGetLatestProfile(CG_GL_GEOMETRY) == CG_PROFILE_UNKNOWN) {
    fprintf(stderr, "%s: geometry profile is not available.\n", myProgramName);
    exit(0);
  }

  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgGLSetDebugMode(CG_FALSE);
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);

  /** Create vertex programs **/
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

#define GET_VERTEX_PARAM(name) \
  myCgVertexParam_##name = \
    cgGetNamedParameter(myCgVertexProgram, #name); \
  checkForCgError("could not get " #name " parameter");

  GET_VERTEX_PARAM(modelViewProj);
  GET_VERTEX_PARAM(keyFrameBlend);
  GET_VERTEX_PARAM(eyePosition);
  GET_VERTEX_PARAM(lightPosition);

  myCgVertexParam_eyePosition =
    cgGetNamedParameter(myCgVertexProgram, "eyePosition");
  checkForCgError("could not get eyePosition parameter");

  myCgVertexParam_lightPosition =
    cgGetNamedParameter(myCgVertexProgram, "lightPosition");
  checkForCgError("could not get lightPosition parameter");

  param = cgGetNamedParameter(myCgVertexProgram, "scaleFactor");
  checkForCgError("could not get scaleFactor parameter");
  cgSetParameter2f(param,
    1.0f/myModel->header.skinWidth, 1.0f/myModel->header.skinHeight);

  /** Create geomtry programs **/
  myCgGeometryProfile = cgGLGetLatestProfile(CG_GL_GEOMETRY);
  cgGLSetOptimalOptions(myCgGeometryProfile);
  checkForCgError("selecting geometry profile");

  myCgGeometryProgram =
    cgCreateProgramFromFile(
      myCgContext,               /* Cg runtime context */
      CG_SOURCE,                 /* Program in human-readable form */
      myGeometryProgramFileName,
      myCgGeometryProfile,       /* Profile: latest Geometry profile */
      myGeometryProgramName,     /* Entry function name */
      NULL);                     /* No extra compiler options */
  checkForCgError("creating geometry program from string");
  cgGLLoadProgram(myCgGeometryProgram);
  checkForCgError("loading geometry program");

  /** Creat fragment programs **/
  myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  cgGLSetOptimalOptions(myCgFragmentProfile);
  checkForCgError("selecting fragment profile");

  myCgFragmentProgram =
    cgCreateProgramFromFile(
      myCgContext,               /* Cg runtime context */
      CG_SOURCE,                 /* Program in human-readable form */
      myFragmentProgramFileName,
      myCgFragmentProfile,       /* Profile: latest fragment profile */
      myFragmentProgramName,     /* Entry function name */
      NULL);                     /* No extra compiler options */
  checkForCgError("creating fragment program from string");
  cgGLLoadProgram(myCgFragmentProgram);
  checkForCgError("loading fragment program");

  myCgFragmentParam_decalGlossMap =
    cgGetNamedParameter(myCgFragmentProgram, "decalGlossMap");
  checkForCgError("could not get decalGlossMap parameter");
  cgGLSetTextureParameter(myCgFragmentParam_decalGlossMap, TO_DECAL);
  checkForCgError("setting decal 2D texture");

  myCgFragmentParam_normalMap =
    cgGetNamedParameter(myCgFragmentProgram, "normalMap");
  checkForCgError("could not get normalMap parameter");
  cgGLSetTextureParameter(myCgFragmentParam_normalMap, TO_NORMAL_MAP);
  checkForCgError("setting normalMap 2D texture");

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAddMenuEntry("[w] Toggle wireframe", 'w');
  glutAddMenuEntry("[Esc] Quit", 27);
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

void loadModel(const ModelFiles *modelFiles)
{
  gliGenericImage *image, *alpha;
  int mergeOK;

  /* Load Quake2 MD2 model. */
  myModel = md2ReadModel(modelFiles->md2FileName);
  if (NULL == myModel) {
    fprintf(stderr, "%s: count not load %s\n",
      myProgramName, modelFiles->md2FileName);
    exit(1);
  }
  myMD2render = createMD2render(myModel);

  /* Load decal, load gloss map, merge decal and gloss maps, and
     create OpenGL texture object. */
  image = readImage(modelFiles->decalFileName);
  if (NULL == image) {
    printf("%s: failed to load decal skin %s\n",
      myProgramName, modelFiles->decalFileName);
    exit(1);
  }
  if (modelFiles->glossFileName) {
    alpha = readImage(modelFiles->glossFileName);
    if (NULL == alpha) {
      printf("%s: failed to load gloss map skin %s\n",
        myProgramName, modelFiles->glossFileName);
      exit(1);
    }
    mergeOK = gliMergeAlpha(image, alpha);
    if (!mergeOK) {
      printf("%s: failed to merge gloss map\n", myProgramName);
      exit(1);
    }
    gliFree(alpha);
  }

  glBindTexture(GL_TEXTURE_2D, TO_DECAL);
  image = loadTextureDecal(image, 1);
  gliFree(image);
  image = NULL;

  /* Load height map, convert to normal map, and create OpenGL texture object. */
  image = readImage(modelFiles->heightFileName);
  if (NULL == image) {
    printf("%s: failed to load height map skin %s\n",
      myProgramName, modelFiles->heightFileName);
    exit(1);
  }

  glBindTexture(GL_TEXTURE_2D, TO_NORMAL_MAP);
  loadTextureNormalMap(image, modelFiles->heightFileName, 5.0f);
  gliFree(image);
}

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
            frameB = addDelta(myFrameKnob, 1, myModel->header.numFrames);

  float viewMatrix[16], modelViewProjMatrix[16];

  buildLookAtMatrix(eyePosition[0], eyePosition[1], eyePosition[2],
                    0, 0, 0,
                    0, 1, 0,
                    viewMatrix);

  /* modelViewProj = projectionMatrix * viewMatrix (model is identity) */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, viewMatrix);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  cgGLBindProgram(myCgVertexProgram);
  checkForCgError("binding vertex program");
  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  cgGLBindProgram(myCgGeometryProgram);
  checkForCgError("binding geometry program");
  cgGLEnableProfile(myCgGeometryProfile);
  checkForCgError("enabling geometry profile");

  cgGLBindProgram(myCgFragmentProgram);
  checkForCgError("binding fragment program");
  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgSetParameter1f(myCgVertexParam_keyFrameBlend, myFrameKnob-floor(myFrameKnob));
  /* Set eye and light positions if lighting. */
  cgSetParameter3fv(myCgVertexParam_lightPosition, lightPosition);
  cgSetParameter3fv(myCgVertexParam_eyePosition, eyePosition);

  cgUpdateProgramParameters(myCgVertexProgram);
  cgUpdateProgramParameters(myCgGeometryProgram);
  cgUpdateProgramParameters(myCgFragmentProgram);

  cgGLEnableTextureParameter(myCgFragmentParam_decalGlossMap);
  checkForCgError("enable texture decal map");
  cgGLEnableTextureParameter(myCgFragmentParam_normalMap);
  checkForCgError("enable texture normal map");

  /* Draw the model */
  drawMD2render(myMD2render, frameA, frameB);

  /* Disable all profiles */
  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");
  cgGLDisableProfile(myCgGeometryProfile);
  checkForCgError("disabling geometry profile");
  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  glEnable(GL_DEPTH_TEST);
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

  glutSwapBuffers();
}

static int myLastElapsedTime;
static float myKeyFramesPerSecond = 3.0f;

static void idle(void)
{
  const float millisecondsPerSecond = 1000.0f;
  int now = glutGet(GLUT_ELAPSED_TIME);
  float delta = (now - myLastElapsedTime) / millisecondsPerSecond;

  myLastElapsedTime = now;  /* This time become "prior time". */

  delta *= myKeyFramesPerSecond;
  myFrameKnob = addDelta(myFrameKnob, delta, myModel->header.numFrames);
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
  case 'w':
    wireframe = !wireframe;
    if (wireframe) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    break;
  case '+':
    if (myKeyFramesPerSecond < 120) {
      myKeyFramesPerSecond *= 1.4;
    }
    break;
  case '-':
    if (myKeyFramesPerSecond > 0.5) {
      myKeyFramesPerSecond /= 1.4;
    }
    break;
  case 27:  /* Esc key */
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

/* Variables used by motion & mouse callbacks to control view/light. */
static int beginx, beginy;
static int moving = 0;
static int movingLight = 0;
static int xLightBegin, yLightBegin;

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
