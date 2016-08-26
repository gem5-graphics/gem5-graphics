
/* gs_md2shadow.c - Cg 2.0 geometry program example rendering Quake2
   MD2 models with bump-mapping and shadowing generated via stenciled
   shadow volumes. */

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
#include "showfps.h"

static CGcontext   myCgContext;
static CGeffect    myCgEffect;
static CGtechnique myCgTechnique;
static CGparameter myCgParam_modelViewProj,
                   myCgParam_keyFrameBlend,
                   myCgParam_eyePosition,
                   myCgParam_lightPosition,
                   myCgParam_scaleFactor;

const char *myProgramName = "gs_md2shadow";

static float myLightAngle = 1.59f;   /* Angle in radians light rotates around knight. */
static float myLightHeight = 13.3f;  /* Vertical height of light. */
static float myEyeAngle = 0.095f;    /* Angle in radians eye rotates around knight. */

static float myProjectionMatrix[16];

static int enableSync = 1;  /* Sync buffer swaps to monitor refresh rate. */
static int showBackdrop = 1;  /* Show walls, floor, and ceiling for shadow to fall on. */

static Md2Model *myModel;
static MD2render *myMD2render;
static float myFrameKnob = 0;
static int has_NV_depth_clamp = 0;

/* Texture object names */
enum {
  TO_BOGUS = 0,
  TO_DECAL_GLOSS,
  TO_NORMAL_MAP,
};

void colorizeStencil(void);

static void checkForCgError(const char *situation)
{
  CGerror error;
  const char *string = cgGetLastErrorString(&error);

  if (error != CG_NO_ERROR) {
    printf("%s: %s: %s\n",
      myProgramName, situation, string);
    if (error == CG_COMPILER_ERROR) {
      const char *compilerMessage = cgGetLastListing(myCgContext);

      printf("%s\n", compilerMessage);
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
static void techniqueMenu(int item);
static void mouse(int button, int state, int x, int y);
static void motion(int x, int y);

static void loadModel(void);
static void useSamplerParameter(CGeffect effect, const char *paramName, GLuint texobj);

int main(int argc, char **argv)
{
  CGtechnique technique;
  int techniqueNum;
  int submenu;
  int i;

  glutInitWindowSize(600, 600);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE | GLUT_STENCIL);

  glutInit(&argc, argv);

  for (i=1; i<argc; i++) {
    if (!strcmp("-nosync", argv[i])) {
      enableSync = 0;
    }
  }

  /* Register GLUT window callbacks. */
  glutCreateWindow("gs_md2shadow - Cg 2.0 geometry program example of shadow volumes");
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

  loadModel();

  requestSynchronizedSwapBuffers(enableSync);
  glClearColor(0.3, 0.3, 0.1, 0);  /* Dark amber background. */
  glClearStencil(0);               /* Stencil clears reset stencil to zero. */
  glLineWidth(3.0f);               /* Wide lines for wireframe mode */
  colorFPS(0.7, 0.7, 0.5);         /* Black */

  /* This program needs geometry profile support */
  if (cgGLGetLatestProfile(CG_GL_GEOMETRY) == CG_PROFILE_UNKNOWN) {
    fprintf(stderr, "%s: geometry profile is not available.\n", myProgramName);
    exit(0);
  }

  has_NV_depth_clamp = glutExtensionSupported("GL_NV_depth_clamp");
  if (has_NV_depth_clamp) {
    glEnable(GL_DEPTH_CLAMP_NV);
  }

  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgGLSetDebugMode(CG_FALSE);
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);

  cgGLRegisterStates(myCgContext);
  checkForCgError("registering standard CgFX states");
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
  checkForCgError("manage texture parameters");

  myCgEffect = cgCreateEffectFromFile(myCgContext, "gs_md2shadow.cgfx", NULL);
  checkForCgError("loading effect");

  submenu = glutCreateMenu(techniqueMenu);
  technique = cgGetFirstTechnique(myCgEffect);
  techniqueNum = 0;
  while (technique) {
    const char *techniqueName = cgGetTechniqueName(technique);

    if (cgValidateTechnique(technique)) {
      glutAddMenuEntry(techniqueName, techniqueNum);
    } else {
      fprintf(stderr, "%s: could not validate technique %s\n",
        myProgramName, techniqueName);
    }
    technique = cgGetNextTechnique(technique);
    techniqueNum++;
  }

  myCgTechnique = cgGetFirstTechnique(myCgEffect);
  if (cgValidateTechnique(myCgTechnique)) {
    // Valid technique
  } else {
    const char *lastListing = cgGetLastListing(myCgContext);

    fprintf(stderr, "%s: technique failed to valiate\n", myProgramName);
    if (lastListing) {
        fprintf(stderr, "%s\n", lastListing);
    }
    exit(1);
  }

  myCgParam_modelViewProj = cgGetNamedEffectParameter(myCgEffect, "modelViewProj");
  myCgParam_keyFrameBlend = cgGetNamedEffectParameter(myCgEffect, "keyFrameBlend");
  myCgParam_eyePosition = cgGetNamedEffectParameter(myCgEffect, "eyePosition");
  myCgParam_lightPosition = cgGetNamedEffectParameter(myCgEffect, "lightPosition");
  myCgParam_scaleFactor = cgGetNamedEffectParameter(myCgEffect, "scaleFactor");

  cgSetParameter2f(myCgParam_scaleFactor, 1.0f/myModel->header.skinWidth, 1.0f/myModel->header.skinHeight);

  useSamplerParameter(myCgEffect, "decalGlossMap", TO_DECAL_GLOSS);
  useSamplerParameter(myCgEffect, "normalMap", TO_NORMAL_MAP);

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAddMenuEntry("[w] Toggle wireframe", 'w');
  glutAddMenuEntry("[b] Toggle backdrop", 'b');
  glutAddMenuEntry("[v] Toggle frame synchronization", 'v');
  glutAddMenuEntry("[Enter] Next technique", 13);
  glutAddMenuEntry("[z] Next technique", 'a');
  glutAddMenuEntry("[a] Previous technique", 'z');
  glutAddMenuEntry("[Esc] Quit", 27);
  glutAddSubMenu("Techniques...", submenu);
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

void loadModel(void)
{
  gliGenericImage *image, *alpha;
  int mergeOK;

  const char *md2FileName = "knight.md2",
             *decalFileName = "knight_decal.tga",
             *glossFileName = "knight_gloss.tga",
             *heightFileName = "knight_height.tga";

  /* Load Quake2 MD2 model. */
  myModel = md2ReadModel(md2FileName);
  if (NULL == myModel) {
    fprintf(stderr, "%s: count not load %s\n",
      myProgramName, md2FileName);
    exit(1);
  }
  md2ComputeAdjacencyInfo(myModel);
  myMD2render = createMD2renderWithAdjacency(myModel);

  /* Load decal, load gloss map, merge decal and gloss maps, and
     create OpenGL texture object. */
  image = readImage(decalFileName);
  if (NULL == image) {
    printf("%s: failed to load decal skin %s\n",
      myProgramName, decalFileName);
    exit(1);
  }
  if (glossFileName) {
    alpha = readImage(glossFileName);
    if (NULL == alpha) {
      printf("%s: failed to load gloss map skin %s\n",
        myProgramName, glossFileName);
      exit(1);
    }
    mergeOK = gliMergeAlpha(image, alpha);
    if (!mergeOK) {
      printf("%s: failed to merge gloss map\n", myProgramName);
      exit(1);
    }
    gliFree(alpha);
  }

  glBindTexture(GL_TEXTURE_2D, TO_DECAL_GLOSS);
  image = loadTextureDecal(image, 1);
  gliFree(image);
  image = NULL;

  /* Load height map, convert to normal map, and create OpenGL texture object. */
  image = readImage(heightFileName);
  if (NULL == image) {
    printf("%s: failed to load height map skin %s\n",
      myProgramName, heightFileName);
    exit(1);
  }

  glBindTexture(GL_TEXTURE_2D, TO_NORMAL_MAP);
  loadTextureNormalMap(image, heightFileName, 5.0f);
  gliFree(image);
}

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

static void reshape(int width, int height)
{
  double aspectRatio = (float) width / (float) height;
  double fieldOfView = 60.0; /* Degrees */

  /* Build projection matrix once. */
  if (has_NV_depth_clamp) {
    buildPerspectiveMatrix(fieldOfView, aspectRatio,
      10, 200,  /* Znear */
      myProjectionMatrix);
  } else {
    buildPinfMatrix(fieldOfView, aspectRatio,
      10,  /* Znear */
      myProjectionMatrix);
  }
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

static void drawWalls(void)
{
  const GLfloat xmin = -75.0,
                ymin = -24.0,
                zmin = -75.0;
  const GLfloat xmax = 75.0,
                ymax = 65.0,
                zmax = 75.0;

  glBegin(GL_TRIANGLES);
    /* Right wall */
    glNormal3f(0, 0, 1);
    glVertex3f(xmin, ymin, zmin);
    glVertex3f(xmax, ymin, zmin);
    glVertex3f(xmin, ymax, zmin);

    glVertex3f(xmax, ymax, zmin);
    glVertex3f(xmin, ymax, zmin);
    glVertex3f(xmax, ymin, zmin);

    /* Left wall */
    glNormal3f(0, 0, -1);
    glVertex3f(xmin, ymin, zmax);
    glVertex3f(xmin, ymax, zmax);
    glVertex3f(xmax, ymin, zmax);

    glVertex3f(xmax, ymax, zmax);
    glVertex3f(xmax, ymin, zmax);
    glVertex3f(xmin, ymax, zmax);

    /* Front wall */
    glNormal3f(-1, 0, 0);
    glVertex3f(xmax, ymin, zmin);
    glVertex3f(xmax, ymax, zmax);
    glVertex3f(xmax, ymax, zmin);

    glVertex3f(xmax, ymin, zmin);
    glVertex3f(xmax, ymin, zmax);
    glVertex3f(xmax, ymax, zmax);

    /* Back wall */
    glNormal3f(1, 0, 0);
    glVertex3f(xmin, ymin, zmin);
    glVertex3f(xmin, ymax, zmin);
    glVertex3f(xmin, ymax, zmax);

    glVertex3f(xmin, ymin, zmin);
    glVertex3f(xmin, ymax, zmax);
    glVertex3f(xmin, ymin, zmax);

    /* Ceiling */
    glNormal3f(0, -1, 0);
    glVertex3f(xmin, ymax, zmin);
    glVertex3f(xmax, ymax, zmin);
    glVertex3f(xmax, ymax, zmax);

    glVertex3f(xmin, ymax, zmin);
    glVertex3f(xmax, ymax, zmax);
    glVertex3f(xmin, ymax, zmax);

    /* Floor */
    glNormal3f(0, 1, 0);
    glVertex3f(xmin, ymin, zmin);
    glVertex3f(xmax, ymin, zmax);
    glVertex3f(xmax, ymin, zmin);

    glVertex3f(xmin, ymin, zmin);
    glVertex3f(xmin, ymin, zmax);
    glVertex3f(xmax, ymin, zmax);
  glEnd();
}

static void drawScene(void)
{
  /* World-space positions for light and eye. */
  const float eyeRadius = 70,
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

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgParam_modelViewProj, modelViewProjMatrix);
  cgSetParameter1f(myCgParam_keyFrameBlend, myFrameKnob-floor(myFrameKnob));
  /* Set eye and light positions if lighting. */
  cgSetParameter3fv(myCgParam_eyePosition, eyePosition);
  cgSetParameter3fv(myCgParam_lightPosition, lightPosition);

  {
    CGpass pass = cgGetFirstPass(myCgTechnique);

    while (pass) {
      /* Use the existance of certain annotations to control rendering of passes... */
      CGannotation passNeedsAdjacency, passDrawsBackdrop;

      passDrawsBackdrop = cgGetNamedPassAnnotation(pass, "DrawBackdrop");
      /* If we should render this pass... */
      if (!passDrawsBackdrop || showBackdrop) {
        cgSetPassState(pass);
        if (passDrawsBackdrop) {
          drawWalls();
        } else {
          /* Draw the MD2 model */
          passNeedsAdjacency = cgGetNamedPassAnnotation(pass, "NeedsAdjacency");
          if (passNeedsAdjacency) {
            drawMD2renderWithAdjacency(myMD2render, frameA, frameB);
          } else {
            drawMD2render(myMD2render, frameA, frameB);
          }
        }
        cgResetPassState(pass);
      } else {
        /* Skip setting pass state and rendering for walls when not showing backdrop. */
      }

      pass = cgGetNextPass(pass);
    }
  }

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

  handleFPS();
}

static void showTexture(GLuint texobj, int showAlpha)
{
  glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
      glLoadIdentity();
      glDisable(GL_DEPTH_TEST);
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, texobj);
      if (showAlpha) {
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE);
        glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, GL_REPLACE);
        glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_RGB, GL_TEXTURE);
        glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND0_RGB, GL_SRC_ALPHA);
      } else {
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      }
      glBegin(GL_QUADS);
        glTexCoord2f(0,0);
        glVertex2f(-1,-1);
        glTexCoord2f(1,0);
        glVertex2f(1,-1);
        glTexCoord2f(1,1);
        glVertex2f(1,1);
        glTexCoord2f(0,1);
        glVertex2f(-1,1);
      glEnd();
      glDisable(GL_TEXTURE_2D);
    glPopMatrix();
  glPopMatrix();
}

enum {
  RM_DRAW_SCENE,
  RM_SHOW_NORMAL_MAP,
  RM_SHOW_DECAL,
  RM_SHOW_GLOSS,
  RM_NUM_MODES
} renderMode = RM_DRAW_SCENE;

static void display(void)
{
  switch (renderMode) {
  case RM_DRAW_SCENE:
    drawScene();
    break;
  case RM_SHOW_NORMAL_MAP:
    showTexture(TO_NORMAL_MAP, 0);
    break;
  case RM_SHOW_DECAL:
    showTexture(TO_DECAL_GLOSS, 0);
    break;
  case RM_SHOW_GLOSS:
    showTexture(TO_DECAL_GLOSS, 1);
    break;
  default:
    assert(!"bogus renderMode");
    break;
  }

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

int lineIsComment(const char *line)
{
  if (line[0] == '#') {
    /* OpenGL assembly style comment. */
    return 1;
  }
  if (line[0] == '/' && line[1] == '/') {
    /* DirectX assembly style comment. */
    return 1;
  }
  return 0;
}

/* Print compiled programs
*/
static void printProgram(int indent, const char *string, int printComments)
{
  const char *line, *c;

  line = string;
  while (*line != '\0') {
    size_t len;

    c = line;
    while (*c!='\n' && *c!='\0') {
      c++;
    }
    len = c-line;
    if (printComments || !lineIsComment(line)) {
      int i;

      for (i=0; i<indent; i++) {
        putchar(' ');
      }
      fwrite(line, 1, len, stdout);
      putchar('\n');
    }
    if (*c == '\n') {
      c++;
    }
    line = c;
  }
}

static void dumpPrograms(CGtechnique technique, int printComments)
{
  CGpass pass = cgGetFirstPass(technique);
  int passNum = 0;

  printf(">> programs dump, technique handle=%p\n",
    technique);
  while (pass) {
    CGstateassignment sa = cgGetFirstStateAssignment(pass);
    int programNum = 0;
    int needsPassHeader = 1;

    passNum++;
    while (sa) {
      CGprogram program;

      program = cgGetProgramStateAssignmentValue(sa);
      if (program) {
        const char *compiledProgram =
          cgGetProgramString(program, CG_COMPILED_PROGRAM);

        programNum++;
        if (compiledProgram) {
          if (needsPassHeader) {
            printf(">>>> pass #%d, handle=%p\n", passNum, pass);
            needsPassHeader = 0;
          }
          printf(">>>>>> program #%d.%d, handle=%p\n",
            passNum, programNum, program);
          printProgram(2, compiledProgram, printComments);
        }
      }

      sa = cgGetNextStateAssignment(sa);
    }
    pass = cgGetNextPass(pass);
  }
  printf(">> end programs dump\n");
}

static void keyboard(unsigned char c, int x, int y)
{
  CGtechnique technique, prev_technique;
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
  case 'D':
    dumpPrograms(myCgTechnique, /*printComments*/1);
    return;
  case 'd':
    dumpPrograms(myCgTechnique, /*printComments*/0);
    return;
  case 'r':
    renderMode = (renderMode+1) % RM_NUM_MODES;
    break;
  case 'f':
    toggleFPS();
    break;
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
    if (c == '0') {
      /* Make '0' behave like 10. */
      c = '0' + 10;
    }
    c -= '1';
    technique = cgGetFirstTechnique(myCgEffect);
    while (c > 0) {
      technique = cgGetNextTechnique(technique);
      c--;
    }
    if (technique) {
      if (cgValidateTechnique(technique)) {
        myCgTechnique = technique;
      } else {
        printf("%s: technique %s not valid\n", myProgramName, cgGetTechniqueName(technique));
      }
    }
    break;
  case 'z':
  case 13:
    /* Forward one technique. */
    do {
      technique = cgGetNextTechnique(myCgTechnique);
      if (!technique) {
        technique = cgGetFirstTechnique(myCgEffect);
      }
    } while(!cgValidateTechnique(technique));
    myCgTechnique = technique;
    break;
  case 'a':
    /* Back one technique. */
    technique = myCgTechnique;
    do {
      prev_technique = technique;
      do {
        technique = cgGetNextTechnique(technique);
        if (!technique) {
          technique = cgGetFirstTechnique(myCgEffect);
        }
      } while(!cgValidateTechnique(technique));
    } while(technique != myCgTechnique);
    myCgTechnique = prev_technique;
    break;
  case 'w':
    wireframe = !wireframe;
    if (wireframe) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    break;
  case 'b':
    showBackdrop = !showBackdrop;
    break;
  case 'v':
    enableSync = !enableSync;
    requestSynchronizedSwapBuffers(enableSync);
    break;
  case 's':
    /* Colorize stencil for a single frame. */
    colorizeStencil();
    glutSwapBuffers();
    return;
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

static void techniqueMenu(int item)
{
  CGtechnique technique = cgGetFirstTechnique(myCgEffect);

  while (item > 0 || !technique) {
    technique = cgGetNextTechnique(technique);
    item--;
  }
  myCgTechnique = technique;
  glutPostRedisplay();
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

/* Used for a visualization mode to help see how the stencil
   values are left after stenciled shadow volume rendering. */
void
colorizeStencil(void)
{
  int i;
  static GLfloat colors[][3] = {
    { 1,     0,    1 }, /* 0 */

    { 1,     0,    0 }, /* 1 */
    { 0.1,   0,    0 }, /* 2 */
    { 0.2,   0,    0 }, /* 3 */
    { 0.3,   0,    0 }, /* 4 */
    { 0.4,   0,    0 }, /* 5 */
    { 0.5,   0, 0, }, /* 6 */
    { 0.6,   0,    0 }, /* 7 */
    { 0.7,   0,    0 }, /* 8 */
    { 0.8,   0,    0 }, /* 9 */
    { 0.0,   0.9,    0 }, /* 10 */

    { 0,    0, 1.0   }, /* 11 */
    { 0,    0, 0.9 }, /* 12 */
    { 0,    0, 0.8 }, /* 13 */
    { 0,    0, 0.7 }, /* 14 */
    { 0,    0, 0.6 }, /* 15 */
    { 0,    0, 0.5 }, /* 16 */
    { 0,    0, 0.4 }, /* 17 */
    { 0,    0, 0.3 }, /* 18 */
    { 0,    0, 0.2 }, /* 19 */
    { 0,    0, 0.1 }, /* 20 */
  };
  const int numColors = sizeof(colors)/sizeof(colors[0]);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glDisable(GL_CULL_FACE);
  glDisable(GL_TEXTURE_2D);
  glEnable(GL_STENCIL_TEST);
  glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
  glStencilMask(~0);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
      glLoadIdentity();

      /* Colorize stencil values from [0..numColors) with
         the colors in the colors array. */
      for (i=0; i<numColors; i++) {
        glStencilFunc(GL_EQUAL, i, ~0);
        glColor3fv(colors[i]);
        glTexCoord1f(i * (1.0/255.0) * 9);
        glRectf(-1,-1,1,1);
      }
      /* For the remaining stencil values, color them white. */
      glColor3f(1,0,0);
      glStencilFunc(GL_LEQUAL, i, ~0);
      glRectf(-1,-1,1,1);

    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glDisable(GL_STENCIL_TEST);
  glColor3f(1,1,1);
}
