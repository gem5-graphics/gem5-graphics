
/* cgfx_interfaces.c - an OpenGL-based CgFX 1.5 demo */

/* This CgFX demo renders a torus shaded with a sequence
   of "layers" composited together in sequence.  The layers
   are instances of implementations of a common "Layer" interface.
   The sequence of layers is implemented via an unsized array
   (sized and specified at runtime) of Layer interfaces.  An
   annotation in the effect provides an order for the layer
   instances. */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>    /* for exit */

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
#include <Cg/cgGL.h>

static const char *myProgramName = "cgfx_interfaces"; /* Program name for messages. */

/* Cg global variables */
CGcontext   myCgContext;
CGeffect    myCgEffect;
CGtechnique myCgTechnique;
CGpass      myCgPass;
CGparameter myCgEyePositionParam,
            myCgLightPositionParam,
            myCgModelViewProjParam;

/* Forward declare helper functions and callbacks registered by main. */
static void checkForCgError(const char *situation);
static void handleFPS(void);
static void display(void);
static void reshape(int width, int height);
static void keyboard(unsigned char c, int x, int y);
static void initCg();
static void initMenus();
static void initOpenGL();

int main(int argc, char **argv)
{
  myCgContext = cgCreateContext();
  checkForCgError("creating context");

  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(640, 480);
  glutInit(&argc, argv);

  glutCreateWindow("cgfx_interfaces (OpenGL)");
  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_1) {
    fprintf(stderr, "%s: Failed to initialize GLEW. OpenGL 1.1 required.\n", myProgramName);    
    exit(1);
  }

  initCg();
  initMenus();
  initOpenGL();

  glutMainLoop();
  return 0;
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
              "Error: %s\n",
              myProgramName, situation, string);
    }
    exit(1);
  }
}

/* List of annotated layer instances in effect. */
static struct {
  const char *name;
  CGparameter parameter;
} myLayerNames[20];
const int myMaxLayerNames = sizeof(myLayerNames)/sizeof(myLayerNames[0]);
int myNumLayerNames = 0;  /* Actual layers counted in effect. */
int myLayersUsed = 0;     /* Number of layers currently in use. */
int myStartLayer = 0;     /* Effect to start compositing (lower-most). */
CGparameter myCgLayerListParam;  /* Handle to layers effect parameter. */

static char *myLayerListOrder = NULL;

/* Get layer list effect parameter and its annotated ordering of layers. */
static void getLayerNamesFromParameterAnnotation(const char *paramName,
                                                 const char *annotationName)
{

  CGannotation layerListOrderAnn;
  const char *layerListOrderStr;
  char *layerName;
  int i;

  myCgLayerListParam = cgGetNamedEffectParameter(myCgEffect, paramName);
  if (!myCgLayerListParam) {
    fprintf(stderr, "%s: no parameter found named %s\n",
      myProgramName, paramName);
    exit(1);
  }
  layerListOrderAnn = cgGetNamedParameterAnnotation(myCgLayerListParam, annotationName);
  if (!layerListOrderAnn) {
    fprintf(stderr, "%s: no annotation found named %s\n",
      myProgramName, annotationName);
    exit(1);
  }
  layerListOrderStr = cgGetStringAnnotationValue(layerListOrderAnn);
  if (!layerListOrderStr) {
    fprintf(stderr, "%s: no annotation string value for %s\n",
      myProgramName, annotationName);
    exit(1);
  }

  /* Get non-const copy of the string for strtok. */
  if (myLayerListOrder) {
    /* Just in case called multiple times. */
    free(myLayerListOrder);
  }
  myLayerListOrder = strdup(layerListOrderStr);
  assert(myLayerListOrder);

  /* Parse space/comma/semi-colon separated instance names. */
  layerName = strtok(myLayerListOrder, " ,;");
  i = 0;
  while (layerName && i<myMaxLayerNames) {
    myLayerNames[i].name = layerName;
    myLayerNames[i].parameter =
      cgGetNamedEffectParameter(myCgEffect, layerName);
    if (!myLayerNames[i].parameter) {
      fprintf(stderr,
        "%s: layer parameter instance %s not an effect parameter\n",
        myProgramName, layerName);
      exit(1);
    }
    i++;
    layerName = strtok(NULL, " ,;");
  }
  myNumLayerNames = i;

  myLayersUsed = myNumLayerNames;
}

/* Configure unsized layers effect parameter. */
static void configLayerUnsizedArray(int layers, int initial)
{
  CGparameter element;
  int i, layerIndex = initial;

  assert(layers <= myNumLayerNames);
  assert(initial < myNumLayerNames);

  cgSetArraySize(myCgLayerListParam, layers);
  for (i=0; i<layers; i++) {
    element = cgGetArrayParameter(myCgLayerListParam, i);
    cgConnectParameter(myLayerNames[layerIndex].parameter, element);
    layerIndex = (layerIndex + 1) % myNumLayerNames;
  }
  if (myCgTechnique) {
    if (cgValidateTechnique(myCgTechnique) == CG_FALSE) {
      printf("%s: could not validate %s\n",
        myProgramName, cgGetTechniqueName(myCgTechnique));
    }
  }
}

static const char *myPassDirections[2] = { "forward", "reverse" };
static int myDoReverseComposite = 0;

static void initCg(void)
{
  cgGLRegisterStates(myCgContext);
  checkForCgError("registering standard CgFX states");
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
  checkForCgError("manage texture parameters");

  myCgEffect = cgCreateEffectFromFile(myCgContext, "layers.cgfx", NULL);
  checkForCgError("creating layers.cgfx effect");
  assert(myCgEffect);

  getLayerNamesFromParameterAnnotation("layers", "LayerOrderList");
  configLayerUnsizedArray(myLayersUsed, myStartLayer);

  myCgTechnique = cgGetFirstTechnique(myCgEffect);
  while (myCgTechnique && cgValidateTechnique(myCgTechnique) == CG_FALSE) {
    fprintf(stderr, "%s: Technique %s did not validate.  Skipping.\n",
      myProgramName, cgGetTechniqueName(myCgTechnique));
    myCgTechnique = cgGetNextTechnique(myCgTechnique);
  }
  if (myCgTechnique) {
    fprintf(stderr, "%s: Initially using technique %s.\n",
      myProgramName, cgGetTechniqueName(myCgTechnique));
  } else {
    fprintf(stderr, "%s: No valid technique\n",
      myProgramName);
    exit(1);
  }
  myCgPass = cgGetNamedPass(myCgTechnique, myPassDirections[myDoReverseComposite]);

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

CGtechnique validTechnique[20];
#define MAX_TECHNIQUES sizeof(validTechnique)/sizeof(validTechnique[0])

void selectTechnique(int item)
{
  CGtechnique newTechnique = validTechnique[item];
  if (cgValidateTechnique(newTechnique)) {
    myCgTechnique = newTechnique;
    myCgPass = cgGetNamedPass(myCgTechnique, myPassDirections[myDoReverseComposite]);
    glutPostRedisplay();
  } else {
    printf("%s: could not validate %s\n",
      myProgramName, cgGetTechniqueName(newTechnique));
  }
}

static int initTechniqueMenu(void)
{
  CGtechnique technique;
  int entry = 0;
  int menu = glutCreateMenu(selectTechnique);

  technique = cgGetFirstTechnique(myCgEffect);
  while (technique && entry < MAX_TECHNIQUES) {
    validTechnique[entry] = technique;
    glutAddMenuEntry(cgGetTechniqueName(technique), entry);
    entry++;
    technique = cgGetNextTechnique(technique);
  }
  return menu;
}

static void selectMainMenu(int item)
{
  keyboard((unsigned char)item, 0, 0);
}

static int initLayerOrderMenu(void)
{
  char buffer[300];
  int menu = glutCreateMenu(selectMainMenu);
  int i;

  for (i=0; i<myNumLayerNames && i<9; i++) {
    sprintf(buffer, "[%c] Start with %s", i+'1', myLayerNames[i].name);
    glutAddMenuEntry(buffer, '1'+i);
  }
  return menu;
}

static void initMenus(void)
{
  int techniqueSubmenu = initTechniqueMenu();
  int layerOrderSubmenu = initLayerOrderMenu();

  glutCreateMenu(selectMainMenu);
  glutAddMenuEntry("[ ] Toggle animation", ' ');
  glutAddMenuEntry("[W] Toggle wireframe", 'W');
  glutAddMenuEntry("[r] Reverse layer order", 'r');
  glutAddMenuEntry("[+] Increase layers", '+');
  glutAddMenuEntry("[-] Decrease layers", '-');
  glutAddSubMenu("First layer...", layerOrderSubmenu);
  glutAddSubMenu("Techniques...", techniqueSubmenu);
  glutAddMenuEntry("[f] Toggle frames/second", 'f');
  glutAddMenuEntry("Quit", 27);  /* Escape */
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}

static void initOpenGL(void)
{
  glClearColor(0.1, 0.3, 0.6, 0.0);  /* Blue background */
  glEnable(GL_DEPTH_TEST);
}

static int myWindowWidth, myWindowHeight;

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

  myWindowWidth = width;
  myWindowHeight = height;
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

static void output(int x, int y, const char *string)
{
  int len, i;

  glRasterPos2f(x, y);
  len = (int) strlen(string);
  for (i = 0; i < len; i++) {
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
  }
}

/* Overlay the technqiue's current layer ordering. */
static void outputLayers(void)
{
  int layer, i;

  glDisable(GL_FRAGMENT_PROGRAM_ARB);
  glDisable(GL_VERTEX_PROGRAM_ARB);
  glDisable(GL_DEPTH_TEST);
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
    glLoadIdentity();
    glOrtho(0, myWindowWidth, myWindowHeight, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
      glLoadIdentity();
    
      glColor3f(1,1,0);
      output(10, 20, "topmost layer");
      glColor3f(1,1,1);
      if (myDoReverseComposite) {
        layer = myStartLayer;
        for (i=1; i<=myLayersUsed; i++) {
          output(15, 20+18*i, myLayerNames[layer].name);
          layer++;
          if (layer >= myNumLayerNames) {
            layer = 0;
          }
        }
      } else {
        layer = (myStartLayer+myLayersUsed-1) % myNumLayerNames;
        for (i=1; i<=myLayersUsed; i++) {
          output(15, 20+18*i, myLayerNames[layer].name);
          layer--;
          if (layer < 0) {
            layer = myNumLayerNames-1;
          }
        }
      }
      glColor3f(1,1,0);
      output(10, 20+18*i, "bottom layer");
    glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glEnable(GL_DEPTH_TEST);
}

static void display(void)
{
  const float eyeRadius = 18.0,
              eyeElevationRange = 8.0;
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

  /* Set Cg parameters for the technique's effect. */
  cgGLSetStateMatrixParameter(myCgModelViewProjParam,
    CG_GL_MODELVIEW_PROJECTION_MATRIX,
    CG_GL_MATRIX_IDENTITY);
  cgSetParameter3fv(myCgEyePositionParam, eyePosition);
  cgSetParameter3fv(myCgLightPositionParam, myLightPosition);

  cgSetPassState(myCgPass);
  drawFlatPatch(myTorusSides, myTorusRings);
  cgResetPassState(myCgPass);

  outputLayers();

  handleFPS();
  glutSwapBuffers();
}

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

static int myDrawFPS = 1;

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
  static int wireframe = 0;

  int startLayer;

  switch (c) {
  case ' ':
    myAnimating = !myAnimating; /* Toggle */
    if (myAnimating) {
      myLastElapsedTime = glutGet(GLUT_ELAPSED_TIME);
      glutIdleFunc(idle);
    } else {
      glutIdleFunc(NULL);
    }  
    return;
  case 'f':
    myDrawFPS = !myDrawFPS;
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
    startLayer = c - '1';
    if (startLayer < myNumLayerNames) {
      myStartLayer = startLayer;
      configLayerUnsizedArray(myLayersUsed, myStartLayer);
    }
    break;
  case '+':
    if (myLayersUsed < myNumLayerNames) {
      myLayersUsed++;
      configLayerUnsizedArray(myLayersUsed, myStartLayer);
    }
    break;
  case '-':
    if (myLayersUsed > 0) {
      myLayersUsed--;
      configLayerUnsizedArray(myLayersUsed, myStartLayer);
    }
    break;
  case 'W':
    wireframe = !wireframe;
    if (wireframe) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    break;
  case 'r':
    myDoReverseComposite = !myDoReverseComposite;
    myCgPass = cgGetNamedPass(myCgTechnique, myPassDirections[myDoReverseComposite]);
    break;
  case 27:  /* Esc key */
    /* Demonstrate proper deallocation of Cg runtime data structures.
       Not strictly necessary if we are simply going to exit. */
    cgDestroyEffect(myCgEffect);
    cgDestroyContext(myCgContext);
    free(myLayerListOrder);
    exit(0);
    break;
  }
  glutPostRedisplay();
}
