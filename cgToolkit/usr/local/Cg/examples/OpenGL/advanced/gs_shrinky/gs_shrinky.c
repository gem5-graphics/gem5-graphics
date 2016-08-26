
/* gs_shrinky.c - OpenGL-based introductory geometry shader example
   using a pass-through geometry shader to draw a pattern of colored
   stars whose triangles shrink and expand. */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   2.0 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sin and cos */
#include <assert.h>
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>

static CGcontext   myCgContext;
static CGprogram   myCgComboProgram;
static CGparameter myCgParam_shrinkFactor;

static const char *myProgramName = "gs_shrinky",

                  *myVertexProgramFileName = "gs_shrinky.cg",
                  *myVertexProgramName = "vertex_passthru",

                  *myGeometryProgramFileName = "gs_shrinky.cg",
                  *myGeometryProgramName = "triangle_shrinky",

                  *myFragmentProgramFileName = "gs_shrinky.cg",
                  *myFragmentProgramName = "fragment_passthru";

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

/* Forward declared GLUT callbacks registered by main and other functions. */
static void display(void);
static void keyboard(unsigned char c, int x, int y);
static void loadShrinkyShader(void);
static void menu(int item);

int main(int argc, char **argv)
{
  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);

  glClearColor(0.1, 0.3, 0.6, 0.0);  /* Blue background */

  myCgContext = cgCreateContext();
  cgGLSetDebugMode( CG_FALSE );
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);
  checkForCgError("creating context");

  loadShrinkyShader();

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

void loadShrinkyShader(void)
{
  CGprofile profile;
  CGprogram vertexProgram,
            geometryProgram,
            fragmentProgram;

  profile = cgGLGetLatestProfile(CG_GL_VERTEX);
  cgGLSetOptimalOptions(profile);
  checkForCgError("selecting vertex profile");

  vertexProgram =
    cgCreateProgramFromFile(
      myCgContext,              /* Cg runtime context */
      CG_SOURCE,                /* Program in human-readable form */
      myVertexProgramFileName,  /* Name of file containing program */
      profile,                  /* Profile: OpenGL ARB vertex program */
      myVertexProgramName,      /* Entry function name */
      NULL);                    /* No extra compiler options */
  checkForCgError("creating vertex program from file");
  cgGLLoadProgram(vertexProgram);
  checkForCgError("loading vertex program");

  profile = cgGLGetLatestProfile(CG_GL_GEOMETRY);
  if (profile == CG_PROFILE_UNKNOWN) {
    fprintf(stderr, "%s: geometry profile is not available.\n", myProgramName);
    exit(0);
  }
  cgGLSetOptimalOptions(profile);
  checkForCgError("selecting geometry profile");

  geometryProgram =
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myGeometryProgramFileName,  /* Name of file containing program */
      profile,                    /* Profile: OpenGL ARB geometry program */
      myGeometryProgramName,      /* Entry function name */
      NULL);                      /* No extra compiler options */
  checkForCgError("creating geometry program from file");
  if (profile!=CG_PROFILE_GLSLG)  /* A glslg shader can't be loaded by itself */
  {
    cgGLLoadProgram(geometryProgram);
    checkForCgError("loading geometry program");
  }

  profile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  cgGLSetOptimalOptions(profile);
  checkForCgError("selecting fragment profile");

  fragmentProgram =
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myFragmentProgramFileName,  /* Name of file containing program */
      profile,                    /* Profile: OpenGL ARB fragment program */
      myFragmentProgramName,      /* Entry function name */
      NULL);                      /* No extra compiler options */
  checkForCgError("creating fragment program from file");
  cgGLLoadProgram(fragmentProgram);
  checkForCgError("loading fragment program");

  myCgComboProgram = cgCombinePrograms3(vertexProgram,
                                        geometryProgram,
                                        fragmentProgram);
  checkForCgError("combining programs");
  assert(3 == cgGetNumProgramDomains(myCgComboProgram));

  cgDestroyProgram(vertexProgram);
  cgDestroyProgram(geometryProgram);
  cgDestroyProgram(fragmentProgram);
  checkForCgError("destroying original programs after combining");

  cgGLLoadProgram(myCgComboProgram);
  checkForCgError("loading combo program");

  myCgParam_shrinkFactor =
    cgGetNamedParameter(myCgComboProgram, "shrinkFactor");
  checkForCgError("could not get shrinkFactor parameter");
}

static void drawStar(float x, float y,
                     int starPoints, float R, float r,
                     GLfloat *color)
{
  int i;
  double piOverStarPoints = 3.14159 / starPoints,
         angle = 0.0;

  glBegin(GL_TRIANGLE_FAN);
    glColor3f(1,1,1);
    glVertex2f(x, y);  /* Center of star */
    glColor3fv(color);
    /* Emit exterior vertices for star's points. */
    for (i=0; i<starPoints; i++) {
      glVertex2f(x + R*cos(angle), y + R*sin(angle));
      angle += piOverStarPoints;
      glVertex2f(x + r*cos(angle), y + r*sin(angle));
      angle += piOverStarPoints;
    }
    /* End by repeating first exterior vertex of star. */
    angle = 0;
    glVertex2f(x + R*cos(angle), y + R*sin(angle));
  glEnd();
}

static void drawStars(void)
{
  static GLfloat red[]    = { 1, 0, 0 },
                 green[]  = { 0, 1, 0 },
                 blue[]   = { 0, 0, 1 },
                 cyan[]   = { 0, 1, 1 },
                 yellow[] = { 1, 1, 0 },
                 gray[]   = { 0.5f, 0.5f, 0.5f };

  /*                     star    outer   inner  */
  /*        x      y     Points  radius  radius */
  /*       =====  =====  ======  ======  ====== */
  drawStar(-0.1,   0,    5,      0.5,    0.2,  red);
  drawStar(-0.84,  0.1,  5,      0.3,    0.12, green);
  drawStar( 0.92, -0.5,  5,      0.25,   0.11, blue);
  drawStar( 0.3,   0.97, 5,      0.3,    0.1,  cyan);
  drawStar( 0.94,  0.3,  5,      0.5,    0.2,  yellow);
  drawStar(-0.97, -0.8,  5,      0.6,    0.2,  gray);
}

float myShrinkFactor = 0.2,
      myShrinkDirection = 0.02f;

static void display(void)
{
  int i;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  cgSetParameter1f(myCgParam_shrinkFactor, myShrinkFactor);

  cgGLBindProgram(myCgComboProgram);
  checkForCgError("binding combo program");

  // Enable all profiles needed
  for (i=cgGetNumProgramDomains(myCgComboProgram)-1; i>=0; i--)
    cgGLEnableProfile(cgGetProgramDomainProfile(myCgComboProgram, i));
  checkForCgError("enabling profiles");

  drawStars();

  // Disable all profiles needed
  for (i=cgGetNumProgramDomains(myCgComboProgram)-1; i>=0; i--)
    cgGLDisableProfile(cgGetProgramDomainProfile(myCgComboProgram, i));
  checkForCgError("enabling profiles");

  glutSwapBuffers();
}

static void idle(void)
{
  if (myShrinkFactor > 0.8) {
    myShrinkDirection = -0.02f;
  } else if (myShrinkFactor <= 0) {
    myShrinkFactor = 0;
    myShrinkDirection = 0.02f;
  }
  myShrinkFactor += myShrinkDirection;
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
    cgDestroyProgram(myCgComboProgram);
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
