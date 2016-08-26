
/* gs_simple.c - OpenGL-based introductory geometry shader example
   using a pass-through geometry shader to draw a pattern of colored
   stars. */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   2.0 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sin and cos */
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgGeometryProfile,
                   myCgFragmentProfile;
static CGprogram   myCgVertexProgram = 0,
                   myCgGeometryProgram = 0,
                   myCgFragmentProgram = 0,
                   myCgCombinedProgram = 0;  /* For combined GLSL program */

static const char *myProgramName = "gs_simple",

                  *myVertexProgramFileName = "gs_simple.cg",
                  *myVertexProgramName = "vertex_passthru",

                  *myGeometryProgramFileName = "gs_simple.cg",
                  *myGeometryProgramName = "geometry_passthru",

                  *myFragmentProgramFileName = "gs_simple.cg",
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

/* Forward declared GLUT callbacks registered by main. */
static void display(void);
static void keyboard(unsigned char c, int x, int y);

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

  myCgGeometryProfile = cgGLGetLatestProfile(CG_GL_GEOMETRY);
  if (myCgGeometryProfile == CG_PROFILE_UNKNOWN) {
    if ( cgGLIsProfileSupported(CG_PROFILE_GLSLG) )
      myCgGeometryProfile = CG_PROFILE_GLSLG;
    else {
      fprintf(stderr, "%s: geometry profile is not available.\n", myProgramName);
      exit(0);
    }
  }
  cgGLSetOptimalOptions(myCgGeometryProfile);
  checkForCgError("selecting geometry profile");

  myCgGeometryProgram =
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myGeometryProgramFileName,  /* Name of file containing program */
      myCgGeometryProfile,        /* Profile: OpenGL ARB geometry program */
      myGeometryProgramName,      /* Entry function name */
      NULL);                      /* No extra compiler options */
  checkForCgError("creating geometry program from file");

  myCgVertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
  if (myCgGeometryProfile == CG_PROFILE_GLSLG) {
    myCgVertexProfile = CG_PROFILE_GLSLV;
  }
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

  myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  if (myCgGeometryProfile == CG_PROFILE_GLSLG) {
    myCgFragmentProfile = CG_PROFILE_GLSLF;
  }
  cgGLSetOptimalOptions(myCgFragmentProfile);
  checkForCgError("selecting fragment profile");

  myCgFragmentProgram =
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myFragmentProgramFileName,  /* Name of file containing program */
      myCgFragmentProfile,        /* Profile: OpenGL ARB fragment program */
      myFragmentProgramName,      /* Entry function name */
      NULL);                      /* No extra compiler options */
  checkForCgError("creating fragment program from file");

  if
  (
    myCgVertexProfile==CG_PROFILE_GLSLV &&
    myCgGeometryProfile==CG_PROFILE_GLSLG &&
    myCgFragmentProfile==CG_PROFILE_GLSLF
  )
  {
    /* Combine programs for GLSL... */

    myCgCombinedProgram = cgCombinePrograms3(myCgVertexProgram,myCgGeometryProgram,myCgFragmentProgram);
    checkForCgError("combining programs");
    cgGLLoadProgram(myCgCombinedProgram);
    checkForCgError("loading combined program");
  }
  else
  {
    /* ...otherwise load programs orthogonally */

    cgGLLoadProgram(myCgVertexProgram);
    checkForCgError("loading vertex program");
    cgGLLoadProgram(myCgGeometryProgram);
    checkForCgError("loading geometry program");
    cgGLLoadProgram(myCgFragmentProgram);
    checkForCgError("loading fragment program");
  }

  glutMainLoop();
  return 0;
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

static void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  cgGLEnableProfile(myCgGeometryProfile);
  checkForCgError("enabling geometry profile");

  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  if (myCgCombinedProgram)
  {
    cgGLBindProgram(myCgCombinedProgram);
    checkForCgError("binding combined program");
  }
  else
  {
    cgGLBindProgram(myCgVertexProgram);
    checkForCgError("binding vertex program");

    cgGLBindProgram(myCgGeometryProgram);
    checkForCgError("binding geometry program");

    cgGLBindProgram(myCgFragmentProgram);
    checkForCgError("binding fragment program");
  }

  drawStars();

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgGeometryProfile);
  checkForCgError("disabling geometry profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  glutSwapBuffers();
}

static void keyboard(unsigned char c, int x, int y)
{
  switch (c) {
  case 27:  /* Esc key */
    /* Demonstrate proper deallocation of Cg runtime data structures.
       Not strictly necessary if we are simply going to exit. */
    cgDestroyProgram(myCgVertexProgram);
    cgDestroyProgram(myCgGeometryProgram);
    cgDestroyProgram(myCgFragmentProgram);
    if (myCgCombinedProgram)
      cgDestroyProgram(myCgCombinedProgram);
    cgDestroyContext(myCgContext);
    exit(0);
    break;
  }
}
