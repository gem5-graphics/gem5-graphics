
/* 06_vertex_twisting.c - OpenGL-based example using a Cg
   vertex and a Cg fragment programs from Chapter 3 of "The Cg Tutorial"
   (Addison-Wesley, ISBN 0321194969). */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   1.0 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sin and cos */

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
static CGparameter myCgVertexParam_twisting;

static const char *myProgramName = "06_vertex_twisting",
                  *myVertexProgramFileName = "C3E4v_twist.cg",
/* Page 79 */     *myVertexProgramName = "C3E4v_twist",
                  *myFragmentProgramFileName = "C2E2f_passthru.cg",
/* Page 53 */     *myFragmentProgramName = "C2E2f_passthru";

static float myTwisting = 2.9, /* Twisting angle in radians. */
             myTwistDirection = 0.1; /* Animation delta for twist. */

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
  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_1) {
    fprintf(stderr, "%s: failed to initialize GLEW, OpenGL 1.1 required.\n", myProgramName);    
    exit(1);
  }

  requestSynchronizedSwapBuffers();
  glClearColor(1, 1, 1, 1);  /* White background */

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

  myCgVertexParam_twisting =
    cgGetNamedParameter(myCgVertexProgram, "twisting");
  checkForCgError("could not get twisting parameter");

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

  /* No uniform fragment program parameters expected. */

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAddMenuEntry("[w] Wireframe", 'w');
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

/* Apply an inefficient but simple-to-implement subdivision scheme for a triangle. */
static void triangleDivide(int depth,
                           const float a[2], const float b[2], const float c[2],
                           const float ca[3], const float cb[3], const float cc[3])
{
  if (depth == 0) {
    glColor3fv(ca);
    glVertex2fv(a);
    glColor3fv(cb);
    glVertex2fv(b);
    glColor3fv(cc);
    glVertex2fv(c);
  } else {
    const float d[2] = { (a[0]+b[0])/2, (a[1]+b[1])/2 },
                e[2] = { (b[0]+c[0])/2, (b[1]+c[1])/2 },
                f[2] = { (c[0]+a[0])/2, (c[1]+a[1])/2 };
    const float cd[3] = { (ca[0]+cb[0])/2, (ca[1]+cb[1])/2, (ca[2]+cb[2])/2 },
                ce[3] = { (cb[0]+cc[0])/2, (cb[1]+cc[1])/2, (cb[2]+cc[2])/2 },
                cf[3] = { (cc[0]+ca[0])/2, (cc[1]+ca[1])/2, (cc[2]+ca[2])/2 };

    depth -= 1;
    triangleDivide(depth, a, d, f, ca, cd, cf);
    triangleDivide(depth, d, b, e, cd, cb, ce);
    triangleDivide(depth, f, e, c, cf, ce, cc);
    triangleDivide(depth, d, e, f, cd, ce, cf);
  }
}

/* Large vertex displacements such as are possible with C3E4v_twist
   require a high degree of tessellation.  This routine draws a
   triangle recursively subdivided to provide sufficient tessellation. */
static void drawSubDividedTriangle(int subdivisions)
{
  const float a[2] = { -0.8, 0.8 },
              b[2] = {  0.8, 0.8 },
              c[2] = {  0.0, -0.8 },
              ca[3] = { 0, 0, 1 },
              cb[3] = { 0, 0, 1 },
              cc[3] = { 0.7, 0.7, 1 };

  glBegin(GL_TRIANGLES);
    triangleDivide(subdivisions, a, b, c, ca, cb, cc);
  glEnd();
}

static void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  cgSetParameter1f(myCgVertexParam_twisting, myTwisting);

  cgGLBindProgram(myCgVertexProgram);
  checkForCgError("binding vertex program");

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  cgGLBindProgram(myCgFragmentProgram);
  checkForCgError("binding fragment program");

  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  drawSubDividedTriangle(5);

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  glutSwapBuffers();
}

static void idle(void)
{
  if (myTwisting > 3) {
    myTwistDirection = -0.05;
  } else if (myTwisting < -3) {
    myTwistDirection = 0.05;
  }
  myTwisting += myTwistDirection;
  glutPostRedisplay();
}

static void keyboard(unsigned char c, int x, int y)
{
  static int animating = 0,
             wireframe = 0;

  switch (c) {
  case ' ':
    animating = !animating; /* Toggle */
    if (animating) {
      glutIdleFunc(idle);
    } else {
      glutIdleFunc(NULL);
    }    
    break;
  case 'w':
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
    cgDestroyProgram(myCgVertexProgram);
    cgDestroyProgram(myCgFragmentProgram);
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
