
/* 15_particle_system.c - OpenGL-based example implementing a particle
   system in a vertex shader from Chapter 6 of "The Cg Tutorial"
   (Addison-Wesley, ISBN 0321194969). */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   1.4 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit and rand */
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
static CGparameter myCgVertexParam_globalTime;
static CGparameter myCgVertexParam_acceleration;
static CGparameter myCgVertexParam_modelViewProj;

static const char *myProgramName             = "15_particle_system",
                  *myVertexProgramFileName   = "C6E2v_particle.cg",
/* Page 152 */    *myVertexProgramName       = "C6E2v_particle",
                  *myFragmentProgramFileName = "C6E2v_particle.cg",
                  *myFragmentProgramName     = "texcoord2color";

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
static void visibility(int state);
static void keyboard(unsigned char c, int x, int y);
static void menu(int item);
static void requestSynchronizedSwapBuffers(void);

/* Forward declaration to initialize particle system. */
static void resetParticles(void);

int main(int argc, char **argv)
{
  resetParticles();

  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutVisibilityFunc(visibility);
  glutKeyboardFunc(keyboard);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_3) {
    fprintf(stderr, "%s: failed to initialize GLEW, OpenGL 1.3 required.\n", myProgramName);    
    exit(1);
  }

  requestSynchronizedSwapBuffers();
  glClearColor(0.2, 0.6, 1.0, 1);  /* Blue background */

  /* Configure for rendering smooth points */
  glPointSize(6.0f);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);

  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgGLSetDebugMode(CG_FALSE);

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

  myCgVertexParam_globalTime =
    cgGetNamedParameter(myCgVertexProgram, "globalTime");
  checkForCgError("could not get globalTime parameter");

  myCgVertexParam_acceleration =
    cgGetNamedParameter(myCgVertexProgram, "acceleration");
  checkForCgError("could not get acceleration parameter");

  myCgVertexParam_modelViewProj =
    cgGetNamedParameter(myCgVertexProgram, "modelViewProj");
  checkForCgError("could not get modelViewProj parameter");

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

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAddMenuEntry("[p] Toggle point size computation", 'p');
  glutAddMenuEntry("[r] Reset particles", 'r');
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

static int myAnimating = 1;
static int myVerbose = 0;
static float myGlobalTime = 0.0;
static int myPass = 0;

/* State for particles. */
typedef struct Particle_t {
  float pInitial[3];
  float vInitial[3];
  float tInitial;
  int alive;
} Particle;
#define NUM_PARTICLES 800
static Particle myParticleSystem[NUM_PARTICLES]; /* List of particle start times. */

/* Random number generator. */
static float float_rand(void) { return rand() / (float) RAND_MAX; }
#define RANDOM_RANGE(lo, hi) ((lo) + (hi - lo) * float_rand())

static void resetParticles(void)
{
  int i;

  myGlobalTime = 0.0;
  myPass = 0;

  /* Particles will start at random times to gradually get things rolling. */
  for(i = 0; i<NUM_PARTICLES; i++) {
    float radius = 0.25f;
    float initialElevation = -0.5f;

    myParticleSystem[i].pInitial[0] = radius * cos(i * 0.5f);
    myParticleSystem[i].pInitial[1] = initialElevation;
    myParticleSystem[i].pInitial[2] = radius * sin(i * 0.5f);
    myParticleSystem[i].alive = 0;
    myParticleSystem[i].tInitial = RANDOM_RANGE(0,10);
  }
}

static void advanceParticles(void)
{
  float death_time = myGlobalTime - 1.0;
  int i;

  myPass++;
  for(i=0; i<NUM_PARTICLES; i++) {
    /* Birth new particles. */
    if (!myParticleSystem[i].alive &&
        (myParticleSystem[i].tInitial <= myGlobalTime)) {
      myParticleSystem[i].vInitial[0] = RANDOM_RANGE(-1,1);
      myParticleSystem[i].vInitial[1] = RANDOM_RANGE(0,6);
      myParticleSystem[i].vInitial[2] = RANDOM_RANGE(-0.5,0.5);
      myParticleSystem[i].tInitial = myGlobalTime;
      myParticleSystem[i].alive = 1;
      if (myVerbose) {
        printf("Birth %d (%f,%f,%f) at %f\n", i,
          myParticleSystem[i].vInitial[0],
          myParticleSystem[i].vInitial[1],
          myParticleSystem[i].vInitial[2], myGlobalTime);
      }
    }

    /* Kill old particles.  A particle expires in this system when it 
       is 20 passes old. */
    if (myParticleSystem[i].alive
        && (myParticleSystem[i].tInitial <= death_time)) {
      myParticleSystem[i].alive = 0;
      myParticleSystem[i].tInitial = myGlobalTime + .01; /* Rebirth next pass */
      if (myVerbose) {
        printf("Death %d at %f\n", i, myGlobalTime);
      }
    }
  }
}

static void display(void)
{
  const float acceleration = -9.8;  /* Gravity: what comes up, goes down. */
  const float viewAngle = myGlobalTime * 2.8;
  int i;

  /* Clears color (but not depth) buffer. */
  glClear(GL_COLOR_BUFFER_BIT);

  glLoadIdentity();
  gluLookAt(cos(viewAngle), 0.3, sin(viewAngle), /* Rotate eye around Y axis */
            0, 0, 0,                             /* Look at origin */
            0, 1, 0);                            /* +Y axis is up */

  /* Set uniforms before glGLProgram bind. */
  cgSetParameter1f(myCgVertexParam_globalTime, myGlobalTime);
  cgSetParameter4f(myCgVertexParam_acceleration, 0, acceleration, 0, 0);
  cgGLSetStateMatrixParameter(myCgVertexParam_modelViewProj,
    CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY);

  cgGLBindProgram(myCgVertexProgram);
  checkForCgError("binding vertex program");

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  cgGLBindProgram(myCgFragmentProgram);
  checkForCgError("binding fragment program");

  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  if (myVerbose) {
    printf("Pass %d\n", myPass);
  }

  /* Render live particles. */
  glBegin(GL_POINTS);
  for(i=0; i<NUM_PARTICLES; i++) {
    if (myParticleSystem[i].alive) {
      /* initial velocity */
      glTexCoord3fv(myParticleSystem[i].vInitial);
      /* initial time */
      glMultiTexCoord1f(GL_TEXTURE1, myParticleSystem[i].tInitial);
      /* initial position */
      glVertex3fv(myParticleSystem[i].pInitial);
      if (myVerbose) {
        printf("Drew %d (%f,%f,%f) at %f\n", i,
          myParticleSystem[i].vInitial[0],
          myParticleSystem[i].vInitial[1],
          myParticleSystem[i].vInitial[2], myGlobalTime);
      }
    }
  }
  glEnd();

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  glutSwapBuffers();
}

static void idle(void)
{
  if (myAnimating) {
    myGlobalTime += .01;
    advanceParticles();
  }
  glutPostRedisplay();
}

static void visibility(int state)
{
  if (state == GLUT_VISIBLE && myAnimating) {
    glutIdleFunc(idle);
  } else {
    glutIdleFunc(NULL);
  }
}

static void keyboard(unsigned char c, int x, int y)
{
  static int useComputedPointSize = 0;

  switch (c) {
  case 27:  /* Esc key */
    /* Demonstrate proper deallocation of Cg runtime data structures.
       Not strictly necessary if we are simply going to exit. */
    cgDestroyProgram(myCgVertexProgram);
    cgDestroyProgram(myCgFragmentProgram);
    cgDestroyContext(myCgContext);
    exit(0);
    break;
  case ' ':
    myAnimating = !myAnimating; /* Toggle */
    if (myAnimating) {
      glutIdleFunc(idle);
    } else {
      glutIdleFunc(NULL);
    }    
    break;
  case 'p':
    useComputedPointSize = !useComputedPointSize;
    if (useComputedPointSize) {
      glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    } else {
      glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    }
    glutPostRedisplay();
    break;
  case 'r':
    resetParticles();
    glutPostRedisplay();
    break;
  case 'v':
    myVerbose = !myVerbose;
    glutPostRedisplay();
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
