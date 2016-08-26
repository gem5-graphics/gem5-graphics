
/* interpolation_modifier.c - interpolation modifiers (FLAT,
                              NOPERSPECTIVE, and CENTROID) */

/* This example renders the same triangle into four different viewport.
   Each viewport uses a different technique that passes through the an
   interpolated texture coordinate set as the fragment color but with
   different interpolation modifiers.  */

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include <GL/glew.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#ifdef _WIN32
#include <windows.h>  /* for QueryPerformanceCounter */
#endif

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed? */
#include <Cg/cgGL.h>  /* 3D API specific Cg runtime API for OpenGL */

static CGcontext   myCgContext;
static CGeffect    myCgEffect;

static const char *myProgramName = "interpolation_modifier",
                  *myEffectFileName = "interpolation_modifier.cgfx";

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
static void display(void);
static void keyboard(unsigned char c, int x, int y);

int main(int argc, char **argv)
{
  CGtechnique technique;
  int validTechniques = 0;

  glutInitWindowSize(600, 600);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);

  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_1) {
    fprintf(stderr, "%s: Failed to initialize GLEW. OpenGL 1.1 required.\n", myProgramName);    
    exit(1);
  }

  glClearColor(0.2, 0.2, 0.2, 0.0);  /* Gray background */

  myCgContext = cgCreateContext();
  checkForCgError("creating context");

  cgGLRegisterStates(myCgContext);
  checkForCgError("registering standard CgFX states");
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
  checkForCgError("manage texture parameters");

  myCgEffect = cgCreateEffectFromFile( myCgContext, myEffectFileName, NULL );
  checkForCgError("creating interpolation_modifier.cgfx effect");
  assert(myCgEffect);

  technique = cgGetFirstTechnique(myCgEffect);
  while (technique) {
    if (cgValidateTechnique(technique) == CG_FALSE) {
      fprintf(stderr, "%s: Technique %s did not validate.  Skipping.\n",
        myProgramName, cgGetTechniqueName(technique));
    } else {
      validTechniques++;
    }
    technique = cgGetNextTechnique(technique);
  }
  if (0 == validTechniques) {
    fprintf(stderr, "%s: No valid techniques\n",
      myProgramName);
    if (glutExtensionSupported("GL_NV_gpu_program4")) {
      exit(1);
    } else {
      fprintf(stderr, "%s: because your OpenGL implementation lacks NV_gpu_program extension\n",
        myProgramName);
      exit(0);
    }
  }

  glutMainLoop();
  return 0;
}

static int myWindowWidth,
           myWindowHeight;

static void reshape(int width, int height)
{
  myWindowWidth = width;
  myWindowHeight = height;
}

static void drawTriangle(void)
{
  glBegin(GL_TRIANGLES);
    glColor3f(1,0,0);
    glTexCoord3f(1,0,0);
    glVertex2f(-0.7, 0.7);
    glColor3f(0,1,0);
    glTexCoord3f(0,1,0);
    glVertex2f(0.7, 0.7);
    glColor3f(0,0,1);
    glTexCoord3f(0,0,1);
    glVertex4f(0.0, -0.7*5, 0, 5);  /* give perspective */
  glEnd();
}

static void output(float x, float y, const char *string)
{
  int len, i;

  glRasterPos2f(x, y);
  len = (int) strlen(string);
  for (i = 0; i < len; i++) {
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
  }
}

static void drawTriangleWithEffect(int x, int y, const char *techniqueName)
{
  CGtechnique technique = cgGetNamedTechnique(myCgEffect, techniqueName);
  CGbool isValid = cgValidateTechnique(technique);
  int viewWidth = myWindowWidth/2,
      viewHeight = myWindowHeight/2;

  glViewport(x*viewWidth, y*viewHeight, viewWidth, viewHeight);
  if (isValid) {
    CGpass pass = cgGetFirstPass(technique);

    while (pass) {
      cgSetPassState(pass);
    
      drawTriangle();

      cgResetPassState(pass);
      pass = cgGetNextPass(pass);
    }
  } else {
    glColor3f(0.8, 0.1, 0.1);
    output(-0.95, 0, "technique not valid");
  }
  glColor3f(0.8, 0.8, 0.8);
  output(-0.95, -0.8, techniqueName);
}

static const char *myCgTechniqueNames[4] = {
  "NormalColorInterpolation",
  "FlatColorInterpolation",
  "NoPerspectiveColorInterpolation",
  "CentroidColorInterpolation"
};

static void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  /* Lower-left */
  drawTriangleWithEffect(0, 0, myCgTechniqueNames[0]);
  /* Lower-right */
  drawTriangleWithEffect(1, 0, myCgTechniqueNames[1]);
  /* Upper-left */
  drawTriangleWithEffect(0, 1, myCgTechniqueNames[2]);
  /* Upper-right */
  drawTriangleWithEffect(1, 1, myCgTechniqueNames[3]);

  glutSwapBuffers();
}

void dumpShader(int ndx)
{
  const char *techniqueName = myCgTechniqueNames[ndx];
  CGtechnique technique = cgGetNamedTechnique(myCgEffect, techniqueName);
  CGbool isValid = cgValidateTechnique(technique);

  if (isValid) {
    CGpass pass = cgGetFirstPass(technique);

    /* Iterate over all passes... */
    while (pass) {
      CGstateassignment sa = cgGetNamedStateAssignment(pass, "FragmentProgram");
      CGprogram program = cgGetProgramStateAssignmentValue(sa);
      const char *assembly = cgGetProgramString(program, CG_COMPILED_PROGRAM);

      printf("compiled assembly for %s:\n%s\n", techniqueName, assembly);
      pass = cgGetNextPass(pass);
    }
  }
}

static void keyboard(unsigned char c, int x, int y)
{
  switch (c) {
  case 27:  /* Esc key */
    /* Demonstrate proper deallocation of Cg runtime data structures.
       Not strictly necessary if we are simply going to exit. */
    cgDestroyEffect(myCgEffect);
    cgDestroyContext(myCgContext);
    exit(0);
    break;
  case '1':
  case '2':
  case '3':
  case '4':
    dumpShader(c - '1');
    break;
  }
}
