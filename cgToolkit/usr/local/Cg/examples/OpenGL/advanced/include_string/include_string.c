/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   2.1 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <string.h>   /* for strcmp */
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed? */
#include <Cg/cgGL.h>

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile;
static CGprogram   myCgVertexProgram;

static const char *myProgramName = "inclusion";

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

static const char *cgArg[] = { "-I", "shader", NULL };

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
  checkForCgError("creating context");

  cgGLSetDebugMode(CG_FALSE);
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);

  myCgVertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
  cgGLSetOptimalOptions(myCgVertexProfile);
  checkForCgError("selecting vertex profile");

  cgSetCompilerIncludeString(myCgContext,"shader/output.cg",
        "struct Output {                                \n"
        "  float4 position : POSITION;                  \n"
        "  float3 color    : COLOR;                     \n"
        "};                                             \n");

  cgSetCompilerIncludeString(myCgContext,"shader/vertexProgram.cg",
      "#include \"output.cg\"                           \n"
      "                                                 \n"
      "Output vertexProgram(float2 position : POSITION) \n"
      "{                                                \n"
      "  Output OUT;                                    \n"
      "                                                 \n"
      "  OUT.position = float4(position,0,1);           \n"
      "  OUT.color = float3(0,1,0);                     \n"
      "                                                 \n"
      "  return OUT;                                    \n"
      "}                                                \n");

  myCgVertexProgram =
    cgCreateProgram(
      myCgContext,               /* Cg runtime context */
      CG_SOURCE,                 /* Program in human-readable form */
     "#include \"vertexProgram.cg\"\n",
      myCgVertexProfile,         /* Profile: OpenGL ARB vertex program */
      "vertexProgram",           /* Entry function name */
      cgArg);                    /* Include path options */
  checkForCgError("creating vertex program from file");
  cgGLLoadProgram(myCgVertexProgram);
  checkForCgError("loading vertex program");

  glutMainLoop();
  return 0;
}

static void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  cgGLBindProgram(myCgVertexProgram);
  checkForCgError("binding vertex program");

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  /* Rendering code verbatim from Chapter 1, Section 2.4.1 "Rendering
     a Triangle with OpenGL" (page 57). */
  glBegin(GL_TRIANGLES);
    glVertex2f(-0.8, 0.8);
    glVertex2f(0.8, 0.8);
    glVertex2f(0.0, -0.8);
  glEnd();

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  glutSwapBuffers();
}

static void keyboard(unsigned char c, int x, int y)
{
  switch (c) {
  case 27:  /* Esc key */
    /* Demonstrate proper deallocation of Cg runtime data structures.
       Not strictly necessary if we are simply going to exit. */
    cgDestroyProgram(myCgVertexProgram);
    cgDestroyContext(myCgContext);
    exit(0);
    break;
  }
}
