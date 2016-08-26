
/* tess_simple.c - OpenGL-based introductory tessellation shader */
/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version 3.0 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sin and cos */

#include <GL/glew.h>

#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <Cg/cg.h>    /* Can't include this?  Is the Cg Toolkit installed? */
#include <Cg/cgGL.h>

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgTessellationControlProfile,
                   myCgTessellationEvaluationProfile,
                   myCgFragmentProfile;
static CGprogram   myCgVertexProgram = 0,
                   myCgTessellationControlProgram = 0,
                   myCgTessellationEvaluationProgram = 0,
                   myCgFragmentProgram = 0,
                   myCgCombinedProgram = 0;                /* For combined GLSL program */
static CGparameter myCgParameter_innerTess = 0,
                   myCgParameter_outerTess = 0,
                   myCgParameter_modelview = 0,
                   myCgParameter_projection = 0;
static const char *myProgramName       = "tess_simple",
                  *myCgProgramFileName = "tess_simple.cg",
                  *myCgVertexProgramName                 = "mainV",
                  *myCgTessellationControlProgramName    = "main",
                  *myCgTessellationEvaluationProgramName = "main",
                  *myCgFragmentProgramName               = "mainF";

/* Patch control point data */
float patchPosition[6][2] = {
  { -1, 1 }, { 0, 1 }, { 1, 1 },
  { -1,-1 }, { 0,-1 }, { 1,-1 },
};
int patchIndex[2][4] = {
  { 0,1,3,4 },
  { 1,2,4,5 }
};

/* Tessellation configuration */
int   innerTess[2]   = { 4, 4 };
float edgeTess[4]    = { 4, 4, 4, 4 };
const float edgeTessStep = 1;
const float edgeTessMin  = 1;
      float edgeTessMax  = 64;

/* Rendering configuration */
int   tessellation = 1;
float modelview[16]  = { 0.8,0,0,0, 0,0.8,0,0, 0,0,0.8,0, 0,0,0,1 }; /* 4x4 modelview matrix     */
float projection[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };       /* 4x4 projection matrix    */
const GLenum polygonModeList[] = { GL_LINE, GL_FILL, GL_NONE };      /* GL_NONE terminated array */
      GLuint polygonMode = 0;

/* Interaction configuration */
int animation = 0;
int editIndex = -1;
const float editStep = 0.02;

/* Cg version information */
const char *versionStr = NULL;
int         versionMajor = 0;
int         versionMinor = 0;

static void checkForCgError(const char *situation)
{
  CGerror error;
  const char *string = cgGetLastErrorString(&error);

  if (error != CG_NO_ERROR) {
    printf("%s: %s: %s\n", myProgramName, situation, string);
    if (error == CG_COMPILER_ERROR) {
      printf("%s\n", cgGetLastListing(myCgContext));
    }
    exit(1);
  }
}

/* Forward declared GLUT callbacks registered by main. */
static void display(void);
static void keyboard(unsigned char c, int x, int y);
static void special(int key, int x, int y);

int main(int argc, char **argv)
{
  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(special);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_5) {
    fprintf(stderr, "%s: Failed to initialize GLEW. OpenGL 1.5 required.\n", myProgramName);
    exit(1);
  }

  if (!GLEW_NV_gpu_program5 || !GLEW_ARB_tessellation_shader) {
    fprintf(stderr, "%s: NV_gpu_program5 not available.\n", myProgramName);
    tessellation = 0;
  }

  /* Cg 3.0 is required for tessellation support */
  versionStr = cgGetString(CG_VERSION);
  if (!versionStr || sscanf(versionStr, "%d.%d", &versionMajor, &versionMinor)!=2 || versionMajor<3) {
    fprintf(stderr, "%s: Cg 3.0 required, %s detected.\n", myProgramName, versionStr);
    exit(1);
  }

  glClearColor(0.7, 0.7, 0.9, 0.0);  /* Light blue background */

  myCgContext = cgCreateContext();
  cgGLSetDebugMode(CG_FALSE);
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);
  checkForCgError("creating context");

  myCgVertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
  cgGLSetOptimalOptions(myCgVertexProfile);
  checkForCgError("selecting vertex profile");

  myCgVertexProgram =
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myCgProgramFileName,        /* Name of file containing program */
      myCgVertexProfile,          /* Vertex profile */
      myCgVertexProgramName,      /* Entry function name */
      NULL);                      /* No extra compiler options */
  checkForCgError("creating vertex program from file");

  if (tessellation) {
    myCgTessellationControlProfile = cgGLGetLatestProfile(CG_GL_TESSELLATION_CONTROL);
    if (myCgTessellationControlProfile == CG_PROFILE_UNKNOWN) {
      fprintf(stderr, "%s: tessellation control profile is not available.\n", myProgramName);
      exit(0);
    }
    cgGLSetOptimalOptions(myCgTessellationControlProfile);
    checkForCgError("selecting tessellation control profile");

    myCgTessellationControlProgram =
      cgCreateProgramFromFile(
        myCgContext,                            /* Cg runtime context */
        CG_SOURCE,                              /* Program in human-readable form */
        myCgProgramFileName,                    /* Name of file containing program */
        myCgTessellationControlProfile,         /* Tessellation control profile */
        myCgTessellationControlProgramName,     /* Entry function name */
        NULL);                                  /* No extra compiler options */
    checkForCgError("creating tessellation control program from file");

    myCgTessellationEvaluationProfile = cgGLGetLatestProfile(CG_GL_TESSELLATION_EVALUATION);
    if (myCgTessellationEvaluationProfile == CG_PROFILE_UNKNOWN) {
      fprintf(stderr, "%s: tessellation evaluation profile is not available.\n", myProgramName);
      exit(0);
    }
    cgGLSetOptimalOptions(myCgTessellationEvaluationProfile);
    checkForCgError("selecting tessellation evaluation profile");

    myCgTessellationEvaluationProgram =
      cgCreateProgramFromFile(
        myCgContext,                               /* Cg runtime context */
        CG_SOURCE,                                 /* Program in human-readable form */
        myCgProgramFileName,                       /* Name of file containing program */
        myCgTessellationEvaluationProfile,         /* Tessellation evaluation profile */
        myCgTessellationEvaluationProgramName,     /* Entry function name */
        NULL);                                     /* No extra compiler options */
    checkForCgError("creating tessellation evaluation program from file");
  }

  myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  cgGLSetOptimalOptions(myCgFragmentProfile);
  checkForCgError("selecting fragment profile");

  myCgFragmentProgram =
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myCgProgramFileName,        /* Name of file containing program */
      myCgFragmentProfile,        /* Fragment profile */
      myCgFragmentProgramName,    /* Entry function name */
      NULL);                      /* No extra compiler options */
  checkForCgError("creating fragment program from file");

  if (tessellation) {
#if 0
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
#endif
    {
      /* ...otherwise load programs orthogonally */

      cgGLLoadProgram(myCgVertexProgram);
      checkForCgError("loading vertex program");
      cgGLLoadProgram(myCgTessellationControlProgram);
      checkForCgError("loading tessellation control program");
      cgGLLoadProgram(myCgTessellationEvaluationProgram);
      checkForCgError("loading tessellation evaluation program");
      cgGLLoadProgram(myCgFragmentProgram);
      checkForCgError("loading fragment program");
    }

    myCgParameter_innerTess = cgGetNamedParameter(myCgTessellationControlProgram, "innerTess");
    checkForCgError("could not get innerTess parameter");
    myCgParameter_outerTess = cgGetNamedParameter(myCgTessellationControlProgram, "outerTess");
    checkForCgError("could not get outerTess parameter");
    myCgParameter_modelview = cgGetNamedParameter(myCgTessellationControlProgram, "modelview");
    checkForCgError("could not get modelview parameter");
    myCgParameter_projection = cgGetNamedParameter(myCgTessellationEvaluationProgram, "projection");
    checkForCgError("could not get projection parameter");

    cgSetMatrixParameterfr(myCgParameter_modelview, modelview);
    cgSetMatrixParameterfr(myCgParameter_projection, projection);

    /* Query GL for edge tessellation limit */
    glGetFloatv(GL_MAX_TESS_GEN_LEVEL,&edgeTessMax);
  }

  glutMainLoop();
  return 0;
}

static void drawPatchPositions(void)
{
  int i;

  glLineWidth(2.0);
  glColor3f(0.4,0.4,0.4);
  for (i=0; i<2; ++i) {
    glBegin(GL_LINE_LOOP);
    glVertex2fv(patchPosition[patchIndex[i][0]]);
    glVertex2fv(patchPosition[patchIndex[i][1]]);
    glVertex2fv(patchPosition[patchIndex[i][3]]);
    glVertex2fv(patchPosition[patchIndex[i][2]]);
    glEnd();
  }
  glLineWidth(1.0);

  glPointSize(10.0);
  glBegin(GL_POINTS);
  for (i=0; i<6; ++i) {
    if (i==editIndex)
        glColor3f(1,1,0);
    else
        glColor3f(1.0,0.5,0.4);
    glVertex2fv(patchPosition[i]);
  }
  glEnd();
}

static void drawPatches(void)
{
  int i, j;
  glPatchParameteri(GL_PATCH_VERTICES, 4);
  glBegin(GL_PATCHES);
  for (i=0; i<2; ++i)
    for (j=0; j<4; ++j)
      glVertex2fv(patchPosition[patchIndex[i][j]]);
  glEnd();
}

static void display(void)
{
  /* Update patch positions, if animating */
  if (animation) {
    const double radius = 0.2;
    double time = glutGet(GLUT_ELAPSED_TIME)/1000.0;
    patchPosition[1][0] =      radius * cos(time*1.3);
    patchPosition[1][1] =  1 + radius * sin(time*1.3);
    patchPosition[4][0] =      radius * cos(time*0.9);
    patchPosition[4][1] = -1 + radius * sin(time*0.9);
  }

  /* Render */
  glClear(GL_COLOR_BUFFER_BIT);

  if (tessellation) {
    cgSetParameter2iv(myCgParameter_innerTess, innerTess);
    cgSetParameter4fv(myCgParameter_outerTess, edgeTess);

    cgGLEnableProfile(myCgVertexProfile);
    checkForCgError("enabling vertex profile");

    cgGLEnableProfile(myCgTessellationControlProfile);
    checkForCgError("enabling tessellation control profile");

    cgGLEnableProfile(myCgTessellationEvaluationProfile);
    checkForCgError("enabling tessellation evaluation profile");

    cgGLEnableProfile(myCgFragmentProfile);
    checkForCgError("enabling fragment profile");

  #if 0
    if (myCgCombinedProgram)
    {
      cgGLBindProgram(myCgCombinedProgram);
      checkForCgError("binding combined program");
    }
    else
  #endif
    {
      cgGLBindProgram(myCgVertexProgram);
      checkForCgError("binding vertex program");

      cgGLBindProgram(myCgTessellationControlProgram);
      checkForCgError("binding tessellation control program");

      cgGLBindProgram(myCgTessellationEvaluationProgram);
      checkForCgError("binding tessellation evaluation program");

      cgGLBindProgram(myCgFragmentProgram);
      checkForCgError("binding fragment program");
    }

    glPolygonMode(GL_FRONT_AND_BACK, polygonModeList[polygonMode]);
    drawPatches();

    cgGLDisableProfile(myCgVertexProfile);
    checkForCgError("disabling vertex profile");

    cgGLDisableProfile(myCgTessellationControlProfile);
    checkForCgError("disabling tessellation control profile");

    cgGLDisableProfile(myCgTessellationEvaluationProfile);
    checkForCgError("disabling tessellation evaluation profile");

    cgGLDisableProfile(myCgFragmentProfile);
    checkForCgError("disabling fragment profile");
  }

  glLoadMatrixf(modelview);
  drawPatchPositions();

  glutSwapBuffers();

  if (animation)
    glutPostRedisplay();
}

static void keyboard(unsigned char c, int x, int y)
{
  int i;

  switch (c) {
    case 'f': if (polygonModeList[++polygonMode]==GL_NONE) polygonMode = 0; break;
    case 'p': editIndex = (editIndex+1)%6;                                  break;
    case ' ': animation = !animation;                                       break;

    case '+': innerTess[0]++; innerTess[1]++; break;
    case '-': innerTess[0]--; innerTess[1]--; break;

    case '[': edgeTess[0] -= edgeTessStep; edgeTess[2] -= edgeTessStep; break;
    case ']': edgeTess[0] += edgeTessStep; edgeTess[2] += edgeTessStep; break;
    case '{': edgeTess[1] -= edgeTessStep; edgeTess[3] -= edgeTessStep; break;
    case '}': edgeTess[1] += edgeTessStep; edgeTess[3] += edgeTessStep; break;

    case 27:
      /* Esc key */
      /* Demonstrate proper deallocation of Cg runtime data structures.
         Not strictly necessary if we are simply going to exit. */
      cgDestroyProgram(myCgTessellationControlProgram);
      cgDestroyProgram(myCgTessellationEvaluationProgram);
      cgDestroyProgram(myCgFragmentProgram);
      if (myCgCombinedProgram)
        cgDestroyProgram(myCgCombinedProgram);
      cgDestroyContext(myCgContext);
      exit(0);
      break;
  }

  /* Clamp innerTess to valid range */
  for (i=0; i<2; ++i)
    if (innerTess[i]<2)
      innerTess[i] = 2;

  /* Clamp edgeTess to valid range */
  for (i=0; i<4; ++i) {
    if (edgeTess[i]<edgeTessMin) edgeTess[i] = edgeTessMin;
    if (edgeTess[i]>edgeTessMax) edgeTess[i] = edgeTessMax;
  }

  glutPostRedisplay();
}

static void special(int key, int x, int y)
{
  switch (key) {
    case GLUT_KEY_LEFT:  if (editIndex>=0) patchPosition[editIndex][0] -= editStep; break;
    case GLUT_KEY_RIGHT: if (editIndex>=0) patchPosition[editIndex][0] += editStep; break;
    case GLUT_KEY_DOWN:  if (editIndex>=0) patchPosition[editIndex][1] -= editStep; break;
    case GLUT_KEY_UP:    if (editIndex>=0) patchPosition[editIndex][1] += editStep; break;
  }
  glutPostRedisplay();
}
