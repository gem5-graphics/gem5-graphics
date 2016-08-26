
/* tess_simple.c - OpenGL-based cubic Bezier tessellation shader */
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
static const char *myProgramName       = "tess_bezier",
                  *myCgProgramFileName = "tess_bezier.cg",
                  *myCgVertexProgramName                 = "mainV",
                  *myCgTessellationControlProgramName    = "main",
                  *myCgTessellationEvaluationProgramName = "main",
                  *myCgFragmentProgramName               = "mainF";

/* Bezier cubic patch control point data */
float patchPosition[16][3] = {
  { -1.00,  -1.00, 0.0000 },
  { -0.33,  -1.00, 0.2000 },
  {  0.33,  -1.00, 0.2000 },
  {  1.0,   -1.00, 0.0000 },
  { -1.00,  -0.33, 0.2000 },
  { -0.33,  -0.33, 0.5000 },
  {  0.33,  -0.33, 0.5000 },
  {  1.0,   -0.33, 0.2000 },
  { -1.00,   0.33, 0.2000 },
  { -0.33,   0.33, 0.5000 },
  {  0.33,   0.33, 0.5000 },
  {  1.0,    0.33, 0.2000 },
  { -1.00,   1.00, 0.0000 },
  { -0.33,   1.00, 0.2000 },
  {  0.33,   1.00, 0.2000 },
  {  1.0,    1.00, 0.0000 },
};
int patchIndex[16] = {
  0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15
};

/* Tessellation configuration */
int   innerTess[2]   = { 16, 16 };
float edgeTess[4]    = { 16, 16, 16, 16 };
const float edgeTessStep = 1;
const float edgeTessMin  = 1;
      float edgeTessMax  = 64;

/* Rendering configuration */
int   tessellation = 1;
float modelview[16]  = { 0.7,0,0,0, 0,0.7,0,0, 0,0,0.7,0, 0,0,0,1 }; /* 4x4 modelview matrix     */
float projection[16] = { 1,0,0,0, 0,1,0,0, 0,0,0.1,0, 0,0,0,1 };     /* 4x4 projection matrix    */
const GLenum polygonModeList[] = { GL_LINE, GL_FILL, GL_NONE };      /* GL_NONE terminated array */
      GLuint polygonMode = 0;

/* Interaction configuration */
int   animation = 0;
int   spin = 0;
int   spinTime = 0;
float spinAngle = 1.0;
int   editIndex = -1;
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
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
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
    fprintf(stderr, "%s: NV_gpu_program5 not available - using CPU tessellation.\n", myProgramName);
    tessellation = 0;
  }

  /* Cg 3.0 is required for tessellation support */
  versionStr = cgGetString(CG_VERSION);
  if (!versionStr || sscanf(versionStr, "%d.%d", &versionMajor, &versionMinor)!=2 || versionMajor<3) {
    fprintf(stderr, "%s: Cg 3.0 required, %s detected.\n", myProgramName, versionStr);
    exit(1);
  }

  glClearColor(0.7, 0.7, 0.9, 0.0);  /* Light blue background */
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

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

    /* Query GL for edge tessellation limit */
    glGetFloatv(GL_MAX_TESS_GEN_LEVEL,&edgeTessMax);
  }

  glutMainLoop();
  return 0;
}

static void
bicubicBezierPatch(float *position, const float u, const float v)
{
  int i,j,k;

  /* Bernstein weights */
  const float Bu[4] = { (1.0f-u)*(1.0f-u)*(1.0f-u), 3.0f*u*(1.0f-u)*(1.0f-u), 3.0f*u*u*(1.0f-u), u*u*u };
  const float Bv[4] = { (1.0f-v)*(1.0f-v)*(1.0f-v), 3.0f*v*(1.0f-v)*(1.0f-v), 3.0f*v*v*(1.0f-v), v*v*v };

  position[0] = position[1] = position[2] = 0.0;

  /* Not the most efficient implementation, but resembles the math closely */
  for (i=0; i<4; ++i)
    for (j=0; j<4; ++j)
      for (k=0; k<3; ++k)
        position[k] += Bu[i]*Bv[j]*patchPosition[patchIndex[i+j*4]][k];
}

static void drawPatchCPU(void)
{
  const int subdivide = innerTess[0]+1;  /* +1 to account for outer GPU tessellation */
  float position[3];                     /* temporary storage for each surface point */
  int i,j;

  glColor3f(0.0f,0.0f,0.4f);
  for (i=0; i<subdivide; ++i)
  {
    glBegin(GL_LINE_STRIP);
    for (j=0; j<subdivide; ++j)
    {
      bicubicBezierPatch(position,((float) i)/(subdivide-1),((float) j)/(subdivide-1));
      glVertex3fv(position);
    }
    glEnd();

    glBegin(GL_LINE_STRIP);
    for (j=0; j<subdivide; ++j)
    {
      bicubicBezierPatch(position,((float) j)/(subdivide-1),((float) i)/(subdivide-1));
      glVertex3fv(position);
    }
    glEnd();
  }
}

static void drawPatchPositions(void)
{
  int i;

  /* Patch control points */
  glPointSize(10.0);
  glBegin(GL_POINTS);
  for (i=0; i<16; ++i) {
    if (i==editIndex)
      glColor3f(1,1,0);
    else
      glColor3f(1.0,0.5,0.4);
    glVertex3fv(patchPosition[patchIndex[i]]);
  }
  glEnd();

  /* Edges between patch control points */
  glColor3f(0.5,0.5,0.5);
  for (i=0; i<4; ++i)
  {
    glBegin(GL_LINE_STRIP);
    glVertex3fv(patchPosition[patchIndex[i*4  ]]);
    glVertex3fv(patchPosition[patchIndex[i*4+1]]);
    glVertex3fv(patchPosition[patchIndex[i*4+2]]);
    glVertex3fv(patchPosition[patchIndex[i*4+3]]);
    glEnd();

    glBegin(GL_LINE_STRIP);
    glVertex3fv(patchPosition[patchIndex[i    ]]);
    glVertex3fv(patchPosition[patchIndex[i+1*4]]);
    glVertex3fv(patchPosition[patchIndex[i+2*4]]);
    glVertex3fv(patchPosition[patchIndex[i+3*4]]);
    glEnd();
  }
}

static void drawPatchGPU(void)
{
  int i;
  glPatchParameteri(GL_PATCH_VERTICES, 16);
  glBegin(GL_PATCHES);
  for (i=0; i<16; ++i)
    glVertex3fv(patchPosition[patchIndex[i]]);
  glEnd();
}

static void display(void)
{
  /* Update patch positions, if animating */
  if (animation) {
    const double time = glutGet(GLUT_ELAPSED_TIME)/300.0;
    patchPosition[5][2]  =
    patchPosition[6][2]  =
    patchPosition[9][2]  =
    patchPosition[10][2] = sin(time);
  }

  /* Update modelview matrix, if spinning */
  if (spin) {
    const int time = glutGet(GLUT_ELAPSED_TIME);
    spinAngle += (time-spinTime)/4000.0;
    spinTime = time;
  }

  {
    const float zoom = 0.7;
    const float axis[3] = { 0.71,0.71,0 };
    const float sine = (float) sin(spinAngle);
    const float cosine = (float) cos(spinAngle);
    const float ab = axis[0] * axis[1] * (1 - cosine);
    const float bc = axis[1] * axis[2] * (1 - cosine);
    const float ca = axis[2] * axis[0] * (1 - cosine);
    const float tx = axis[0] * axis[0];
    const float ty = axis[1] * axis[1];
    const float tz = axis[2] * axis[2];

    modelview[0]  = zoom * (tx + cosine * (1 - tx));
    modelview[1]  = zoom * (ab - axis[2] * sine);
    modelview[2]  = zoom * (ca + axis[1] * sine);
    modelview[3]  = 0.0f;

    modelview[4]  = zoom * (ab + axis[2] * sine);
    modelview[5]  = zoom * (ty + cosine * (1 - ty));
    modelview[6]  = zoom * (bc - axis[0] * sine);
    modelview[7]  = 0.0f;

    modelview[8]  = zoom * (ca - axis[1] * sine);
    modelview[9]  = zoom * (bc + axis[0] * sine);
    modelview[10] = zoom * (tz + cosine * (1 - tz));
    modelview[11] = 0;
  }

  /* Render */
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  if (tessellation) {

    /* Update modeview and projection matrices for shaders */
    cgSetMatrixParameterfr(myCgParameter_modelview, modelview);
    cgSetMatrixParameterfr(myCgParameter_projection, projection);

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
    drawPatchGPU();

    cgGLDisableProfile(myCgVertexProfile);
    checkForCgError("disabling vertex profile");

    cgGLDisableProfile(myCgTessellationControlProfile);
    checkForCgError("disabling tessellation control profile");

    cgGLDisableProfile(myCgTessellationEvaluationProfile);
    checkForCgError("disabling tessellation evaluation profile");

    cgGLDisableProfile(myCgFragmentProfile);
    checkForCgError("disabling fragment profile");
  }

  /* Update modeview and projection matrices for fixed-function */
  glMatrixMode(GL_MODELVIEW);
  glLoadTransposeMatrixf(modelview);
  glMatrixMode(GL_PROJECTION);
  glLoadTransposeMatrixf(projection);

  if (!tessellation)
    drawPatchCPU();

  if (editIndex>=0)
    drawPatchPositions();

  glutSwapBuffers();

  if (animation || spin)
    glutPostRedisplay();
}

static void keyboard(unsigned char c, int x, int y)
{
  int i;

  switch (c) {
    case 'f': if (polygonModeList[++polygonMode]==GL_NONE) polygonMode = 0; break;
    case 't': tessellation = !tessellation;                                 break;
    case 'p': editIndex = (editIndex+1)%16;                                 break;
    case 'P': editIndex = -1;                                               break;
    case ' ': animation = !animation;                                       break;
    case 's': spin = !spin;          spinTime = glutGet(GLUT_ELAPSED_TIME); break;

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

  if (!GLEW_NV_gpu_program5 || !GLEW_ARB_tessellation_shader)
    tessellation = 0;

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
