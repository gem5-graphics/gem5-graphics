
/* generic_attribs.c - OpenGL-based example for use of generic vertex
   attributes in both assembly and GLSL profiles. */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   1.5 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <string.h>   /* for strcmp and strlen */
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

#include "loadtex.h"
#include "md2.h"
#include "md2render.h"

static CGcontext   myCgContext;
static CGprogram   myCgComboProgram[2][2][2];
static CGparameter myCgVertexParam_modelViewProj,
                   myCgVertexParam_keyFrameBlend,
                   myCgVertexParam_light_eyePosition,
                   myCgVertexParam_light_lightPosition;

const char *myProgramName = "generic_attribs";

static float myLightAngle = 0.78f;  /* Angle in radians light rotates around knight. */
static float myLightHeight = 12.0f; /* Vertical height of light. */
static float myEyeAngle = 0.53f;    /* Angle in radians eye rotates around knight. */

static float myProjectionMatrix[16];
static float mySpecularExponent = 8.0f;
static float myAmbient = 0.2f;
static float myLightColor[3] = { 1, 1, 1 };  /* White */

static int myVerbose = 0;

static int myVertexProgramIndex = 0,
           myFragmentProgramIndex = 0,
           myUseGLSL = 0;

static Md2Model *myKnightModel;
static MD2render *myMD2render;
static float myFrameKnob = 0;

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

typedef struct {
  const char *filename;
  const char *entry;
  CGprofile profile;
} ProgramInfo;

ProgramInfo vertexProgramInfo[] = {
  { "C6E3v_keyFrame.cg",    "generic_attrib_keyFrame",    0 },  /* Page 159 */
  { "C6E4v_litKeyFrame.cg", "generic_attrib_litKeyFrame", 0 },  /* Page 161-2 */
};
#define LENGTHOF(_array) (sizeof(_array)/sizeof(_array[0]))
const int vertexProgramInfoCount = LENGTHOF(vertexProgramInfo);

ProgramInfo fragmentProgramInfo[] = {
  { "texmodulate.cg", "texmodulate", 0 },
  { "colorinterp.cg", "colorinterp", 0 },
};
const int fragmentProgramInfoCount = LENGTHOF(fragmentProgramInfo);

typedef struct {
  const char *name;
  CGtype type;
  CGparameter *param;
} ContextParameter;

ContextParameter ctxParam[] = {
  { "modelViewProj",       CG_FLOAT4x4, &myCgVertexParam_modelViewProj },
  { "keyFrameBlend",       CG_FLOAT,    &myCgVertexParam_keyFrameBlend },
  { "light.eyePosition",   CG_FLOAT3,   &myCgVertexParam_light_eyePosition },
  { "light.lightPosition", CG_FLOAT3,   &myCgVertexParam_light_lightPosition },
};
const int ctxParamCount = LENGTHOF(ctxParam);

static void reportParameters(CGparameter param)
{
  if (param) {
    while (param) {
      const char *name = cgGetParameterName(param);
      CGtype type = cgGetParameterType(param);
      const char *type_name = cgGetTypeString(type);
      const char *resourceName = cgGetParameterResourceName(param);
      const char *semantic = cgGetParameterSemantic(param);

      printf("    %s type=%s", name, type_name);
      if (semantic && *semantic != '\0') {
        printf(" sem=%s", semantic);
      }
      if (resourceName && *resourceName != '\0') {
        printf(" rn=%s", resourceName);
      }
      printf(";\n");

      param = cgGetNextLeafParameter(param);
    }
  } else {
    printf("    <none>:\n");
  }
}

static const char *domainString(CGdomain domain)
{
  switch (domain) {
  case CG_VERTEX_DOMAIN:
    return "VERTEX";
  case CG_FRAGMENT_DOMAIN:
    return "FRAGMENT";
  case CG_GEOMETRY_DOMAIN:
    return "GEOMETRY";
  case CG_UNKNOWN_DOMAIN:
  default:
    return "UNKNOWN";
  }
}

static void reportCombineProgramInfo(CGprogram program)
{
  int numDomains = cgGetNumProgramDomains(program);
  CGparameter param;
  int i;

  assert(numDomains == 2);

  /* Expect a combined program to have empty parameter lists; the
     parameters are associated with the domain programs. */
  param = cgGetFirstLeafParameter(program, CG_GLOBAL);
  assert(0 == param);
  param = cgGetFirstLeafParameter(program, CG_PROGRAM);
  assert(0 == param);

  for (i=0; i<numDomains; i++) {
    CGprogram subprog = cgGetProgramDomainProgram(program, i);
    const char *entry = cgGetProgramString(subprog, CG_PROGRAM_ENTRY);
    CGprofile profile = cgGetProgramProfile(subprog);
    const char *profile_name = cgGetProfileString(profile);
    CGdomain domain = cgGetProfileDomain(profile);

    printf("%d: %s PROGRAM %s %s\n", i, domainString(domain), profile_name, entry);
    printf("  Global parameters:\n");
    param = cgGetFirstParameter(subprog, CG_GLOBAL);
    reportParameters(param);
    printf("  Local parameters:\n");
    param = cgGetFirstParameter(subprog, CG_PROGRAM);
    reportParameters(param);
  }
}

static void reportProgramInfo(CGprogram program)
{
  const char *entry = cgGetProgramString(program, CG_PROGRAM_ENTRY);
  CGprofile profile = cgGetProgramProfile(program);
  const char *profile_name = cgGetProfileString(profile);
  CGdomain domain = cgGetProfileDomain(profile);
  CGparameter param = 0;

  printf("%s PROGRAM %s %s\n", domainString(domain), profile_name, entry);
  printf("  Global parameters:\n");
  param = cgGetFirstParameter(program, CG_GLOBAL);
  reportParameters(param);
  printf("  Local parameters:\n");
  param = cgGetFirstParameter(program, CG_PROGRAM);
  reportParameters(param);
}

static CGprogram createComboProgram(CGcontext ctx,
                                    ProgramInfo *vertex,
                                    ProgramInfo *fragment,
                                    const char *compilerArgs[],
                                    int plistLen,
                                    ContextParameter plist[])
{
  CGprogram v, f;
  CGprogram program, programList[2];
  int i;

  v = cgCreateProgramFromFile(ctx, CG_SOURCE,
                              vertex->filename, vertex->profile, vertex->entry,
                              compilerArgs);
  checkForCgError("creating vertex program from file");

  if (myVerbose) {
    reportProgramInfo(v);
  }

  f = cgCreateProgramFromFile(ctx, CG_SOURCE,
                              fragment->filename, fragment->profile, fragment->entry,
                              compilerArgs);
  checkForCgError("creating fragment program from file");

  if (myVerbose) {
    reportProgramInfo(f);
  }

  programList[0] = v;
  programList[1] = f;
  program = cgCombinePrograms(2, programList);

  checkForCgError("combining programs");
  cgGLLoadProgram(program);
  checkForCgError("loading combo program");

  if (myVerbose) {
    reportCombineProgramInfo(program);
  }

  for (i=0; i<plistLen; i++) {
    CGparameter ctxParam = *plist[i].param;
    CGparameter progParam = cgGetNamedParameter(program, plist[i].name);

    /* Does the combined program have a parameter of this name? */
    if (progParam) {
      /* Yes, so connect the program parameter to a context parameter... */

      /* Has the context CGparameter not been initialized yet? */
      if (ctxParam == 0) {
        /* Not yet initialized, so create the context parameter. */
        ctxParam = cgCreateParameter(ctx, plist[i].type);
        checkForCgError("creating context parameter");
        *plist[i].param = ctxParam;
      }
      cgConnectParameter(/*from*/ctxParam, /*to*/progParam);
      checkForCgError("connect parameter");
    } else {
      if (myVerbose) {
        printf("%s: CGprogram %p lacks a parameter named %s\n",
          myProgramName, program, plist[i].name);
      }
    }
  }
  return program;
}

/* Forward declared GLUT callbacks registered by main. */
static void reshape(int width, int height);
static void visibility(int state);
static void display(void);
static void keyboard(unsigned char c, int x, int y);
static void menu(int item);
static void mouse(int button, int state, int x, int y);
static void motion(int x, int y);
static void requestSynchronizedSwapBuffers(void);

int main(int argc, char **argv)
{
  gliGenericImage *decalImage;
  int i, j, k;

  glutInitWindowSize(700, 700);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);

  glutInit(&argc, argv);

  for (i=1; i<argc; i++) {
    if (!strcmp("-v", argv[i])) {
      myVerbose = 1;
    } else if (!strcmp("-glsl", argv[i])) {
      myUseGLSL = 1;
    } else if (!strcmp("-lit", argv[i])) {
      myVertexProgramIndex = 1;
      myFragmentProgramIndex = 1;
    } else {
      fprintf(stderr, "%s: supports -v, -glsl, and -lit options\n", myProgramName);
    }
  }

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutVisibilityFunc(visibility);
  glutKeyboardFunc(keyboard);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_2_0) {
    fprintf(stderr, "%s: Failed to initialize GLEW. OpenGL 2.0 required.\n", myProgramName);    
    exit(1);
  }

  myKnightModel = md2ReadModel("knight.md2");
  if (0 == myKnightModel) {
    fprintf(stderr, "%s: count not load knight.md2\n", myProgramName);
    exit(1);
  }
  myMD2render = createMD2render(myKnightModel);
  decalImage = readImage("knight.tga");
  decalImage = loadTextureDecal(decalImage, 1);
  gliFree(decalImage);

  requestSynchronizedSwapBuffers();
  glClearColor(0.3, 0.3, 0.1, 0);  /* Gray background. */
  glEnable(GL_DEPTH_TEST);         /* Hidden surface removal. */
  glEnable(GL_CULL_FACE);          /* Backface culling. */
  glLineWidth(3.0f);

  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgGLSetDebugMode(CG_FALSE);

  for (i=0; i<2; i++) { /* 0=assembly, 1=GLSL */
    for (j=0; j<2; j++) { /* 0=justSkinning, 1=litSkinning */
      for (k=0; k<2; k++) { /* 0=modulateTexture, 1=justInterpolateColor */
        CGprogram program;
        CGparameter param;

        /* If assembly... */
        if (i==0) {
          /* Use latest supported assembly profiles. */
          vertexProgramInfo[j].profile = cgGLGetLatestProfile(CG_GL_VERTEX);
          fragmentProgramInfo[k].profile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
        } else {
          /* Use GLSL profiles. */
          vertexProgramInfo[j].profile = CG_PROFILE_GLSLV;
          fragmentProgramInfo[k].profile = CG_PROFILE_GLSLF;
        }
        program = createComboProgram(myCgContext,
          &vertexProgramInfo[j], &fragmentProgramInfo[k],
          NULL, ctxParamCount, ctxParam);

        /* Set texture scale factor once */
        param = cgGetNamedParameter(program, "scaleFactor");
        if (param) {
          cgSetParameter2f(param,
            1.0f/myKnightModel->header.skinWidth, 1.0f/myKnightModel->header.skinHeight);
        }

        /* Set light source color parameters once. */
        param = cgGetNamedParameter(program, "light.lightColor");
        if (param) {
          cgSetParameter4fv(param, myLightColor);
        }
        param = cgGetNamedParameter(program, "light.specularExponent");
        if (param) {
          cgSetParameter1f(param, mySpecularExponent);
        }
        param = cgGetNamedParameter(program, "light.ambient");
        if (param) {
          cgSetParameter1f(param, myAmbient);
        }

        myCgComboProgram[i][j][k] = program;
      }
    }
  }

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAddMenuEntry("[g] Toggle GLSL versus latest profile", 'g');
  glutAddMenuEntry("[w] Toggle wireframe", 'w');
  glutAddMenuEntry("[Enter] Toggle lighting/texture", 13);
  glutAddMenuEntry("[v] Toggle vertex program", 'f');
  glutAddMenuEntry("[f] Toggle fragment program", 'v');
  glutAddMenuEntry("[Esc] Quit", 27);
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

/* Forward declared routine used by reshape callback. */
static void buildPerspectiveMatrix(double fieldOfView,
                                   double aspectRatio,
                                   double zMin, double zMax,
                                   float m[16]);

static void reshape(int width, int height)
{
  double aspectRatio = (float) width / (float) height;
  double fieldOfView = 40.0; /* Degrees */

  /* Build projection matrix once. */
  buildPerspectiveMatrix(fieldOfView, aspectRatio,
                         10.0, 200.0,  /* Znear and Zfar */
                         myProjectionMatrix);
  glViewport(0, 0, width, height);
}

static const double myPi = 3.14159265358979323846;

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for gluPerspective. */
static void buildPerspectiveMatrix(double fieldOfView,
                                   double aspectRatio,
                                   double zNear, double zFar,
                                   float m[16])
{
  double sine, cotangent, deltaZ;
  double radians = fieldOfView / 2.0 * myPi / 180.0;
  
  deltaZ = zFar - zNear;
  sine = sin(radians);
  /* Should be non-zero to avoid division by zero. */
  assert(deltaZ);
  assert(sine);
  assert(aspectRatio);
  cotangent = cos(radians) / sine;
  
  m[0*4+0] = cotangent / aspectRatio;
  m[0*4+1] = 0.0;
  m[0*4+2] = 0.0;
  m[0*4+3] = 0.0;
  
  m[1*4+0] = 0.0;
  m[1*4+1] = cotangent;
  m[1*4+2] = 0.0;
  m[1*4+3] = 0.0;
  
  m[2*4+0] = 0.0;
  m[2*4+1] = 0.0;
  m[2*4+2] = -(zFar + zNear) / deltaZ;
  m[2*4+3] = -2 * zNear * zFar / deltaZ;
  
  m[3*4+0] = 0.0;
  m[3*4+1] = 0.0;
  m[3*4+2] = -1;
  m[3*4+3] = 0;
}

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for gluLookAt. */
static void buildLookAtMatrix(double eyex, double eyey, double eyez,
                              double centerx, double centery, double centerz,
                              double upx, double upy, double upz,
                              float m[16])
{
  double x[3], y[3], z[3], mag;

  /* Difference eye and center vectors to make Z vector. */
  z[0] = eyex - centerx;
  z[1] = eyey - centery;
  z[2] = eyez - centerz;
  /* Normalize Z. */
  mag = sqrt(z[0]*z[0] + z[1]*z[1] + z[2]*z[2]);
  if (mag) {
    z[0] /= mag;
    z[1] /= mag;
    z[2] /= mag;
  }

  /* Up vector makes Y vector. */
  y[0] = upx;
  y[1] = upy;
  y[2] = upz;

  /* X vector = Y cross Z. */
  x[0] =  y[1]*z[2] - y[2]*z[1];
  x[1] = -y[0]*z[2] + y[2]*z[0];
  x[2] =  y[0]*z[1] - y[1]*z[0];

  /* Recompute Y = Z cross X. */
  y[0] =  z[1]*x[2] - z[2]*x[1];
  y[1] = -z[0]*x[2] + z[2]*x[0];
  y[2] =  z[0]*x[1] - z[1]*x[0];

  /* Normalize X. */
  mag = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
  if (mag) {
    x[0] /= mag;
    x[1] /= mag;
    x[2] /= mag;
  }

  /* Normalize Y. */
  mag = sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2]);
  if (mag) {
    y[0] /= mag;
    y[1] /= mag;
    y[2] /= mag;
  }

  /* Build resulting view matrix. */
  m[0*4+0] = x[0];  m[0*4+1] = x[1];
  m[0*4+2] = x[2];  m[0*4+3] = -x[0]*eyex + -x[1]*eyey + -x[2]*eyez;

  m[1*4+0] = y[0];  m[1*4+1] = y[1];
  m[1*4+2] = y[2];  m[1*4+3] = -y[0]*eyex + -y[1]*eyey + -y[2]*eyez;

  m[2*4+0] = z[0];  m[2*4+1] = z[1];
  m[2*4+2] = z[2];  m[2*4+3] = -z[0]*eyex + -z[1]*eyey + -z[2]*eyez;

  m[3*4+0] = 0.0;   m[3*4+1] = 0.0;  m[3*4+2] = 0.0;  m[3*4+3] = 1.0;
}

/* Simple 4x4 matrix by 4x4 matrix multiply. */
static void multMatrix(float dst[16],
                       const float src1[16], const float src2[16])
{
  float tmp[16];
  int i, j;

  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      tmp[i*4+j] = src1[i*4+0] * src2[0*4+j] +
                   src1[i*4+1] * src2[1*4+j] +
                   src1[i*4+2] * src2[2*4+j] +
                   src1[i*4+3] * src2[3*4+j];
    }
  }
  /* Copy result to dst (so dst can also be src1 or src2). */
  for (i=0; i<16; i++)
    dst[i] = tmp[i];
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

void *font = GLUT_BITMAP_9_BY_15;

void
output(const char *string)
{
  int len, i;

  len = (int) strlen(string);
  for (i = 0; i < len; i++) {
    glutBitmapCharacter(font, string[i]);
  }
}

static void reportMode(CGprogram p1, CGprogram p2)
{
  char buf[1024];

  glColor3f(0.5,1,1);
  glWindowPos2i(10, 25);
  sprintf(buf, "Using %s (\"v\" toggles) & %s (\"f\" toggles)",
    cgGetProgramString(p1,CG_PROGRAM_ENTRY),
    cgGetProgramString(p2,CG_PROGRAM_ENTRY));
  output(buf);
  glWindowPos2i(10, 10);
  if (myUseGLSL) {
    output("Using GLSL profiles, \"g\" toggles");
  } else {
    sprintf(buf, "Using %s/%s profiles, \"g\" toggles",
      cgGetProfileString(cgGetProgramProfile(p1)),
      cgGetProfileString(cgGetProgramProfile(p2)));
    output(buf);
  }
}

static CGprogram currentComboProgram(void)
{
  return myCgComboProgram[myUseGLSL][myVertexProgramIndex][myFragmentProgramIndex];
}

static void display(void)
{
  const CGprogram program = currentComboProgram();
  const CGprofile profile0 = cgGetProgramDomainProfile(program, 0);
  const CGprofile profile1 = cgGetProgramDomainProfile(program, 1);

  /* World-space positions for light and eye. */
  const float eyeRadius = 85,
              lightRadius = 40;
  const float eyePosition[3] = { cos(myEyeAngle)*eyeRadius, 0, sin(myEyeAngle)*eyeRadius };
  const float lightPosition[3] = { lightRadius*sin(myLightAngle), 
                                   myLightHeight,
                                   lightRadius*cos(myLightAngle) };

  const int frameA = floor(myFrameKnob),
            frameB = addDelta(myFrameKnob, 1, myKnightModel->header.numFrames);

  float viewMatrix[16], modelViewProjMatrix[16];

  buildLookAtMatrix(eyePosition[0], eyePosition[1], eyePosition[2],
                    0, 0, 0,
                    0, 1, 0,
                    viewMatrix);
  /* modelViewProj = projectionMatrix * viewMatrix (model is identity) */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, viewMatrix);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgSetParameter1f(myCgVertexParam_keyFrameBlend, myFrameKnob-floor(myFrameKnob));
  /* Set eye and light positions if lighting. */
  if (myVertexProgramIndex == 1) {
    cgSetParameter3fv(myCgVertexParam_light_lightPosition, lightPosition);
    cgSetParameter3fv(myCgVertexParam_light_eyePosition, eyePosition);
  }

  cgGLBindProgram(program);
  checkForCgError("binding combo program");

  cgGLEnableProfile(profile0);
  checkForCgError("enabling vertex profile");
  cgGLEnableProfile(profile1);
  checkForCgError("enabling fragment profile");

  glVertexAttrib4f(/*attr*/5, 1,1,1,1); /* white */
  drawMD2render_withGenericAttribs(myMD2render, frameA, frameB);

  cgGLDisableProfile(profile0);
  checkForCgError("enabling vertex profile");
  cgGLDisableProfile(profile1);
  checkForCgError("enabling fragment profile");
  cgGLUnbindProgram(profile0);
  checkForCgError("unbinding vertex profile");
  cgGLUnbindProgram(profile1);
  checkForCgError("unbinding fragment profile");

  /* If using lighting vertex program, render light position as yellow sphere. */
  if (myVertexProgramIndex == 1) {
    glPushMatrix();
      /* glLoadMatrixf expects a column-major matrix but Cg matrices are
         row-major (matching C/C++ arrays) so used glLoadTransposeMatrixf
         which OpenGL 1.3 introduced. */
      glLoadTransposeMatrixf(modelViewProjMatrix);
      glTranslatef(lightPosition[0], lightPosition[1], lightPosition[2]);
      glColor3f(1,1,0); /* yellow */
      glutSolidSphere(1, 10, 10);  /* sphere to represent light position */
    glPopMatrix();
  }

  reportMode(cgGetProgramDomainProgram(program, 0),
             cgGetProgramDomainProgram(program, 1));

  glutSwapBuffers();
}

static int myLastElapsedTime;

static void idle(void)
{
  const float millisecondsPerSecond = 1000.0f;
  const float keyFramesPerSecond = 3.0f;
  int now = glutGet(GLUT_ELAPSED_TIME);
  float delta = (now - myLastElapsedTime) / millisecondsPerSecond;

  myLastElapsedTime = now;  /* This time become "prior time". */

  delta *= keyFramesPerSecond;
  myFrameKnob = addDelta(myFrameKnob, delta, myKnightModel->header.numFrames);
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

static void dumpCompiledPrograms(CGprogram program)
{
  const int domains = cgGetNumProgramDomains(program);
  int i;

  printf("=================================\n");
  for (i=0; i<domains; i++) {
    CGprogram subprog = cgGetProgramDomainProgram(program, i);
    const char *entry = cgGetProgramString(subprog, CG_PROGRAM_ENTRY);
    const char *compiledText = cgGetProgramString(subprog, CG_COMPILED_PROGRAM);
    const char *profile = cgGetProgramString(subprog, CG_PROGRAM_PROFILE);

    printf("ENTRY %s for PROFILE %s\n", entry, profile);
    printf("-----------------------------------\n");
    printf("%s\n", compiledText);
  }
}

static void keyboard(unsigned char c, int x, int y)
{
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
  case 13:
    myVertexProgramIndex = 1-myVertexProgramIndex;
    myFragmentProgramIndex = myVertexProgramIndex;
    break;
  case 'd':
    dumpCompiledPrograms(currentComboProgram());
    return;
  case 'f':
    myFragmentProgramIndex = 1-myFragmentProgramIndex;
    break;
  case 'v':
    myVertexProgramIndex = 1-myVertexProgramIndex;
    break;
  case 'g':
    myUseGLSL = !myUseGLSL;
    break;
  case 'w':
    wireframe = !wireframe;
    if (wireframe) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
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

int beginx, beginy;
int moving = 0;
int movingLight = 0;
int xLightBegin, yLightBegin;

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
