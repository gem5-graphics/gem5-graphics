
/* vertex_texture.c - Cg vertex textureing example for simple displacement mapping. */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed? */
#include <Cg/cgGL.h>  /* 3D API specific Cg runtime API for OpenGL */

#include "pgm_load.h"
#include "request_vsync.h"
#include "mesh2d.h"

const char *programName = "vertex_texture"; /* Program name for messages. */
static const char *myVertexProgramFileName = "vertex_texture.cg",
                  *myVertexProgramName = "displace_mesh";

/* Cg global variables */
static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile;
static CGprogram   myCgVertexProgram;
static CGparameter myCgVertexParam_modelViewProj;

/* OpenGL texture objects */
static GLuint bumpsTexture, surfaceTexture, texobj;

static int enableSync = 1;  /* Sync buffer swaps to monitor refresh rate. */
static float myProjectionMatrix[16];

/* Forward declared GLUT callbacks registered by main. */
static void display(void);
static void keyboard(unsigned char c, int x, int y);
static void reshape(int width, int height);

static void initMesh(void);

static void checkForCgError(const char *situation)
{
  CGerror error;
  const char *string = cgGetLastErrorString(&error);

  if (error != CG_NO_ERROR) {
    printf("%s: %s: %s\n",
      programName, situation, string);
    if (error == CG_COMPILER_ERROR) {
      printf("%s\n", cgGetLastListing(myCgContext));
    }
    exit(1);
  }
}

int main(int argc, char **argv)
{
  const char *bumpsFileName = "bumps.pgm";
  const char *surfaceFileName = "surface.pgm";
  GLenum vertexTextureInternalFormat;
  GLint maxVertexTextureImageUnits = 0;
  int i;

  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInit(&argc, argv);

  for (i=1; i<argc; i++) {
    if (!strcmp("-nosync", argv[i])) {
      enableSync = 0;
    }
  }

  glutCreateWindow(programName);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutReshapeFunc(reshape);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_5) {
    fprintf(stderr, "%s: Failed to initialize GLEW. OpenGL 1.5 required.\n", programName);    
    exit(1);
  }

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

  myCgVertexParam_modelViewProj =
    cgGetNamedParameter(myCgVertexProgram, "modelViewProj");
  checkForCgError("could not get modelViewProj parameter");

  /* Query OpenGL 2.0 vertex texture image units.  We need at least 1
     vertex texture image unit to expect this example to work. */
  glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS,
    &maxVertexTextureImageUnits);
  printf("%s: OpenGL reports %d vertex texture image units supported\n",
    programName, (int) maxVertexTextureImageUnits);
  if (maxVertexTextureImageUnits < 1) {
    fprintf(stderr, "%s: at least 1 vertex texture image unit is required\n",
      programName);
    exit(0);
  }

  requestSynchronizedSwapBuffers(enableSync);
  glClearColor(0.3, 0.1, 0.4, 0.0);  /* Blue background */
  glEnable(GL_DEPTH_TEST);
  initMesh();

  if (glutExtensionSupported("GL_ARB_texture_float")) {
    /* GeForce 6 and 7 support vertex textures but only for specifc
       float texture formats provided by the ARB_texture_float
       extension.  So use 32-bit float intensity texture instead of
       regular (fixed-point) intensity format if the ARB_texture_float
       extension is available. */
    vertexTextureInternalFormat = GL_INTENSITY32F_ARB;
  } else {
    vertexTextureInternalFormat = GL_INTENSITY;
  }
  bumpsTexture = pgm_load(bumpsFileName, vertexTextureInternalFormat);
  if (0 == bumpsTexture) {
    fprintf(stderr, "%s: failed to load PGM file %s\n",
      programName, bumpsFileName);
    exit(1);
  }
  surfaceTexture = pgm_load(surfaceFileName, vertexTextureInternalFormat);
  if (0 == surfaceTexture) {
    fprintf(stderr, "%s: failed to load PGM file %s\n",
      programName, surfaceFileName);
    exit(1);
  }
  texobj = surfaceTexture;
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texobj);

  glutMainLoop();
  return 0;
}

/* Forward declared routine used by reshape callback. */
static void buildPerspectiveMatrix(double, double,
                                   double, double, float m[16]);

static void reshape(int width, int height)
{
  double aspectRatio = (float) width / (float) height;
  double fieldOfView = 40.0; /* Degrees */

  /* Build projection matrix once. */
  buildPerspectiveMatrix(fieldOfView, aspectRatio,
                         1.0, 20.0,  /* Znear and Zfar */
                         myProjectionMatrix);
  glViewport(0, 0, width, height);
}

static const double myPi = 3.14159265358979323846;

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

static Mesh2D_GL glmesh;

static void initMesh(void)
{
  Mesh2D mesh;

  mesh = createMesh2D(0, 1, 0, 1, 170, 170);
  glmesh = createMesh2D_GL(mesh);
  freeMesh2D(mesh);
}

static void drawMesh(void)
{
  bindMesh2D_GL(glmesh);
  renderMesh2D_GL(glmesh);
}

static float eyeAngle = 0.2;

static void display(void)
{
  float viewMatrix[16], modelViewProjMatrix[16];

  buildLookAtMatrix(8*cos(eyeAngle), 8*sin(eyeAngle), 4.0,  /* eye position */
                    0, 0, 0, /* view center */
                    0, 0, 1, /* up vector */
                    viewMatrix);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  cgGLBindProgram(myCgVertexProgram);
  checkForCgError("binding vertex program");

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  multMatrix(modelViewProjMatrix, myProjectionMatrix, viewMatrix);
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgUpdateProgramParameters(myCgVertexProgram);

  drawMesh();

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  glutSwapBuffers();
}

static void idle(void)
{
  eyeAngle += 0.01;
  while (eyeAngle > 2*3.14159) {
    eyeAngle -= 2*3.14159;
  }
  glutPostRedisplay();
}

static void keyboard(unsigned char c, int x, int y)
{
  static int wireframe = 0;
  static int animating = 0;

  /* Asume pgm_load defaults to GL_NEAREST. */
  static int filtering = GL_NEAREST;

  switch (c) {
  case 27:  /* Esc key */
    /* Demonstrate proper deallocation of Cg runtime data structures.
       Not strictly necessary if we are simply going to exit. */
    cgDestroyProgram(myCgVertexProgram);
    cgDestroyContext(myCgContext);
    exit(0);
    break;
  case 'w':
    wireframe = !wireframe;
    if (wireframe) {
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    break;
  case 'f':
    /* Toggle between nearest and linear filtering. */
    if (filtering == GL_NEAREST) {
      /* GeForce 6 and 7 hardware support vertex textures but only when
         using nearest filtering.  Specifying linear filtering will work
         but causes the driver to revert to sloow CPU-based transform
         and vertex texturing.

         GeForce 8 and subsequent hardware supports vertex texturing in
         hardware without regard to the filtering mode.  So you can use
         linear filtering on GeForce 8 with no significant performance
         change. */
      filtering = GL_LINEAR;
    } else {
      filtering = GL_NEAREST;
    }
    glActiveTexture(GL_TEXTURE0);
    /* Update filtering state for both textures. */
    glBindTexture(GL_TEXTURE_2D, bumpsTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filtering);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filtering);
    glBindTexture(GL_TEXTURE_2D, surfaceTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filtering);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filtering);
    /* Bind back to current texture. */
    glBindTexture(GL_TEXTURE_2D, texobj);
    break;
  case 't':
    if (texobj == surfaceTexture) {
      texobj = bumpsTexture;
    } else {
      texobj = surfaceTexture;
    }
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texobj);
    break;
  case ' ':
    animating = !animating; /* Toggle */
    if (animating) {
      glutIdleFunc(idle);
    } else {
      glutIdleFunc(NULL);
    }    
    break;
  default:
    return;
  }
  glutPostRedisplay();
}
