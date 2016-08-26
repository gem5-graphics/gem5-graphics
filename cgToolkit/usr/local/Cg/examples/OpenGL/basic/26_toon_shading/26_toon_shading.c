
/* 26_toon_shading.c - OpenGL-based cartoon (or "toon") shading
   using Cg program from Chapter 7 of "The Cg Tutorial"
   (Addison-Wesley, ISBN 0321194969). */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   1.5 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <string.h>   /* for strlen */
#include <math.h>     /* for sqrt, sin, and cos */
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

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgFragmentProfile;
static CGprogram   myCgVertexProgram,
                   myCgFragmentProgram;
static CGparameter myCgVertexParam_modelViewProj,
                   myCgVertexParam_lightPosition,
                   myCgVertexParam_eyePosition,
                   myCgVertexParam_shininess,
                   myCgFragmentParam_Kd,
                   myCgFragmentParam_Ks;

static const char *myProgramName = "26_toon_shading",
                  *myVertexProgramFileName = "C9E3v_toonShading.cg",
/* Page 247 */    *myVertexProgramName = "C9E3v_toonShading",
                  *myFragmentProgramFileName = "C9E4f_toonShading.cg",
/* Page 248 */    *myFragmentProgramName = "C9E4f_toonShading";

static float myProjectionMatrix[16];

static float eyeHeight = 0.0f;    /* Vertical height of light. */
static float eyeAngle  = 0.53f;   /* Angle in radians eye rotates around monkey. */
static float lightAngle = -0.4;   /* Angle light rotates around scene. */
static float lightHeight = 1.0f;  /* Vertical height of light. */

static float headSpin = 0.0f;  /* Head spin in degrees. */
static float shininess = 8.9f;

static float Kd[4] = { 0.8f, 0.6f, 0.2f, 1.0f }; /* Diffuse color */
static float Ks[4] = { 0.3f, 0.3f, 4.0f, 0.0f }; /* Specular color */

/* Model data: MonkeyHead_vertices, MonkeyHead_normals, and MonkeyHead_triangles */
#include "MonkeyHead.h"

static void drawMonkeyHead(void)
{
  static GLfloat *texcoords = NULL;  /* Malloc'ed buffer, never freed. */

  /* Generate a set of 2D texture coordinate from the scaled (x,y)
     vertex positions. */
  if (texcoords == NULL) {
    const int numVertices = sizeof(MonkeyHead_vertices) /
                            (3*sizeof(MonkeyHead_vertices[0]));
    const float scaleFactor = 1.5;
    int i;

    texcoords = (GLfloat*) malloc(2 * numVertices * sizeof(GLfloat));
    if (texcoords == NULL) {
      fprintf(stderr, "%s: malloc failed\n", myProgramName);
      exit(1);
    }
    for (i=0; i<numVertices; i++) {
      texcoords[i*2 + 0] = scaleFactor * MonkeyHead_vertices[i*3 + 0];
      texcoords[i*2 + 1] = scaleFactor * MonkeyHead_vertices[i*3 + 1];
    }
  }

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  glVertexPointer(3, GL_FLOAT, 3*sizeof(GLfloat), MonkeyHead_vertices);
  glNormalPointer(GL_FLOAT, 3*sizeof(GLfloat), MonkeyHead_normals);
  glTexCoordPointer(2, GL_FLOAT, 2*sizeof(GLfloat), texcoords);

  glDrawElements(GL_TRIANGLES, 3*MonkeyHead_num_of_triangles,
    GL_UNSIGNED_SHORT, MonkeyHead_triangles);
}

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
static void menu(int item);
static void mouse(int button, int state, int x, int y);
static void motion(int x, int y);

/* Other forward declared functions. */
static void requestSynchronizedSwapBuffers(void);
static float diffuseRamp(float x);
static float specularRamp(float x);
static float edgeRamp(float x);
static void loadRamp(GLuint texobj, int size, float (*func)(float x));

/* Use enum to assign unique symbolic OpenGL texture names. */
enum {
  TO_BOGUS = 0,
  TO_DIFFUSE_RAMP,
  TO_SPECULAR_RAMP,
  TO_EDGE_RAMP,
};

int main(int argc, char **argv)
{
  int i;

  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_1) {
    fprintf(stderr, "%s: failed to initialize GLEW, OpenGL 1.1 required.\n", myProgramName);    
    exit(1);
  }

  requestSynchronizedSwapBuffers();
  glClearColor(0.1, 0.1, 0.5, 0);  /* Gray background. */
  glEnable(GL_DEPTH_TEST);         /* Hidden surface removal. */

  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgGLSetDebugMode(CG_FALSE);
  /* The example uses two texture units so let the Cg runtime manage
     binding our samplers. */
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);

  /* Compile and load the vertex program. */
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

#define GET_VERT_PARAM(name) \
  myCgVertexParam_##name = \
    cgGetNamedParameter(myCgVertexProgram, #name); \
  checkForCgError("could not get " #name " parameter");

  GET_VERT_PARAM(modelViewProj);
  GET_VERT_PARAM(lightPosition);
  GET_VERT_PARAM(eyePosition);
  GET_VERT_PARAM(shininess);

  cgSetParameter1f(myCgVertexParam_shininess, shininess);

  myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  cgGLSetOptimalOptions(myCgFragmentProfile);
  checkForCgError("selecting fragment profile");

  /* Compile and load the fragment program. */
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

#define GET_FRAG_PARAM(name) \
  myCgFragmentParam_##name = \
    cgGetNamedParameter(myCgFragmentProgram, #name); \
  checkForCgError("could not get " #name " parameter");

  GET_FRAG_PARAM(Kd);
  GET_FRAG_PARAM(Ks);

  cgSetParameter4fv(myCgFragmentParam_Kd, Kd);
  cgSetParameter4fv(myCgFragmentParam_Ks, Ks);

  for (i=0; i<3; i++) {
    static GLuint texobj[3] = { TO_DIFFUSE_RAMP, TO_SPECULAR_RAMP, TO_EDGE_RAMP };
    static const char *name[3] = { "diffuseRamp", "specularRamp", "edgeRamp" };
    static float (*func[3])(float x) = { diffuseRamp, specularRamp, edgeRamp };
    CGparameter sampler;

    sampler = cgGetNamedParameter(myCgFragmentProgram, name[i]);
    checkForCgError("getting sampler ramp parameter");

    loadRamp(texobj[i], 256, func[i]);

    cgGLSetTextureParameter(sampler, texobj[i]);
    checkForCgError("setting sampler ramp texture");
  }

  /* Create GLUT menu. */
  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAddMenuEntry("[+] Increase shininess", ' ');
  glutAddMenuEntry("[-] Decrease shininess", ' ');
  glutAddMenuEntry("[w] Toggle wireframe", 'w');
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

/* Callback function for loadRamp */
float diffuseRamp(float x)
{
  if (x > 0.5) {
    return x*x*(3-2*x);
  } else {
    return 0.5f;
  }
}

/* Callback function for loadRamp */
float specularRamp(float x)
{
  if (x > 0.2f) {
    return x;
  } else {
    return 0.0f;
  }
}

/* Callback function for loadRamp */
float edgeRamp(float x)
{
  if (x < 0.2f) {
    return 1.0f;
  } else {
    return 0.85f;
  }
}

/* Create a 1D texture ramp by evaluating func over the range [0,1]. */
void loadRamp(GLuint texobj, int size, float (*func)(float x))
{
  int bytesForRamp = size*sizeof(float);
  float *ramp = malloc(bytesForRamp);
  float *slot = ramp;
  float dx = 1.0 / size;
  float x;
  int i;

  if (NULL == ramp) {
    fprintf(stderr, "%s: memory allocation failed\n", myProgramName);
    exit(1);
  }
  for (i=0, x=0.0, slot=ramp; i<size; i++, x += dx, slot++) {
    float v = func(x);

    *slot = v;
  }

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE                  0x812F  /* Added by OpenGL 1.2 */
#endif

  glBindTexture(GL_TEXTURE_1D, texobj);
  glTexImage1D(GL_TEXTURE_1D, 0, GL_INTENSITY16, size, 0, GL_LUMINANCE, GL_FLOAT, ramp);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
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
                         1.0, 50.0,  /* Znear and Zfar */
                         myProjectionMatrix);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(fieldOfView, aspectRatio,
                 1.0, 50.0);  /* Znear and Zfar */
  glMatrixMode(GL_MODELVIEW);
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

static void makeRotateMatrix(float angle,
                             float ax, float ay, float az,
                             float m[16])
{
  float radians, sine, cosine, ab, bc, ca, tx, ty, tz;
  float axis[3];
  float mag;

  axis[0] = ax;
  axis[1] = ay;
  axis[2] = az;
  mag = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
  if (mag) {
    axis[0] /= mag;
    axis[1] /= mag;
    axis[2] /= mag;
  }

  radians = angle * myPi / 180.0;
  sine = sin(radians);
  cosine = cos(radians);
  ab = axis[0] * axis[1] * (1 - cosine);
  bc = axis[1] * axis[2] * (1 - cosine);
  ca = axis[2] * axis[0] * (1 - cosine);
  tx = axis[0] * axis[0];
  ty = axis[1] * axis[1];
  tz = axis[2] * axis[2];

  m[0]  = tx + cosine * (1 - tx);
  m[1]  = ab + axis[2] * sine;
  m[2]  = ca - axis[1] * sine;
  m[3]  = 0.0f;
  m[4]  = ab - axis[2] * sine;
  m[5]  = ty + cosine * (1 - ty);
  m[6]  = bc + axis[0] * sine;
  m[7]  = 0.0f;
  m[8]  = ca + axis[1] * sine;
  m[9]  = bc - axis[0] * sine;
  m[10] = tz + cosine * (1 - tz);
  m[11] = 0;
  m[12] = 0;
  m[13] = 0;
  m[14] = 0;
  m[15] = 1;
}

static void makeTranslateMatrix(float x, float y, float z, float m[16])
{
  m[0]  = 1;  m[1]  = 0;  m[2]  = 0;  m[3]  = x;
  m[4]  = 0;  m[5]  = 1;  m[6]  = 0;  m[7]  = y;
  m[8]  = 0;  m[9]  = 0;  m[10] = 1;  m[11] = z;
  m[12] = 0;  m[13] = 0;  m[14] = 0;  m[15] = 1;
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

void
loadMVP(const float modelView[16])
{
  float transpose[16];
  int i, j;

  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      transpose[i*4+j] = modelView[j*4+i];
    }
  }
  glLoadMatrixf(transpose);
}

static void display(void)
{
  /* World-space positions for light and eye. */
  const float eyePosition[4] = { 8*sin(eyeAngle), 
                                 eyeHeight,
                                 8*cos(eyeAngle), 1 };
  const float lightPosition[4] = { 2.5*sin(lightAngle), 
                                   lightHeight,
                                   2.5*cos(lightAngle), 1 };

  float translateMatrix[16], rotateMatrix[16],
        modelMatrix[16], viewMatrix[16],
        modelViewMatrix[16], modelViewProjMatrix[16];

  buildLookAtMatrix(eyePosition[0], eyePosition[1], eyePosition[2],
                    0, 0, 0,
                    0, 1, 0,
                    viewMatrix);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  cgGLBindProgram(myCgVertexProgram);
  checkForCgError("binding vertex program");

  cgGLBindProgram(myCgFragmentProgram);
  checkForCgError("binding fragment program");

  /* modelView = rotateMatrix * translateMatrix */
  makeRotateMatrix(headSpin, 0, 1, 0, rotateMatrix);
  makeTranslateMatrix(0, 0, 0, translateMatrix);
  multMatrix(modelMatrix, translateMatrix, rotateMatrix);

  /* Set world-space eye position. */
  cgSetParameter3fv(myCgVertexParam_eyePosition, eyePosition);
  cgSetParameter3fv(myCgVertexParam_lightPosition, lightPosition);

  /* modelViewMatrix = viewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, viewMatrix, modelMatrix);

  /* modelViewProj = projectionMatrix * modelViewMatrix */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, modelViewMatrix);

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgUpdateProgramParameters(myCgVertexProgram);
  drawMonkeyHead();

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  glPushMatrix();
    /* glLoadMatrixf expects a column-major matrix but Cg matrices are
       row-major (matching C/C++ arrays) so used loadMVP to transpose
       the Cg version. */
    loadMVP(modelViewMatrix);
    glTranslatef(lightPosition[0], lightPosition[1], lightPosition[2]);
    glColor3f(1,1,0); /* yellow */
    glutSolidSphere(0.05, 10, 10);  /* sphere to represent light position */
    glColor3f(1,1,1); /* reset back to white */
  glPopMatrix();

  glutSwapBuffers();
}

/* Spin the monkey's head when animating. */
static void idle(void)
{
  headSpin -= 0.5;
  if (headSpin < -360) {
    headSpin += 360;
  }

  glutPostRedisplay();
}

static void keyboard(unsigned char c, int x, int y)
{
  static int animating = 0;
  static int wireframe = 0;

  switch (c) {
  case ' ':
    animating = !animating; /* Toggle */
    if (animating) {
      glutIdleFunc(idle);
    } else {
      glutIdleFunc(NULL);
    }    
    break;
  case '+':
    shininess *= 1.05;
    printf("shininess = %f\n", shininess);
    cgSetParameter1f(myCgVertexParam_shininess, shininess);
    glutPostRedisplay();
    break;
  case '-':
    shininess /= 1.05;
    printf("shininess = %f\n", shininess);
    cgSetParameter1f(myCgVertexParam_shininess, shininess);
    glutPostRedisplay();
    break;
  case 'w':
    wireframe = !wireframe; /* Toggle */
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

/* Use a motion and mouse GLUT callback to allow the viewer and light to
   rotate around the monkey head and move the viewer up and down. */

static int beginx, beginy;
static int moving = 0;
static int movingLight = 0;
static int xLightBegin, yLightBegin;

void
motion(int x, int y)
{
  const float heightBound = 8;

  if (moving) {
    eyeAngle += 0.005*(beginx - x);
    eyeHeight += 0.01*(y - beginy);
    if (eyeHeight > heightBound) {
      eyeHeight = heightBound;
    }
    if (eyeHeight < -heightBound) {
      eyeHeight = -heightBound;
    }
    beginx = x;
    beginy = y;
    glutPostRedisplay();
  }
  if (movingLight) {
    lightAngle += 0.005*(x - xLightBegin);
    lightHeight += 0.03*(yLightBegin - y);
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
