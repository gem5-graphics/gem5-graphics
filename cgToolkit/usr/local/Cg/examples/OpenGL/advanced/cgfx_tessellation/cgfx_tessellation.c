
/* cgfx_tessellation - tessellation implemented with CgFX  */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <GL/glew.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed? */
#include <Cg/cgGL.h>

#include "showfps.h"
#include "zpr.h"

/* RGB8 image data for 128x128 demon texture */
static const GLubyte myDemonTextureImage[3*(128*128)] = {
#include "demon_image.h"
};

/* OpenGL texture object handle. */
enum {
  DEMON_TEXTURE = 1
};

extern const int teapotPatch[32][16];
extern double teapotVertex[306][3];

static CGcontext   myCgContext;
static CGeffect    myCgEffect;
static CGtechnique myCgTechnique = 0;
static CGparameter myCgParameterInnerTess, myCgParameterOuterTess, myCgParameterModelView, myCgParameterProjection;
static CGparameter myCgParameterTexture;

static const char *myProgramName = "cgfx_tessellation",
                  *myEffectFileName = "cgfx_tessellation.cgfx";

int   innerTess[2]   = { 4, 4 };
int   outerTess[4]   = { 4, 4, 4, 4 };
const float outerTessStep = 1;
const float outerTessMin  = 1;
      float outerTessMax  = 64;

/* ZPR module controls viewing */
float modelView[16];        /* 4x4 modelView matrix     */
float projection[16];       /* 4x4 projection matrix    */

/* Interaction configuration */
int spin = 0;
int spinTime = 0;

/* Cg version information */
const char *versionStr = NULL;
int         versionMajor = 0;
int         versionMinor = 0;

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

static void nextTechnique()
{
  CGtechnique start = myCgTechnique;

  if (myCgTechnique)
    myCgTechnique = cgGetNextTechnique(myCgTechnique);

  if (!myCgTechnique)
    myCgTechnique = cgGetFirstTechnique(myCgEffect);

  while (myCgTechnique && cgValidateTechnique(myCgTechnique) == CG_FALSE) {
    fprintf(stderr, "%s: Technique %s did not validate.  Skipping.\n",
      myProgramName, cgGetTechniqueName(myCgTechnique));
    myCgTechnique = cgGetNextTechnique(myCgTechnique);
    if (myCgTechnique==start)
    {
      fprintf(stderr, "%s: No valid technique\n", myProgramName);
      return;
    }
    if (!myCgTechnique)
      myCgTechnique = cgGetFirstTechnique(myCgEffect);
  }

  if (myCgTechnique) {
    fprintf(stderr, "%s: Using technique %s.\n",
      myProgramName, cgGetTechniqueName(myCgTechnique));
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

  glMatrixMode(GL_MODELVIEW_MATRIX);
  glScalef(0.25,0.25,0.25);
  zprInit();

  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(special);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_5) {
    fprintf(stderr, "%s: Failed to initialize GLEW. OpenGL 1.5 required.\n", myProgramName);
    exit(1);
  }

  if (!GLEW_NV_gpu_program5) {
    fprintf(stderr, "%s: NV_gpu_program5 not available.\n", myProgramName);
  }

  if (!GLEW_ARB_tessellation_shader) {
    fprintf(stderr, "%s: ARB_tessellation_shader not available.\n", myProgramName);
  }

  /* Cg 3.0 is required for tessellation support */
  versionStr = cgGetString(CG_VERSION);
  if (!versionStr || sscanf(versionStr, "%d.%d", &versionMajor, &versionMinor)!=2 || versionMajor<3) {
    fprintf(stderr, "%s: Cg 3.0 required, %s detected.\n", myProgramName, versionStr);
    exit(1);
  }

  glClearColor(0.8, 0.8, 0.8, 0.0);  /* Gray background */

  myCgContext = cgCreateContext();
  checkForCgError("creating context");

  cgGLRegisterStates(myCgContext);
  checkForCgError("registering standard CgFX states");
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
  checkForCgError("manage texture parameters");

  myCgEffect = cgCreateEffectFromFile( myCgContext, myEffectFileName, NULL );
  checkForCgError("creating cgfx_tessellation.cgfx effect");
  assert(myCgEffect);

  myCgParameterInnerTess = cgGetNamedEffectParameter(myCgEffect, "innerTess");
  checkForCgError("could not get innerTess parameter");
  myCgParameterOuterTess = cgGetNamedEffectParameter(myCgEffect, "outerTess");
  checkForCgError("could not get outerTess parameter");
  myCgParameterModelView = cgGetNamedEffectParameter(myCgEffect, "modelView");
  checkForCgError("could not get modelView parameter");
  myCgParameterProjection = cgGetNamedEffectParameter(myCgEffect, "projection");
  checkForCgError("could not get projection parameter");

  myCgParameterTexture = cgGetNamedEffectParameter(myCgEffect, "texture");
  checkForCgError("could not get texture parameter");

  nextTechnique();

  /* Query GL for edge tessellation limit */
  glGetFloatv(GL_MAX_TESS_GEN_LEVEL,&outerTessMax);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  glEnable(GL_CULL_FACE);
  glDisable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); /* Tightly packed texture data. */
  glBindTexture(GL_TEXTURE_2D, DEMON_TEXTURE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 128, 128, 0, GL_RGB, GL_UNSIGNED_BYTE, myDemonTextureImage);
  cgGLSetTextureParameter(myCgParameterTexture, DEMON_TEXTURE);
  cgSetSamplerState(myCgParameterTexture);

  checkForCgError("setting source image texture");

  colorFPS(0.0,0.0,0.0);
  enableFPS();

  glutMainLoop();
  return 0;
}

static void drawBezierPatch4x4(void)
{
  int i, j;
  if (GLEW_ARB_tessellation_shader) {
    glPatchParameteri(GL_PATCH_VERTICES, 16);
    glBegin(GL_PATCHES);
    for (i=0; i<32; ++i)
      for (j=0; j<16; ++j)
        glVertex3dv(teapotVertex[teapotPatch[i][j]]);
    glEnd();
  }
}

static void drawBezierPatch4x4Points(void)
{
  int i, j;
  glBegin(GL_POINTS);
  for (i=0; i<32; ++i)
    for (j=0; j<16; ++j)
      glVertex3dv(teapotVertex[teapotPatch[i][j]]);
  glEnd();
}

static void drawBezierPatch4x4LineLoop(void)
{
  int i;
  for (i=0; i<32; ++i)
  {
    glBegin(GL_LINE_LOOP);
    glVertex3dv(teapotVertex[teapotPatch[i][0]]);
    glVertex3dv(teapotVertex[teapotPatch[i][3]]);
    glVertex3dv(teapotVertex[teapotPatch[i][15]]);
    glVertex3dv(teapotVertex[teapotPatch[i][12]]);
    glEnd();
  }
}

static void display(void)
{
  CGpass pass;
  CGannotation annotation;
  const GLubyte *i;
  const char *value;

  /* Update modelView matrix, if spinning */
  if (spin) {
    const int time = glutGet(GLUT_ELAPSED_TIME);
    const float spinAngle = (time-spinTime)/50.0;
    spinTime = time;
    glRotatef(spinAngle,0,0,1);
  }

  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
  glGetFloatv(GL_PROJECTION_MATRIX, projection);

  cgSetParameter2iv(myCgParameterInnerTess, innerTess);
  cgSetParameter4iv(myCgParameterOuterTess, outerTess);
  cgSetMatrixParameterfc(myCgParameterModelView, modelView);
  cgSetMatrixParameterfc(myCgParameterProjection, projection);

  // Set modelview and projection for CgFX purposes

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);

  pass = cgGetFirstPass( myCgTechnique );
  while( pass )
  {
    cgSetPassState( pass );

    annotation = cgGetNamedPassAnnotation(pass, "geometry");
    if (cgIsAnnotation(annotation))
    {
       value = cgGetStringAnnotationValue(annotation);
       if (!strcmp(value,"drawBezierPatch4x4"))         drawBezierPatch4x4();
       if (!strcmp(value,"drawBezierPatch4x4Points"))   drawBezierPatch4x4Points();
       if (!strcmp(value,"drawBezierPatch4x4LineLoop")) drawBezierPatch4x4LineLoop();
    }

    cgResetPassState( pass );
    pass = cgGetNextPass( pass );
  }

  // Set modelview and projection for fixed-function purposes

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  handleFPS();

  i = (const GLubyte *) cgGetTechniqueName(myCgTechnique);
  if (i && *i)
  {
    glPushMatrix();
      glLoadIdentity();
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
        glLoadIdentity();
        glOrtho(0, 1, 0, 1, -1, 1);
        glDisable(GL_DEPTH_TEST);
        glRasterPos2f(0,0);
        glBitmap(0, 0, 0, 0, 9, 15, i);
        for (; *i; i++)
          glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *i);
      glEnable(GL_DEPTH_TEST);
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
  }

  glutSwapBuffers();

  if (spin)
    glutPostRedisplay();
}

static void keyboard(unsigned char c, int x, int y)
{
  int i;

  switch (c) {
    case 't': nextTechnique();                                              break;
    case ' ': spin = !spin;          spinTime = glutGet(GLUT_ELAPSED_TIME); break;

    case '+': innerTess[0]++; innerTess[1]++; break;
    case '-': innerTess[0]--; innerTess[1]--; break;

    case '[': outerTess[0] -= outerTessStep; outerTess[2] -= outerTessStep; break;
    case ']': outerTess[0] += outerTessStep; outerTess[2] += outerTessStep; break;
    case '{': outerTess[1] -= outerTessStep; outerTess[3] -= outerTessStep; break;
    case '}': outerTess[1] += outerTessStep; outerTess[3] += outerTessStep; break;

    case 27:  /* Esc key */
      /* Demonstrate proper deallocation of Cg runtime data structures.
         Not strictly necessary if we are simply going to exit. */
      cgDestroyEffect(myCgEffect);
      cgDestroyContext(myCgContext);
      exit(0);
      break;
  }

  /* Clamp innerTess to valid range */
  for (i=0; i<2; ++i)
    if (innerTess[i]<2)
      innerTess[i] = 2;

  /* Clamp outerTess to valid range */
  for (i=0; i<4; ++i) {
    if (outerTess[i]<outerTessMin) outerTess[i] = outerTessMin;
    if (outerTess[i]>outerTessMax) outerTess[i] = outerTessMax;
  }

  glutPostRedisplay();
}

static void special(int key, int x, int y)
{
  glutPostRedisplay();
}
