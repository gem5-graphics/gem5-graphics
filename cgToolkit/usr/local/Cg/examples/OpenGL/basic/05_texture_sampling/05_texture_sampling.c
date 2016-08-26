
/* 05_texture_sampling.c - OpenGL-based example using a Cg
   vertex and a Cg fragment programs from Chapter 3 of "The Cg Tutorial"
   (Addison-Wesley, ISBN 0321194969). */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   1.0 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sin and cos */
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgFragmentProfile;
static CGprogram   myCgVertexProgram,
                   myCgFragmentProgram;
static CGparameter myCgFragmentParam_decal;                   

static const char *myProgramName = "05_texture_sampling",
                  *myVertexProgramFileName = "C3E2v_varying.cg",
/* Page 65 */     *myVertexProgramName = "C3E2v_varying",
                  *myFragmentProgramFileName = "C3E3f_texture.cg",
/* Page 67 */     *myFragmentProgramName = "C3E3f_texture";

static const GLubyte
myDemonTextureImage[3*(128*128)] = {
/* RGB8 image data for a mipmapped 128x128 demon texture */
#include "demon_image.h"
};

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

int main(int argc, char **argv)
{
  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);

  glClearColor(0.1, 0.3, 0.6, 0.0);  /* Blue background */

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); /* Tightly packed texture data. */

  glBindTexture(GL_TEXTURE_2D, 666);
  /* Load demon decal texture with mipmaps. */
  gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB8,
    128, 128, GL_RGB, GL_UNSIGNED_BYTE, myDemonTextureImage);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
    GL_LINEAR_MIPMAP_LINEAR);

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

  /* No uniform vertex program parameters expected. */

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

  myCgFragmentParam_decal =
    cgGetNamedParameter(myCgFragmentProgram, "decal");
  checkForCgError("getting decal parameter");

  cgGLSetTextureParameter(myCgFragmentParam_decal, 666);
  checkForCgError("setting decal 2D texture");

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

  cgGLBindProgram(myCgFragmentProgram);
  checkForCgError("binding fragment program");

  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  cgGLEnableTextureParameter(myCgFragmentParam_decal);
  checkForCgError("enable decal texture");

  glBegin(GL_TRIANGLES);
    glTexCoord2f(0, 0);
    glVertex2f(-0.8, 0.8);

    glTexCoord2f(1, 0);
    glVertex2f(0.8, 0.8);

    glTexCoord2f(0.5, 1);
    glVertex2f(0.0, -0.8);
  glEnd();


  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  cgGLDisableTextureParameter(myCgFragmentParam_decal);
  checkForCgError("disabling decal texture");

  glutSwapBuffers();
}

static void keyboard(unsigned char c, int x, int y)
{
  switch (c) {
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
