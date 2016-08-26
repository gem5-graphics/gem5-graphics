
/* cgfx_texture_array - texture arrays implemented with CgFX  */

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

static CGcontext   myCgContext;
static CGeffect    myCgEffect;
static CGparameter myCgEffect_myLayerOffset;
static CGtechnique myCgTechnique;

static const char *myProgramName = "cgfx_texture_array",

                  *myEffectFileName = "texture_array.cgfx";

static int myLayerCount = 13,
           myStarCount = 0,
           myLayerOffset = 0;

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
static void menu(int item);

enum {
  TO_BOGUS = 0,  /* Skip zero since it is the OpenGL default texture. */
  TO_DECAL_ARRAY
};

static void buildDecalLayers(int w, int h, int layers);
static void useSamplerParameter(CGeffect effect, const char *paramName, GLuint texobj);

int main(int argc, char **argv)
{
  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);

  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_2 || !GLEW_EXT_texture_array) {
    fprintf(stderr, "%s: Failed to initialize GLEW. OpenGL 1.2 and GL_EXT_texture_array required.\n", myProgramName);    
    exit(0);
  }

  glClearColor(0.2, 0.2, 0.2, 0.0);  /* Gray background */

  myCgContext = cgCreateContext();
  checkForCgError("creating context");

  cgGLRegisterStates(myCgContext);
  checkForCgError("registering standard CgFX states");
  cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
  checkForCgError("manage texture parameters");

  myCgEffect = cgCreateEffectFromFile( myCgContext, myEffectFileName, NULL );
  checkForCgError("creating texture_array.cgfx effect");
  assert(myCgEffect);

  myCgTechnique = cgGetFirstTechnique(myCgEffect);
  while (myCgTechnique && cgValidateTechnique(myCgTechnique) == CG_FALSE) {
    fprintf(stderr, "%s: Technique %s did not validate.  Skipping.\n",
      myProgramName, cgGetTechniqueName(myCgTechnique));
    myCgTechnique = cgGetNextTechnique(myCgTechnique);
  }
  if (myCgTechnique) {
    fprintf(stderr, "%s: Initially using technique %s.\n",
      myProgramName, cgGetTechniqueName(myCgTechnique));
  } else {
    fprintf(stderr, "%s: No valid technique\n",
      myProgramName);
    if (glutExtensionSupported("GL_EXT_texture_array")) {
      /* Unexpected. */
      exit(1);
    } else {
      /* Expect no valid technique if texture arrays not supported. */
      exit(0);
    }
  }

  myCgEffect_myLayerOffset = cgGetNamedEffectParameter(myCgEffect, "LayerOffset");
  checkForCgError("could not get myLayerOffset parameter");
  if (myCgEffect_myLayerOffset == 0) {
    printf("%s: %s: %s\n",
      myEffectFileName, "could not get parameter", "LayerOffset");
    exit(1);
  }

  buildDecalLayers(64, 64, myLayerCount);

  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAddMenuEntry("[+] Increment layer offset", '+');
  glutAddMenuEntry("[-] Decrement layer offset", '-');
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

static void useSamplerParameter(CGeffect effect,
                                const char *paramName, GLuint texobj)
{
  CGparameter param = cgGetNamedEffectParameter(effect, paramName);

  if (!param) {
    fprintf(stderr, "%s: expected effect parameter named %s\n",
      myProgramName, paramName);
    exit(1);
  }
  cgGLSetTextureParameter(param, texobj);
  cgSetSamplerState(param);
}

static void buildDecalLayers(int w, int h, int layers)
{
  float wf = w-1, hf = h-1;
  int i, j, k;
  GLubyte *img, *texel;

  img = (GLubyte*)malloc(w * h * layers * 3 * sizeof(GLubyte));
  if (!img) {
    printf("%s: %s\n", myProgramName, "malloc failed");
    exit(1);
  }
  texel = img;
  for (k=0; k<layers; k++) {
    float bias = k * 0.2;
    float scale = ((k % 4) + 2) * 5.1;
    int colorMask = ((k + 3) % 7) + 1;
    for (j=0; j<h; j++) {
      float y = j/hf * 2 - 1;
      for (i=0; i<w; i++) {
        static const float colors[3][3] = { { 0.1, 0.1, 0.1 },
                                            { 0.0, 0.3, 0.6 },
                                            { 0.4, 0.2, 0.5 } };
        float x = i/wf * 2 - 1;
        float dist = sqrt(x*x + y*y);
        float atten = sin(dist * scale + bias) / 3.0f + 0.6f;

        float red1   = (colorMask & 1) ? 0.9 : 0.5,
              green1 = (colorMask & 2) ? 1.0 : 0.6,
              blue1  = (colorMask & 4) ? 0.9 : 0.7;
        float red2   = colors[colorMask % 3][0],
              green2 = colors[colorMask % 3][1],
              blue2  = colors[colorMask % 3][2];

        float red    = red1   + (red2   - red1)   * atten;
        float green  = green1 + (green2 - green1) * atten;
        float blue   = blue1  + (blue2  - blue1)  * atten;

        assert(red >= 0);
        assert(green >= 0);
        assert(blue >= 0);
        assert(red <= 1);
        assert(green <= 1);
        assert(blue <= 1);
        *texel++ = (GLubyte) (red * 255.0f);
        *texel++ = (GLubyte) (green * 255.0f);
        *texel++ = (GLubyte) (blue * 255.0f);
      }
    }
  }
  assert(texel == img + (w * h * layers * 3));
  glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, TO_DECAL_ARRAY);
  glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_GENERATE_MIPMAP, 1); 
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); /* Tightly packed texture data. */
  glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, /*level*/0, GL_RGB8,
    w, h, layers, /*border*/0,
    GL_RGB, GL_UNSIGNED_BYTE, img);
  free(img);

  useSamplerParameter( myCgEffect, "decalArray", TO_DECAL_ARRAY );
}

static void drawStar(float x, float y, float layer,
                     int starPoints, float R, float r)
{
  int i;
  double piOverStarPoints = 3.14159 / starPoints,
         angle = 0.0;
  float cosine, sine, ratio  = r/R;

  glBegin(GL_TRIANGLE_FAN);
    glTexCoord3f(0.5,0.5,layer);
    glVertex2f(x, y);  /* Center of star */
    /* Emit exterior vertices for star's points. */
    for (i=0; i<starPoints; i++) {

      cosine = cos(angle);
      sine   = sin(angle);
      glTexCoord3f(cosine*0.5+0.5,sine*0.5+0.5,layer);
      glVertex2f(x + R*cosine, y + R*sine);
      angle += piOverStarPoints;

      cosine = cos(angle);
      sine   = sin(angle);
      glTexCoord3f(ratio*cosine*0.5+0.5,ratio*sine*0.5+0.5,layer);
      glVertex2f(x + r*cosine, y + r*sine);
      angle += piOverStarPoints;
    }
    /* End by repeating first exterior vertex of star. */
    angle = 0;
    cosine = cos(angle);
    sine   = sin(angle);
    glTexCoord3f(cosine*0.5+0.5,sine*0.5+0.5,layer);
    glVertex2f(x + R*cosine, y + R*sine);
  glEnd();
}

static void drawStars(void)
{
  int layer = 0;

  /*                              star    outer   inner  */
  /*        x      y     layer    Points  radius  radius */
  /*       =====  =====  ======== ======  ======  ====== */
  drawStar(-0.1,   0,    layer++, 5,      0.5,    0.2);
  drawStar(-0.84,  0.1,  layer++, 5,      0.3,    0.12);
  drawStar( 0.72, -0.5,  layer++, 7,      0.25,   0.11);
  drawStar( 0.3,   0.97, layer++, 5,      0.3,    0.1);
  drawStar( 0.94,  0.3,  layer++, 6,      0.5,    0.2);
  drawStar(-0.97, -0.8,  layer++, 5,      0.6,    0.2);
  myStarCount = layer;
}

static int myAnimating = 0;

static void advance(int ignored)
{
  static int animateDirection = 1;

  if (!myAnimating)
    return;  /* Return without rendering or registering another timer func. */

  if (myLayerOffset <= 0) {
    animateDirection = 1;
  } else if (myLayerOffset >= myLayerCount-myStarCount) {
    animateDirection = -1;
  }
  myLayerOffset += animateDirection;
  glutPostRedisplay();
}

static void display(void)
{
  CGpass pass;

  if( myAnimating )
    glutTimerFunc( 150, advance, 0 );

  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  cgSetParameter1i( myCgEffect_myLayerOffset, myLayerOffset );

  pass = cgGetFirstPass( myCgTechnique );
  while( pass ) 
  {
    cgSetPassState( pass );
    
    drawStars();

    cgResetPassState( pass );
    pass = cgGetNextPass( pass );
  }

  glutSwapBuffers();
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
  case '+':
    myLayerOffset++;
    glutPostRedisplay();
    break;
  case '-':
    myLayerOffset--;
    glutPostRedisplay();
    break;
  case ' ':
    myAnimating = !myAnimating; /* Toggle */
    if (myAnimating)
      advance(0);
    break;
  }
}

static void menu(int item)
{
  /* Pass menu item character code to keyboard callback. */
  keyboard((unsigned char)item, 0, 0);
}
