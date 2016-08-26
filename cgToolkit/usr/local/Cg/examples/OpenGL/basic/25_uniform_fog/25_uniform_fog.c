
/* 25_uniform_fog.c - OpenGL-based example demonstrating uniform
   exponential fog based on radial vertex distance using Cg programs from
   Chapter 9 of "The Cg Tutorial" (Addison-Wesley, ISBN 0321194969). */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   1.5 or higher). */

#include <assert.h>   /* for assert */
#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sin and cos */

#include <GL/glew.h>

#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>

#include "matrix.h"

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgFragmentProfile;
static CGprogram   myCgVertexProgram,
                   myCgFragmentProgram;
static CGparameter myCgVertexParam_modelViewProj,
                   myCgVertexParam_modelView,
                   myCgVertexParam_fogDensity,
                   myCgFragmentParam_fogColor,
                   myCgFragmentParam_decal;

static const char *myProgramName = "25_uniform_fog",
                  *myVertexProgramFileName = "C9E2v_fog.cg",
/* Page 240 */    *myVertexProgramName = "C9E2v_fog",
                  *myFragmentProgramFileName = "C9E1f_fog.cg",
/* Page 240 */    *myFragmentProgramName = "C9E1f_fog";

static float myProjectionMatrix[16];

static float eyeHeight = 30.0f; /* Vertical height of light. */
static float eyeAngle  = 0.53f;    /* Angle in radians eye rotates around knight. */

static float fogDensity = 0.08f;
static float fogColor[3] = { 0.8, 0.9, 0.8 };  /* Green-ish gray */
static int city_height_mode = 0;

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
static void menu(int item);
static void keyboard(unsigned char c, int x, int y);
static void mouse(int button, int state, int x, int y);
static void motion(int x, int y);

/* Other forward declared functions. */
static void loadDecalFromDDS(const char *filename);

/* Use enum to assign unique symbolic OpenGL texture names. */
enum {
  TO_BOGUS = 0,
  TO_SIDES,
  TO_ROOF,
  TO_PAVEMENT,
};

int supports_texture_anisotropy = 0;

int main(int argc, char **argv)
{
  glutInitWindowSize(800, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  /* OpenGL 1.3 incorporated compressed textures (based on the
     ARB_texture_compression extension.  However we also need to
     be sure the EXT_texture_compression_s3tc extension is present
     because that supports the specific DXT1 format we expect the
     DDS file to contain. */

  /* Initialize OpenGL entry points. */
  if (glewInit()!=GLEW_OK || !GLEW_VERSION_1_3 || !GLEW_EXT_texture_compression_s3tc) {
    fprintf(stderr, "%s: failed to initialize GLEW, OpenGL 1.3 and GL_EXT_texture_compression_s3tc required.\n", myProgramName);    
    exit(1);
  }

  /* Clear to fog color */
  glClearColor(fogColor[0], fogColor[1], fogColor[2], 1.0); 
  glEnable(GL_DEPTH_TEST);
  supports_texture_anisotropy = glutExtensionSupported("GL_EXT_texture_filter_anisotropic");

  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgGLSetDebugMode(CG_FALSE);
  //cgGLSetManageTextureParameters(myCgContext, CG_TRUE);
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

#define GET_PARAM(name) \
  myCgVertexParam_##name = \
    cgGetNamedParameter(myCgVertexProgram, #name); \
  checkForCgError("could not get " #name " parameter");

  GET_PARAM(fogDensity);
  GET_PARAM(modelViewProj);
  GET_PARAM(modelView);
  cgSetParameter1f(myCgVertexParam_fogDensity, fogDensity);

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

  myCgFragmentParam_fogColor =
    cgGetNamedParameter(myCgFragmentProgram, "fogColor");
  checkForCgError("could not get fogColor parameter");
  cgSetParameter3fv(myCgFragmentParam_fogColor, fogColor);

  myCgFragmentParam_decal =
    cgGetNamedParameter(myCgFragmentProgram, "decal");
  checkForCgError("getting decal parameter");
  cgGLSetTextureParameter(myCgFragmentParam_decal, TO_SIDES);
  checkForCgError("setting decal 2D texture");

#define GL_TEXTURE_MAX_ANISOTROPY_EXT     0x84FE
#define GL_CLAMP_TO_EDGE                  0x812F

  /* Load the decal map the fragment program will sample. */
  glBindTexture(GL_TEXTURE_2D, TO_SIDES);
  loadDecalFromDDS("BuildingWindows.dds");
  if (supports_texture_anisotropy) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0);
  }

  glBindTexture(GL_TEXTURE_2D, TO_ROOF);
  loadDecalFromDDS("BuildingRoof.dds");
  if (supports_texture_anisotropy) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0);
  }
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  
  glBindTexture(GL_TEXTURE_2D, TO_PAVEMENT);
  loadDecalFromDDS("Pavement.dds");
  if (supports_texture_anisotropy) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0);
  }

  /* Create GLUT menu. */
  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Cycle city configuration", ' ');
  glutAddMenuEntry("[+] Increase fog density", ' ');
  glutAddMenuEntry("[-] Decrease fog density", ' ');
  glutAddMenuEntry("[w] Toggle wireframe", 'w');
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  glutMainLoop();
  return 0;
}

static void reshape(int width, int height)
{
  double aspectRatio = (float) width / (float) height;
  double fieldOfView = 40.0; /* Degrees */

  /* Build projection matrix once. */
  makePerspectiveMatrix(fieldOfView, aspectRatio,
                        1.0, 500.0,  /* Znear and Zfar */
                        myProjectionMatrix);
  glViewport(0, 0, width, height);
}

/** Simple image loaders for DirectX's DirectDraw Surface (DDS) format **/

/* Structure matching the Microsoft's "DDS File Reference" documentation. */
typedef struct {
  int magic; /* must be "DDS\0" */
  int size; /* must be 124 */
  int flags;
  int height;
  int width;
  int pitchOrLinearSize;
  int depth;
  int mipMapCount;
  int reserved[11];
  struct {
    int size;
    int flags;
    int fourCC;
    int bitsPerPixel;
    int redMask;
    int greenMask;
    int blueMask;
    int alphaMask;
  } pixelFormat;
  struct {
    int caps;
    int caps2;
    int caps3;
    int caps4;
  } caps;
  int reserved2[1];
} DDS_file_header;

/* Compile time assertions */
#define ct_assert(b)         ct_assert_i(b,__LINE__)
#define ct_assert_i(b,line)  ct_assert_ii(b,line)
#define ct_assert_ii(b,line) void compile_time_assertion_failed_in_line_##line(int _compile_time_assertion_failed_in_line_##line[(b) ? 1 : -1])

ct_assert(sizeof(DDS_file_header) == 128);

/* assume _WIN32 (Windows) is always little-endian */
#if defined(__LITTLE_ENDIAN__) || defined(_WIN32)
/* target is already little endian so no swapping is needed to read little-endian data */
#else
static const unsigned int nativeIntOrder = 0x03020100;
#define LE_INT32_BYTE_OFFSET(a) (((unsigned char*)&nativeIntOrder)[a])
#endif

/* The DDS file format is little-endian so we need to byte-swap
   its 32-bit header words to work on big-endian architectures. */
static int int_le2native(int v)
{
/* Works even if little-endian target and __LITTLE_ENDIAN__ not defined. */
#if defined(__LITTLE_ENDIAN__) || defined(_WIN32)
  return v;
#else
  union {
    int i;
    unsigned char b[4];
  } src, dst;

  src.i = v;
  dst.b[0] = src.b[LE_INT32_BYTE_OFFSET(0)];
  dst.b[1] = src.b[LE_INT32_BYTE_OFFSET(1)];
  dst.b[2] = src.b[LE_INT32_BYTE_OFFSET(2)];
  dst.b[3] = src.b[LE_INT32_BYTE_OFFSET(3)];
  return dst.i;
#endif
}

/* This is a "good enough" loader for DDS 2D decals compressed in the DXT1 format. */
void loadDecalFromDDS(const char *filename)
{
  FILE *file = fopen(filename, "rb");
  long size;
  void *data;
  char *beginning, *image;
  int *words;
  size_t bytes;
  DDS_file_header *header;
  int i, level;

  if (!file) {
    fprintf(stderr, "%s: could not open decal %s\n", myProgramName, filename);
    exit(1);
  }

  fseek(file, 0L, SEEK_END);
  size = ftell(file);
  if (size < 0) {
    fprintf(stderr, "%s: ftell failed\n", myProgramName);
    exit(1);
  }
  fseek(file, 0L, SEEK_SET);
  data = (char*) malloc((size_t)(size));
  if (data == NULL) {
    fprintf(stderr, "%s: malloc failed\n", myProgramName);
    exit(1);
  }
  bytes = fread(data, 1, (size_t)(size), file);
  fclose(file);

  if (bytes < sizeof(DDS_file_header)) {
    fprintf(stderr, "%s: DDS header to short for %s\n", myProgramName, filename);
    exit(1);
  }

  /* Byte swap the words of the header if needed. */
  for (words = data, i=0; i<sizeof(DDS_file_header)/sizeof(int); i++) {
    words[i] = int_le2native(words[i]);
  }

#define FOURCC(a) ((a[0]) | (a[1] << 8) | (a[2] << 16) | (a[3] << 24))
#define EXPECT(f,v) \
  if ((f) != (v)) { \
    fprintf(stderr, "%s: field %s mismatch (got 0x%x, expected 0x%x)\n", \
      myProgramName, #f, (f), (v)); exit(1); \
  }

  /* Examine the header to make sure it is what we expect. */
  header = data;
  EXPECT(header->magic, FOURCC("DDS "));

#define DDSD_CAPS               0x00000001  /* caps field is valid */
#define DDSD_HEIGHT             0x00000002  /* height field is valid */
#define DDSD_WIDTH              0x00000004  /* width field is valid */
#define DDSD_PIXELFORMAT        0x00001000  /* pixelFormat field is valid */
#define DDSD_MIPMAPCOUNT        0x00020000  /* mipMapCount field is valid */

#define DDSD_NEEDED (DDSD_CAPS | DDSD_WIDTH | DDSD_HEIGHT | \
                     DDSD_PIXELFORMAT | DDSD_MIPMAPCOUNT)

  EXPECT(header->flags & DDSD_NEEDED, DDSD_NEEDED);
  EXPECT(header->size, 124);
  EXPECT(header->depth, 0);
  EXPECT(header->pixelFormat.size, 32);  /* 32 bytes in a DXT1 block */
  EXPECT(header->pixelFormat.fourCC, FOURCC("DXT1"));
  EXPECT(header->caps.caps2, 0);

  beginning = data;
  image = (char*) &header[1];
  {
    int levels = header->mipMapCount;
    int width = header->width;
    int height = header->height;
    const int border = 0;

    /* For each mipmap level... */
    for (level=0; level<levels; level++) {
      /* DXT1 has contains two 16-bit (565) colors and a 2-bit field for
         each of the 16 texels in a given 4x4 block.  That's 64 bits
         per block or 8 bytes. */
      const int bytesPer4x4Block = 8;
      GLsizei imageSizeInBytes = ((width+3)>>2)*((height+3)>>2) * bytesPer4x4Block;
      size_t offsetInToRead = image + imageSizeInBytes - beginning;

      if (offsetInToRead > bytes) {
        fprintf(stderr, "%s: DDS images over read the data!\n", myProgramName);
        exit(1);
      }
      glCompressedTexImage2D(GL_TEXTURE_2D, level,
        GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
        width, height, border, imageSizeInBytes, image);
      image += imageSizeInBytes;

      /* Half the width and height either iteration, but do not allow
         the width or height to become less than 1. */
      width = width >> 1;
      if (width < 1) {
        width = 1;
      }
      height = height >> 1;
      if (height < 1) {
        height = 1;
      }
    }
  }
  assert(image <= beginning + bytes);

  /* Configure texture parameters reasonably. */
  if (header->mipMapCount > 1) {
    /* Clamp the range of levels to however levels the DDS file actually has.
       If the DDS file has less than a full mipmap chain all the way down,
       this allows OpenGL to still use the texture. */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, header->mipMapCount-1);
    /* Use better trilinear mipmap minification filter instead of the default. */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  } else {
    /* OpenGL's default minification filter (GL_NEAREST_MIPMAP_LINEAR) requires
       mipmaps this DDS file does not have so switch to a linear filter that
       doesn't require mipmaps. */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  }
  /* Fine to leave a decal using the default wrap modes (GL_REPEAT). */
  free(data);
}

#define CITY_COLS 22
#define CITY_ROWS 22

static float random0to1(void) {
  return ((float) rand()) / RAND_MAX;
}

static void drawBuilding(float x, float y, float height)
{
  static GLfloat roof_coords[4][2] = { { 1,1 }, { 1,0 }, { 0,0 }, { 0, 1 } };
  static GLfloat streetColor[3] = {1, 1, 1};
  static GLfloat buildingTopColor[3] = {0.8, 0.8, 0.8};
  const float streetLevel = 0.0;
  const float buildingTopTexCoord = 1 + floor(height/2);
  const float deltaHeight = height / buildingTopTexCoord;
  const int roof_config = rand() % 4;
  GLfloat buildingColor[3];
  float tex0, tex1, height0, height1;

  buildingColor[0] = random0to1()*0.2f+0.8f;
  buildingColor[1] = random0to1()*0.2f+0.8f;
  buildingColor[2] = random0to1()*0.2f+0.8f;

  glBindTexture(GL_TEXTURE_2D, TO_PAVEMENT);
  glColor3fv(streetColor);
  glBegin(GL_TRIANGLE_FAN);
    glTexCoord2f(1.0/4.0, 1.0/4.0);
    glVertex3f(x + 1.0, streetLevel, y + 1.0);

    glTexCoord2f(4.0/4.0, 1.0/4.0);
    glVertex3f(x + 4.0, streetLevel, y + 1.0);

    glTexCoord2f(4.0/4.0, 0.0/4.0);
    glVertex3f(x + 4.0, streetLevel, y + 0.0);

    glTexCoord2f(0.0/4.0, 0.0/4.0);
    glVertex3f(x + 0.0, streetLevel, y + 0.0);

    glTexCoord2f(0.0/4.0, 4.0/4.0);
    glVertex3f(x + 0.0, streetLevel, y + 4.0);

    glTexCoord2f(1.0/4.0, 4.0/4.0);
    glVertex3f(x + 1.0, streetLevel, y + 4.0);
  glEnd();

  /* Orient the roof texture with a random 90 degree rotation 
     to make roofs slightly varied. */
  glBindTexture(GL_TEXTURE_2D, TO_ROOF);
  glBegin(GL_QUADS);
    glColor3fv(buildingTopColor);
    glTexCoord2fv(roof_coords[(roof_config + 0) % 4]);
    glVertex3f(x + 4.0, height, y + 4.0);
    glTexCoord2fv(roof_coords[(roof_config + 1) % 4]);
    glVertex3f(x + 4.0, height, y + 1.0);
    glTexCoord2fv(roof_coords[(roof_config + 2) % 4]);
    glVertex3f(x + 1.0, height, y + 1.0);
    glTexCoord2fv(roof_coords[(roof_config + 3) % 4]);
    glVertex3f(x + 1.0, height, y + 4.0);
  glEnd();

  glBindTexture(GL_TEXTURE_2D, TO_SIDES);

  glColor3fv(buildingColor);

  /* Draw tessellated sides of the building. */
  glBegin(GL_QUADS);
  for (tex0=0, tex1=1, height0=0, height1=deltaHeight;
       tex0<buildingTopTexCoord;
       tex0++, tex1++, height0+=deltaHeight, height1+=deltaHeight) {
    glTexCoord2f(0,tex0);
    glVertex3f(x + 1.0, height0, y + 1.0);
    glTexCoord2f(0,tex1);
    glVertex3f(x + 1.0, height1, y + 1.0);
    glTexCoord2f(1,tex1);
    glVertex3f(x + 4.0, height1, y + 1.0);
    glTexCoord2f(1,tex0);
    glVertex3f(x + 4.0, height0, y + 1.0);

    glTexCoord2f(1,tex0);
    glVertex3f(x + 1.0, height0, y + 4.0);
    glTexCoord2f(1,tex1);
    glVertex3f(x + 1.0, height1, y + 4.0);
    glTexCoord2f(0,tex1);
    glVertex3f(x + 1.0, height1, y + 1.0);
    glTexCoord2f(0,tex0);
    glVertex3f(x + 1.0, height0, y + 1.0);

    glTexCoord2f(0,tex0);
    glVertex3f(x + 4.0, height0, y + 1.0);
    glTexCoord2f(0,tex1);
    glVertex3f(x + 4.0, height1, y + 1.0);
    glTexCoord2f(1,tex1);
    glVertex3f(x + 4.0, height1, y + 4.0);
    glTexCoord2f(1,tex0);
    glVertex3f(x + 4.0, height0, y + 4.0);

    glTexCoord2f(1,tex0);
    glVertex3f(x + 4.0, height0, y + 4.0);
    glTexCoord2f(1,tex1);
    glVertex3f(x + 4.0, height1, y + 4.0);
    glTexCoord2f(0,tex1);
    glVertex3f(x + 1.0, height1, y + 4.0);
    glTexCoord2f(0,tex0);
    glVertex3f(x + 1.0, height0, y + 4.0);
  }
  glEnd();
}

static void
drawCity(int heightMode)
{
  int i, j;

  /* Re-seed the random number generator every render to make city deterministic. */
  srand(333);
  for (i=0; i<CITY_COLS; i++) {
    for (j=0; j<CITY_ROWS; j++) {
      float x, y, height = 0.0;

      x = 4.0 * i - (2*CITY_COLS+0.5);
      y = 4.0 * j - (2*CITY_ROWS+0.5);

      switch (heightMode) {
      case 0:  /* Default building arrangement; buildings randomly taller in the distance. */
        height = 0.1 * (x*x + y*y) * random0to1() + 0.3 * sqrt(x*x + y*y) * random0to1() + 1.0;
        break;
      case 1:  /* Constant height buildings. */
        height = 3.0;
        break;
      case 2:  /* Buildings taller in the distance. */
        height = 0.05 * (x*x + y*y) + 0.15 * sqrt(x*x + y*y) + 1.0;
        break;
      case 3:  /* Buildings taller as magnitude x (but not y) increases. */
        height = 0.05 * (x*x) + 0.15 * sqrt(x*x) + 1.0;
        break;
      }

      drawBuilding(x, y, height);
    }
  }
}

static void display(void)
{
  /* World-space positions for light and eye. */
  const float eyeRadius = 7.0f;
  const float eyePosition[4] = { eyeRadius*sin(eyeAngle), 
                                 eyeHeight,
                                 eyeRadius*cos(eyeAngle), 1 };

  float translateMatrix[16], rotateMatrix[16],
        modelMatrix[16], viewMatrix[16],
        modelViewMatrix[16], modelViewProjMatrix[16];

  makeLookAtMatrix(eyePosition[0], eyePosition[1], eyePosition[2],
                   0, 20, 0,
                   0, 1, 0,
                   viewMatrix);

  makeRotateMatrix(0, 0, 1, 0, rotateMatrix);
  makeTranslateMatrix(0, 0, 0, translateMatrix);
  multMatrix(modelMatrix, translateMatrix, rotateMatrix);

  /* modelViewMatrix = viewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, viewMatrix, modelMatrix);

  /* modelViewProj = projectionMatrix * modelViewMatrix */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, modelViewMatrix);

  /* modelViewMatrix = viewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, viewMatrix, modelMatrix);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  cgGLBindProgram(myCgVertexProgram);
  checkForCgError("binding vertex program");

  cgGLEnableProfile(myCgVertexProfile);
  checkForCgError("enabling vertex profile");

  cgGLBindProgram(myCgFragmentProgram);
  checkForCgError("binding fragment program");

  cgGLEnableProfile(myCgFragmentProfile);
  checkForCgError("enabling fragment profile");

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgSetMatrixParameterfr(myCgVertexParam_modelView, modelViewMatrix);
  cgUpdateProgramParameters(myCgVertexProgram);
  drawCity(city_height_mode);

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  glutSwapBuffers();
}

/* Use a motion and mouse GLUT callback to allow the viewer to
   rotate around the monkey head and move the viewer up and down. */

static int beginx, beginy;
static int moving = 0;

static void motion(int x, int y)
{
  const float minHeight = 15,
              maxHeight = 95;

  if (moving) {
    eyeAngle += 0.01*(beginx - x);
    eyeHeight += 0.04*(beginy - y);
    if (eyeHeight > maxHeight) {
      eyeHeight = maxHeight;
    }
    if (eyeHeight < minHeight) {
      eyeHeight = minHeight;
    }
    beginx = x;
    beginy = y;
    glutPostRedisplay();
  }
}

static void mouse(int button, int state, int x, int y)
{
  const int spinButton = GLUT_LEFT_BUTTON;

  if (button == spinButton && state == GLUT_DOWN) {
    moving = 1;
    beginx = x;
    beginy = y;
  }
  if (button == spinButton && state == GLUT_UP) {
    moving = 0;
  }
}

static void keyboard(unsigned char c, int x, int y)
{
  static int wireframe = 0;

  switch (c) {
  case '-':
  case 'f':
    fogDensity /= 1.5;
    if (fogDensity < 0) {
      fogDensity = 0;
    }
    cgSetParameter1f(myCgVertexParam_fogDensity, fogDensity);
    printf("fogDensity = %f\n", fogDensity);
    glutPostRedisplay();
    break;
  case '+':
  case 'g':
    fogDensity *= 1.5f;
    printf("fogDensity = %f\n", fogDensity);
    cgSetParameter1f(myCgVertexParam_fogDensity, fogDensity);
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
  case ' ':
    city_height_mode = (city_height_mode+1) % 4;
    glutPostRedisplay();
    break;
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

static void menu(int item)
{
  /* Pass menu item character code to keyboard callback. */
  keyboard((unsigned char)item, 0, 0);
}
