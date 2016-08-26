
/* 19_cube_map_refraction.c - OpenGL-based refractive environment
   mapping example using Cg programs from Chapter 7 of "The Cg Tutorial"
   (Addison-Wesley, ISBN 0321194969). */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   1.5 or higher). */

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
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
                   myCgVertexParam_eyePositionW,
                   myCgVertexParam_modelToWorld,
                   myCgVertexParam_etaRatio,
                   myCgFragmentParam_transmittance,
                   myCgFragmentParam_decalMap,
                   myCgFragmentParam_environmentMap;

static const char *myProgramName = "19_cube_map_refraction",
                  *myVertexProgramFileName = "C7E3v_refraction.cg",
/* Page 187 */    *myVertexProgramName = "C7E3v_refraction",
                  *myFragmentProgramFileName = "C7E4f_refraction.cg",
/* Page 188 */    *myFragmentProgramName = "C7E4f_refraction";

static float myProjectionMatrix[16];

static float eyeHeight = 0.0f; /* Vertical height of light. */
static float eyeAngle  = 0.53f;    /* Angle in radians eye rotates around knight. */

static float headSpin = 0.0f;  /* Head spin in degrees. */
static float transmittance = 0.6f;
static float etaRatio = 1.5f;  /* Index of refraction. */

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
static void loadDecalFromDDS(const char *filename);
static void loadCubeMapFromDDS(const char *filename);

/* Use enum to assign unique symbolic OpenGL texture names. */
enum {
  TO_BOGUS = 0,
  TO_DECAL,
  TO_ENVIRONMENT,
};

int main(int argc, char **argv)
{
  glutInitWindowSize(400, 400);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInit(&argc, argv);

  glutCreateWindow(myProgramName);
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

#define GET_PARAM(name) \
  myCgVertexParam_##name = \
    cgGetNamedParameter(myCgVertexProgram, #name); \
  checkForCgError("could not get " #name " parameter");

  GET_PARAM(modelViewProj);
  GET_PARAM(eyePositionW);
  GET_PARAM(modelToWorld);

  myCgVertexParam_etaRatio =
    cgGetNamedParameter(myCgVertexProgram, "etaRatio");
  checkForCgError("could not get etaRatio parameter");
  cgSetParameter1f(myCgVertexParam_etaRatio, etaRatio);

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

  myCgFragmentParam_transmittance =
    cgGetNamedParameter(myCgFragmentProgram, "transmittance");
  checkForCgError("could not get transmittance parameter");
  cgSetParameter1f(myCgFragmentParam_transmittance, transmittance);

  myCgFragmentParam_decalMap =
    cgGetNamedParameter(myCgFragmentProgram, "decalMap");
  checkForCgError("getting decalMap parameter");

  cgGLSetTextureParameter(myCgFragmentParam_decalMap, TO_DECAL);
  checkForCgError("setting decal 2D texture");

  myCgFragmentParam_environmentMap =
    cgGetNamedParameter(myCgFragmentProgram, "environmentMap");
  checkForCgError("getting environmentMap parameter");

  cgGLSetTextureParameter(myCgFragmentParam_environmentMap, TO_ENVIRONMENT);
  checkForCgError("setting environment cube map texture");

  /* Load the decal and environment map the fragment program will sample. */
  glBindTexture(GL_TEXTURE_2D, TO_DECAL);
  loadDecalFromDDS("TilePattern.dds");
  glBindTexture(GL_TEXTURE_CUBE_MAP, TO_ENVIRONMENT);
  loadCubeMapFromDDS("CloudyHillsCubemap.dds");

  /* Create GLUT menu. */
  glutCreateMenu(menu);
  glutAddMenuEntry("[ ] Animate", ' ');
  glutAddMenuEntry("[}] Increase index of refraction", ' ');
  glutAddMenuEntry("[{] Decrease index of refraction", ' ');
  glutAddMenuEntry("[+] Increase transmittance", ' ');
  glutAddMenuEntry("[-] Decrease transmittance", ' ');
  glutAddMenuEntry("[w] Toggle wireframe", 'w');
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

/* This is a "good enough" loader for DDS cube maps compressed in the DXT1 format. */
void loadCubeMapFromDDS(const char *filename)
{
  FILE *file = fopen(filename, "rb");
  long size;
  void *data;
  char *beginning, *image;
  int *words;
  size_t bytes;
  DDS_file_header *header;
  int i, face, level;

  if (!file) {
    fprintf(stderr, "%s: could not open cube map %s\n", myProgramName, filename);
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

/* From the DirectX SDK's ddraw.h */
#define DDSCAPS2_CUBEMAP                        0x00000200
#define DDSCAPS2_CUBEMAP_POSITIVEX              0x00000400
#define DDSCAPS2_CUBEMAP_NEGATIVEX              0x00000800
#define DDSCAPS2_CUBEMAP_POSITIVEY              0x00001000
#define DDSCAPS2_CUBEMAP_NEGATIVEY              0x00002000
#define DDSCAPS2_CUBEMAP_POSITIVEZ              0x00004000
#define DDSCAPS2_CUBEMAP_NEGATIVEZ              0x00008000
#define DDSCAPS2_CUBEMAP_ALLFACES ( DDSCAPS2_CUBEMAP_POSITIVEX |\
                                    DDSCAPS2_CUBEMAP_NEGATIVEX |\
                                    DDSCAPS2_CUBEMAP_POSITIVEY |\
                                    DDSCAPS2_CUBEMAP_NEGATIVEY |\
                                    DDSCAPS2_CUBEMAP_POSITIVEZ |\
                                    DDSCAPS2_CUBEMAP_NEGATIVEZ )
  EXPECT(header->caps.caps2, DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_ALLFACES);

  beginning = data;
  image = (char*) &header[1];
  /* For each face of the cube map (in +X, -X, +Y, -Y, +Z, and -Z order)... */
  for (face=0; face<6; face++) {
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
      /* Careful formula to compute the size of a DXT1 mipmap level.
         This formula accounts for the fact that mipmap levels get
         no smaller than a 4x4 block. */
      GLsizei imageSizeInBytes = ((width+3)>>2)*((height+3)>>2) * bytesPer4x4Block;
      size_t offsetInToRead = image + imageSizeInBytes - beginning;

      if (offsetInToRead > bytes) {
        fprintf(stderr, "%s: DDS images over read the data!\n", myProgramName);
        exit(1);
      }
      glCompressedTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+face, level,
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
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, header->mipMapCount-1);
    /* Use better trilinear mipmap minification filter instead of the default. */
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  } else {
    /* OpenGL's default minification filter (GL_NEAREST_MIPMAP_LINEAR) requires
       mipmaps this DDS file does not have so switch to a linear filter that
       doesn't require mipmaps. */
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  }
  /* To eliminate artifacts at the seems from the default wrap mode (GL_REPEAT),
     switch the wrap modes to clamp to edge. */
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  
  free(data);
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

  /* Examine the header to make sure it is what we expect. */
  header = data;
  EXPECT(header->magic, FOURCC("DDS "));
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

/* Draw the surroundings as a cube with each face of the
   cube map environment map applied.  This routine doesn't
   use Cg. */
static void drawSurroundings(const GLfloat *eyePosition)
{
  static const GLfloat vertex[4*6][3] = {
    /* Positive X face. */
    { 1, -1, -1 },  { 1, 1, -1 },  { 1, 1, 1 },  { 1, -1, 1 },
    /* Negative X face. */
    { -1, -1, -1 },  { -1, 1, -1 },  { -1, 1, 1 },  { -1, -1, 1 },
    /* Positive Y face. */
    { -1, 1, -1 },  { 1, 1, -1 },  { 1, 1, 1 },  { -1, 1, 1 },
    /* Negative Y face. */
    { -1, -1, -1 },  { 1, -1, -1 },  { 1, -1, 1 },  { -1, -1, 1 },
    /* Positive Z face. */
    { -1, -1, 1 },  { 1, -1, 1 },  { 1, 1, 1 },  { -1, 1, 1 },
    /* Negatieve Z face. */
    { -1, -1, -1 },  { 1, -1, -1 },  { 1, 1, -1 },  { -1, 1, -1 },
  };

  const float surroundingsDistance = 8;
  int i;

  glLoadIdentity();
  gluLookAt(eyePosition[0], eyePosition[1], eyePosition[2],
            0, 0, 0,
            0, 1, 0);
  /* Scale the cube to be drawn by the desired surrounding distance. */
  glScalef(surroundingsDistance,
           surroundingsDistance,
           surroundingsDistance);

  glEnable(GL_TEXTURE_CUBE_MAP);
  glBindTexture(GL_TEXTURE_CUBE_MAP, TO_ENVIRONMENT);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glBegin(GL_QUADS);
  /* For each vertex of each face of the cube... */
  for (i=0; i<4*6; i++) {
    glTexCoord3fv(vertex[i]);
    glVertex3fv(vertex[i]);
  }
  glEnd();
}

static void display(void)
{
  /* World-space positions for light and eye. */
  const float eyePosition[4] = { 6*sin(eyeAngle), 
                                 eyeHeight,
                                 6*cos(eyeAngle), 1 };

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
  cgSetParameter3fv(myCgVertexParam_eyePositionW, eyePosition);

  /* modelViewMatrix = viewMatrix * modelMatrix */
  multMatrix(modelViewMatrix, viewMatrix, modelMatrix);

  /* modelViewProj = projectionMatrix * modelViewMatrix */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, modelViewMatrix);

  /* Set matrix parameter with row-major matrix. */
  cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
  cgSetMatrixParameterfr(myCgVertexParam_modelToWorld, modelMatrix);
  cgUpdateProgramParameters(myCgVertexProgram);
  drawMonkeyHead();

  cgGLDisableProfile(myCgVertexProfile);
  checkForCgError("disabling vertex profile");

  cgGLDisableProfile(myCgFragmentProfile);
  checkForCgError("disabling fragment profile");

  drawSurroundings(eyePosition);

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
  case '[':
  case '{':
    etaRatio -= 0.1;
    printf("etaRatio = %f\n", etaRatio);
    cgSetParameter1f(myCgVertexParam_etaRatio, etaRatio);
    glutPostRedisplay();
    break;
  case ']':
  case '}':
    etaRatio += 0.1;
    printf("etaRatio = %f\n", etaRatio);
    cgSetParameter1f(myCgVertexParam_etaRatio, etaRatio);
    glutPostRedisplay();
    break;
  case '+':
    transmittance += 0.1;
    printf("transmittance = %f\n", transmittance);
    cgSetParameter1f(myCgFragmentParam_transmittance, transmittance);
    glutPostRedisplay();
    break;
  case '-':
    transmittance -= 0.1;
    printf("transmittance = %f\n", transmittance);
    cgSetParameter1f(myCgFragmentParam_transmittance, transmittance);
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

/* Use a motion and mouse GLUT callback to allow the viewer to
   rotate around the monkey head and move the viewer up and down. */

int beginx, beginy;
int moving = 0;

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
}

void
mouse(int button, int state, int x, int y)
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
