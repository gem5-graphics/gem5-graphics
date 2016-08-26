
/* cgfx_bumpdemo.c - a Direct3D9-based Cg 1.5 demo */

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <windows.h>
#include <d3d9.h>      /* Direct3D9 API: Can't include this?  Is DirectX SDK installed? */
#include "DXUT.h"      /* DirectX Utility Toolkit (part of the DirectX SDK) */
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "d3d9.lib")
#include <Cg/cg.h>     /* Cg Core API: Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgD3D9.h> /* Cg Direct3D9 API (part of Cg Toolkit) */

static const char *myProgramName = "cgfx_bumpdemo"; /* Program name for messages. */

/* Cg global variables */
CGcontext   myCgContext;
CGeffect    myCgEffect;
CGtechnique myCgTechnique, myCgTechniqueHLSL, myCurrentCgTechninque;
CGparameter myCgEyePositionParam,
            myCgLightPositionParam,
            myCgModelViewProjParam;

int myRenderWithHLSLProfile = 0;

/* Forward declare helper functions and callbacks registered by main. */
static void checkForCgError(const char *situation);
static HRESULT CALLBACK OnResetDevice(IDirect3DDevice9*, const D3DSURFACE_DESC*, void*);
static void CALLBACK OnFrameRender(IDirect3DDevice9*, double, float, void*);
static void CALLBACK OnLostDevice(void*);
static void CALLBACK OnFrameMove(IDirect3DDevice9*, double, float, void*);
static void CALLBACK KeyboardProc(UINT, bool, bool, void*);

int main(int argc, char **argv)
{
  myCgContext = cgCreateContext();
  checkForCgError("creating context");

  /* Parse command line, handle default hotkeys, and show messages. */
  DXUTInit();

  DXUTSetCallbackDeviceReset(OnResetDevice);
  DXUTSetCallbackDeviceLost(OnLostDevice);
  DXUTSetCallbackFrameRender(OnFrameRender);
  DXUTSetCallbackFrameMove(OnFrameMove);
  DXUTSetCallbackKeyboard(KeyboardProc);
  DXUTCreateWindow(L"cgfx_bumpdemo (Direct3D9)");

  bool windowed = true;
  DXUTCreateDevice(D3DADAPTER_DEFAULT, windowed, 640, 480);
  DXUTMainLoop();

  /* Demonstrate proper deallocation of Cg runtime data structures.
     Not strictly necessary if we are simply going to exit. */
  cgDestroyEffect(myCgEffect);
  checkForCgError("destroying effect");
  cgDestroyContext(myCgContext);
  cgD3D9SetDevice(NULL);

  return DXUTGetExitCode();
}

static void checkForCgError(const char *situation)
{
  CGerror error;
  const char *string = cgGetLastErrorString(&error);
  
  if (error != CG_NO_ERROR) {
    if (error == CG_COMPILER_ERROR) {
      fprintf(stderr,
             "Program: %s\n"
             "Situation: %s\n"
             "Error: %s\n\n"
             "Cg compiler output...\n%s",
             myProgramName, situation, string,
             cgGetLastListing(myCgContext));
    } else {
      fprintf(stderr,
              "Program: %s\n"
              "Situation: %s\n"
              "Error: %s",
              myProgramName, situation, string);
    }
    exit(1);
  }
}

static void initCg(void)
{
  cgD3D9RegisterStates(myCgContext);
  checkForCgError("registering standard CgFX states");
  cgD3D9SetManageTextureParameters(myCgContext, CG_TRUE);
  checkForCgError("manage texture parameters");

  myCgEffect = cgCreateEffectFromFile(myCgContext, "bumpdemo.cgfx", NULL);
  checkForCgError("creating bumpdemo.cgfx effect");
  assert(myCgEffect);

  myCgTechnique = cgGetFirstTechnique(myCgEffect);
  while (myCgTechnique && cgValidateTechnique(myCgTechnique) == CG_FALSE) {
    fprintf(stderr, "%s: Technique %s did not validate.  Skipping.\n",
      myProgramName, cgGetTechniqueName(myCgTechnique));
    myCgTechnique = cgGetNextTechnique(myCgTechnique);
  }
  if (myCgTechnique) {
    fprintf(stderr, "%s: Use technique %s.\n",
      myProgramName, cgGetTechniqueName(myCgTechnique));
  } else {
    fprintf(stderr, "%s: No valid technique\n",
      myProgramName);
    exit(1);
  }
  myCurrentCgTechninque = myCgTechnique;

  myCgTechniqueHLSL = cgGetNamedTechnique(myCgEffect, "bumpdemo_hlsl");
  if (!cgValidateTechnique(myCgTechniqueHLSL)) {
    fprintf(stderr, "%s: HLSL technique %s failed to validate.\n",
      myProgramName, cgGetTechniqueName(myCgTechniqueHLSL));
    myCgTechniqueHLSL = 0;
    myRenderWithHLSLProfile = 0;
  }

  myCgModelViewProjParam =
    cgGetEffectParameterBySemantic(myCgEffect, "ModelViewProjection");
  if (!myCgModelViewProjParam) {
    fprintf(stderr,
      "%s: must find parameter with ModelViewProjection semantic\n",
      myProgramName);
    exit(1);
  }
  myCgEyePositionParam =
    cgGetNamedEffectParameter(myCgEffect, "EyePosition");
  if (!myCgEyePositionParam) {
    fprintf(stderr, "%s: must find parameter named EyePosition\n",
      myProgramName);
    exit(1);
  }
  myCgLightPositionParam =
    cgGetNamedEffectParameter(myCgEffect, "LightPosition");
  if (!myCgLightPositionParam) {
    fprintf(stderr, "%s: must find parameter named LightPosition\n",
      myProgramName);
    exit(1);
  }
}

static const unsigned char
myBrickNormalMapImage[3*(128*128+64*64+32*32+16*16+8*8+4*4+2*2+1*1)] = {
/* RGB8 image data for a mipmapped 128x128 normal map for a brick pattern */
#include "brick_image.h"
};

static const unsigned char
myNormalizeVectorCubeMapImage[6*3*32*32] = {
/* RGB8 image data for a normalization vector cube map with 32x32 faces */
#include "normcm_image.h"
};

static void useSamplerParameter(CGeffect effect,
                                const char *paramName, IDirect3DBaseTexture9 *tex)
{
  CGparameter param = cgGetNamedEffectParameter(effect, paramName);

  if (!param) {
    fprintf(stderr, "%s: expected effect parameter named %s\n",
      myProgramName, paramName);
    exit(1);
  }

  cgD3D9SetTextureParameter(param, tex);
  cgSetSamplerState(param);
}

static PDIRECT3DTEXTURE9 myBrickNormalMap = NULL;
static PDIRECT3DCUBETEXTURE9 myNormalizeVectorCubeMap = NULL;

static HRESULT initTextures(IDirect3DDevice9* pDev)
{
  unsigned int size, level;
  int face;
  const unsigned char *image;
  D3DLOCKED_RECT lockedRect;

  if (FAILED(pDev->CreateTexture(128, 128, 0,
                                 0, D3DFMT_X8R8G8B8,
                                 D3DPOOL_MANAGED,
                                 &myBrickNormalMap, NULL))) {
    return E_FAIL;
  }

  for (size = 128, level = 0, image = myBrickNormalMapImage;
       size > 0;
       image += 3*size*size, size /= 2, level++) {

    if (FAILED(myBrickNormalMap->LockRect(level, &lockedRect, 0, 0)))
      return E_FAIL;

    DWORD *texel = (DWORD*) lockedRect.pBits;

    const int bytes = size*size*3;

    for (int i=0; i<bytes; i+=3) {
      *texel++ = image[i+0] << 16 |
                 image[i+1] << 8  |
                 image[i+2];
    }

    myBrickNormalMap->UnlockRect(level);
  }

  if (FAILED(pDev->CreateCubeTexture(32, 1,
                                     0, D3DFMT_X8R8G8B8,
                                     D3DPOOL_MANAGED,
                                     &myNormalizeVectorCubeMap, NULL)))
    return E_FAIL;

  const int bytesPerFace = 32*32*3;
  for (face = D3DCUBEMAP_FACE_POSITIVE_X, image = myNormalizeVectorCubeMapImage;
       face <= D3DCUBEMAP_FACE_NEGATIVE_Z;
       face += 1, image += bytesPerFace) {
    if (FAILED(myNormalizeVectorCubeMap->LockRect((D3DCUBEMAP_FACES)face, 0, &lockedRect, 0, 0)))
      return E_FAIL;

    DWORD *texel = (DWORD*) lockedRect.pBits;

    for (int i=0; i<bytesPerFace; i+=3) {
      *texel++ = image[i+0] << 16 |
                 image[i+1] << 8  |
                 image[i+2];
    }

    myNormalizeVectorCubeMap->UnlockRect((D3DCUBEMAP_FACES)face, 0);
  }

  useSamplerParameter(myCgEffect, "normalMap",
                      myBrickNormalMap);

  useSamplerParameter(myCgEffect, "normalizeCube",
                      myNormalizeVectorCubeMap);

  return S_OK;
}

struct MY_V3F {
  FLOAT x, y, z;  // Really just need (x,y) so z is always zero.
};

static PDIRECT3DVERTEXBUFFER9 myVertexBuffer = NULL;

static HRESULT initTorusVertexBuffer(IDirect3DDevice9* pDev, int sides, int rings)
{
  const float m = 1.0f / float(rings);
  const float n = 1.0f / float(sides);

  const int numVertsPerStrip = 2 * sides + 2;
  const int numVertsPerPatch = numVertsPerStrip * rings;

  if (FAILED(pDev->CreateVertexBuffer(numVertsPerPatch * sizeof(MY_V3F),
                                      0, D3DFVF_XYZ,
                                      D3DPOOL_DEFAULT,
                                      &myVertexBuffer, NULL)))
    return E_FAIL;

  MY_V3F* pVertices;
  if (FAILED(myVertexBuffer->Lock(0, 0, /* map entire buffer */
                                  (VOID**)&pVertices, 0)))
    return E_FAIL;

  int index = 0;
  for( int i = 0; i < rings; ++i ) {
    for( int j = 0; j <= sides; ++j ) {            
      pVertices[index].x = i*m;
      pVertices[index].y = j*n;
      pVertices[index].z = 0;
      index++;
      pVertices[index].x = (i+1)*m;
      pVertices[index].y = j*n;
      pVertices[index].z = 0;
      index++;
    }        
  }

  myVertexBuffer->Unlock();
  return S_OK;
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
  
  m[0*4+0] = -float(cotangent / aspectRatio);
  m[1*4+0] = 0.0;
  m[2*4+0] = 0.0;
  m[3*4+0] = 0.0;
  
  m[0*4+1] = 0.0;
  m[1*4+1] = float(cotangent);
  m[2*4+1] = 0.0;
  m[3*4+1] = 0.0;
  
  m[0*4+2] = 0.0;
  m[1*4+2] = 0.0;
  m[2*4+2] = float(zFar / deltaZ);
  m[3*4+2] = 1;
  
  m[0*4+3] = 0.0;
  m[1*4+3] = 0.0;
  m[2*4+3] = float(-(zFar / deltaZ)*zNear);
  m[3*4+3] = 0;
}

static float myProjectionMatrix[16];
const int myTorusSides = 20,
          myTorusRings = 40;

static HRESULT CALLBACK OnResetDevice(IDirect3DDevice9* pDev, 
                                      const D3DSURFACE_DESC* backBuf,
                                      void* userContext)
{
  cgD3D9SetDevice(pDev);
  checkForCgError("setting Direct3D device");

  static int firstTime = 1;
  if (firstTime) {
    /* Cg runtime resources such as CGprogram and CGparameter handles
       survive a device reset so we just need to compile a Cg program
       just once.  We do however need to unload Cg programs with
       cgD3DUnloadProgram upon when a Direct3D device is lost and load
       Cg programs every Direct3D device reset with cgD3D9UnloadProgram. */
    initCg();
    firstTime = 0;
  }

  if (FAILED(initTextures(pDev)))
    return E_FAIL;
  if (FAILED(initTorusVertexBuffer(pDev, myTorusSides, myTorusRings)))
    return E_FAIL;

  double fieldOfView = 60.0;  // In degrees
  double width = backBuf->Width;
  double height = backBuf->Height;
  double aspectRatio = width / height;
  double zNear = 0.1;
  double zFar = 100.0;
  buildPerspectiveMatrix(fieldOfView, aspectRatio,
                         zNear, zFar,
                         myProjectionMatrix);

  return S_OK;
}

static void CALLBACK OnLostDevice(void* userContext)
{
  myVertexBuffer->Release();
  myBrickNormalMap->Release();
  myNormalizeVectorCubeMap->Release();
  cgD3D9SetDevice(NULL);
}

/* Initial scene state */
static int myAnimating = 0;
static float myEyeAngle = 0;
static const float myLightPosition[3] = { -8, 0, 15 };

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for gluLookAt. */
static void buildLookAtMatrix(double eyex, double eyey, double eyez,
                              double centerx, double centery, double centerz,
                              double upx, double upy, double upz,
                              float m[16])
{
  double x[3], y[3], z[3], mag;

  /* Difference center and eye vectors to make Z vector. */
  z[0] = centerx - eyex;
  z[1] = centery - eyey;
  z[2] = centerz - eyez;
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
  m[0*4+0] = float(x[0]);
  m[0*4+1] = float(x[1]);
  m[0*4+2] = float(x[2]);
  m[0*4+3] = -float(x[0]*eyex + x[1]*eyey + x[2]*eyez);

  m[1*4+0] = float(y[0]);
  m[1*4+1] = float(y[1]);
  m[1*4+2] = float(y[2]);
  m[1*4+3] = -float(y[0]*eyex + y[1]*eyey + y[2]*eyez);

  m[2*4+0] = float(z[0]);
  m[2*4+1] = float(z[1]);
  m[2*4+2] = float(z[2]);
  m[2*4+3] = -float(z[0]*eyex + z[1]*eyey + z[2]*eyez);

  m[3*4+0] = 0.0;
  m[3*4+1] = 0.0;
  m[3*4+2] = 0.0;
  m[3*4+3] = 1.0;
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

HRESULT drawFlatPatch(IDirect3DDevice9* pDev, IDirect3DVertexBuffer9 *vb, int sides, int rings)
{
  HRESULT hr = S_OK;

  hr = pDev->SetStreamSource(0, vb, 0, sizeof(MY_V3F));
  if (FAILED(hr))
    return hr;

  hr = pDev->SetFVF(D3DFVF_XYZ);
  if (FAILED(hr))
    return hr;

  for (int i = 0, vertStart = 0;
      i < rings;
      i++, vertStart += (2 * sides + 2)) {
    hr = pDev->DrawPrimitive(D3DPT_TRIANGLESTRIP, vertStart, sides * 2);
    if (FAILED(hr))
      return hr;
  }
  return hr;
}

static void CALLBACK OnFrameRender(IDirect3DDevice9* pDev,
                                   double time,
                                   float elapsedTime,
                                   void* userContext)
{
  const float eyeRadius = 18.0,
              eyeElevationRange = 8.0;
  float eyePosition[3];
  CGpass pass;
  float modelViewMatrix[16],
        modelViewProjMatrix[16];

  pDev->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DXCOLOR( 0.1f, 0.3f, 0.6f, 1.0f ), 1.0f, 0);
  pDev->SetRenderState(D3DRS_ZENABLE, D3DZB_TRUE);
  pDev->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);

  if (FAILED(pDev->BeginScene())) 
    return;

  // Get the projection & view matrix from the camera class

  eyePosition[0] = eyeRadius * sin(myEyeAngle);
  eyePosition[1] = eyeElevationRange * sin(myEyeAngle);
  eyePosition[2] = eyeRadius * cos(myEyeAngle);

  buildLookAtMatrix(
  eyePosition[0], eyePosition[1], eyePosition[2], 
    0.0 ,0.0,  0.0,   /* XYZ view center */
    0.0, 1.0,  0.0,   /* Up is in positive Y direction */
    modelViewMatrix);

  /* modelViewProj = projectionMatrix * modelViewMatrix */
  multMatrix(modelViewProjMatrix, myProjectionMatrix, modelViewMatrix);

  // Row major version 
  cgSetMatrixParameterfr(myCgModelViewProjParam, modelViewProjMatrix );

  cgSetParameter3fv(myCgEyePositionParam, eyePosition);
  cgSetParameter3fv(myCgLightPositionParam, myLightPosition);

  /* Iterate through rendering passes for technique (even
     though bumpdemo.cgfx has just one pass). */
  pass = cgGetFirstPass(myCurrentCgTechninque);
  while (pass) {
    cgSetPassState(pass);
    drawFlatPatch(pDev, myVertexBuffer, myTorusSides, myTorusRings );
    cgResetPassState(pass);
    pass = cgGetNextPass(pass);
  }

  pDev->EndScene();
}

static void advanceAnimation(void)
{
  myEyeAngle += 0.05f;
  if (myEyeAngle > 2*3.14159)
    myEyeAngle -= 2*3.14159f;
}

static void CALLBACK OnFrameMove(IDirect3DDevice9* pDev,
                                 double time,
                                 float elapsedTime, 
                                 void* userContext)
{
  if (myAnimating)
    advanceAnimation();
}

static const char *getPassStateAssignmentProgram(CGpass pass, const char *stateName)
{
  CGstateassignment sa = cgGetNamedStateAssignment(pass, stateName);
  CGprogram program = cgGetProgramStateAssignmentValue(sa);
  const char *assembly = cgGetProgramString(program, CG_COMPILED_PROGRAM);

  return assembly;
}

static void dumpTechniqueCompiledPrograms(CGtechnique technique)
{
  const char *techniqueName = cgGetTechniqueName(technique);
  CGbool isValid = cgValidateTechnique(technique);

  if (isValid) {
    CGpass pass = cgGetFirstPass(technique);

    /* Iterate over all passes... */
    while (pass) {
      printf("compiled vertex assembly for %s:\n%s\n",
        techniqueName, getPassStateAssignmentProgram(pass, "VertexProgram"));
      printf("compiled fragment assembly for %s:\n%s\n",
        techniqueName, getPassStateAssignmentProgram(pass, "FragmentProgram"));
      pass = cgGetNextPass(pass);
    }
  } else {
    printf("technique %s not valid\n", techniqueName);
  }
}

static void CALLBACK KeyboardProc(UINT nChar,
                                  bool keyDown,
                                  bool altDown,
                                  void* userContext)
{
  static int wireframe = 0;

  if (!keyDown)
    return;

  switch (nChar) {
  case ' ':
    myAnimating = !myAnimating;
    break;
  case 'H':
    if (myCgTechniqueHLSL) {
      myRenderWithHLSLProfile = !myRenderWithHLSLProfile; // toggle
      if (myRenderWithHLSLProfile) {
        myCurrentCgTechninque = myCgTechniqueHLSL;
      } else {
        myCurrentCgTechninque = myCgTechnique;
      }
      const char *techniqueName = cgGetTechniqueName(myCurrentCgTechninque);
      fprintf(stderr, "%s: Using technique %s.\n",
        myProgramName, techniqueName);
    }
    break;
  case 'L':
    dumpTechniqueCompiledPrograms(myCurrentCgTechninque);
    break;
  case 'W':
    wireframe = !wireframe;
    if (wireframe) {
      cgD3D9GetDevice()->SetRenderState(D3DRS_FILLMODE, D3DFILL_WIREFRAME);
    } else {
      cgD3D9GetDevice()->SetRenderState(D3DRS_FILLMODE, D3DFILL_SOLID);
    }
    break;
  case 27:  /* Esc key */
    // DXUT handles Escape to Exit
    break;
  }
}
