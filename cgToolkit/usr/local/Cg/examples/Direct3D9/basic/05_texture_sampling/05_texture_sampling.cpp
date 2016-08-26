
/* 05_texture_sampling.c - OpenGL-based very simple vertex program example
   using Cg program from Chapter 2 of "The Cg Tutorial" (Addison-Wesley,
   ISBN 0321194969). */

#include <windows.h>
#include <stdio.h>
#include <d3d9.h>     /* Can't include this?  Is DirectX SDK installed? */
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgD3D9.h>

#include "DXUT.h"  /* DirectX Utility Toolkit (part of the DirectX SDK) */

#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "d3d9.lib")

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgFragmentProfile;
static CGprogram   myCgVertexProgram,
                   myCgFragmentProgram;
static CGparameter myCgFragmentParam_decal;      

static PDIRECT3DVERTEXBUFFER9 myVertexBuffer = NULL;
static PDIRECT3DTEXTURE9 myTexture = NULL;

static const WCHAR *myProgramNameW = L"05_texture_sampling";
static const char *myProgramName = "05_texture_sampling",
                  *myVertexProgramFileName = "C3E2v_varying.cg",
/* Page 65 */     *myVertexProgramName = "C3E2v_varying",
                  *myFragmentProgramFileName = "C3E3f_texture.cg",
/* Page 67 */     *myFragmentProgramName = "C3E3f_texture";

static const unsigned char
myDemonTextureImage[3*(128*128+64*64+32*32+16*16+8*8+4*4+2*2+1*1)] = {
/* RGB8 image data for a mipmapped 128x128 demon texture */
#include "demon_image.h"
};

static void checkForCgError(const char *situation)
{
  char buffer[4096];
  CGerror error;
  const char *string = cgGetLastErrorString(&error);
  
  if (error != CG_NO_ERROR) {
    if (error == CG_COMPILER_ERROR) {
      sprintf(buffer,
              "Program: %s\n"
              "Situation: %s\n"
              "Error: %s\n\n"
              "Cg compiler output...\n",
              myProgramName, situation, string);
      OutputDebugStringA(buffer);
      OutputDebugStringA(cgGetLastListing(myCgContext));
      sprintf(buffer,
              "Program: %s\n"
              "Situation: %s\n"
              "Error: %s\n\n"
              "Check debug output for Cg compiler output...",
              myProgramName, situation, string);
      MessageBoxA(0, buffer,
                  "Cg compilation error", MB_OK | MB_ICONSTOP | MB_TASKMODAL);
    } else {
      sprintf(buffer,
              "Program: %s\n"
              "Situation: %s\n"
              "Error: %s",
              myProgramName, situation, string);
      MessageBoxA(0, buffer,
                  "Cg runtime error", MB_OK | MB_ICONSTOP | MB_TASKMODAL);
    }
    exit(1);
  }
}

/* Forward declared DXUT callbacks registered by WinMain. */
static HRESULT CALLBACK OnResetDevice(IDirect3DDevice9*, const D3DSURFACE_DESC*, void*);
static void CALLBACK OnFrameRender(IDirect3DDevice9*, double, float, void*);
static void CALLBACK OnLostDevice(void*);

INT WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);

  DXUTSetCallbackDeviceReset(OnResetDevice);
  DXUTSetCallbackDeviceLost(OnLostDevice);
  DXUTSetCallbackFrameRender(OnFrameRender);

  /* Parse  command line, handle  default hotkeys, and show messages. */
  DXUTInit();

  DXUTCreateWindow(myProgramNameW);

  /* Display 400x400 window. */
  DXUTCreateDevice(D3DADAPTER_DEFAULT, true, 400, 400);

  DXUTMainLoop();

  cgDestroyProgram(myCgVertexProgram);
  checkForCgError("destroying vertex program");
  cgDestroyProgram(myCgFragmentProgram);
  checkForCgError("destroying fragment program");
  cgDestroyContext(myCgContext);

  return DXUTGetExitCode();
}

static void createCgPrograms()
{
  const char **profileOpts;

  /* Determine the best profiles once a device to be set. */
  myCgVertexProfile = cgD3D9GetLatestVertexProfile();
  checkForCgError("getting latest vertex profile");

  profileOpts = cgD3D9GetOptimalOptions(myCgVertexProfile);
  checkForCgError("getting latest vertex profile options");

  myCgVertexProgram =
    cgCreateProgramFromFile(
      myCgContext,              /* Cg runtime context */
      CG_SOURCE,                /* Program in human-readable form */
      myVertexProgramFileName,  /* Name of file containing program */
      myCgVertexProfile,        /* Profile: OpenGL ARB vertex program */
      myVertexProgramName,      /* Entry function name */
      profileOpts);             /* Pass optimal compiler options */
  checkForCgError("creating vertex program from file");

  /* Determine the best profile once a device to be set. */
  myCgFragmentProfile = cgD3D9GetLatestPixelProfile();
  checkForCgError("getting latest fragment profile");

  profileOpts = cgD3D9GetOptimalOptions(myCgFragmentProfile);
  checkForCgError("getting latest fragment profile options");

  myCgFragmentProgram =
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myFragmentProgramFileName,  /* Name of file containing program */
      myCgFragmentProfile,        /* Profile: OpenGL ARB vertex program */
      myFragmentProgramName,      /* Entry function name */
      profileOpts);               /* No extra compiler options */
  checkForCgError("creating fragment program from file");

  myCgFragmentParam_decal =
    cgGetNamedParameter(myCgFragmentProgram, "decal");
  checkForCgError("getting decal parameter");
}

struct MY_V3F_T2F {
  FLOAT x, y, z;
  FLOAT s, t;
};

static HRESULT initVertexBuffer(IDirect3DDevice9* pDev)
{
  /* Initialize three vertices for rendering a triangle. */
  static const MY_V3F_T2F triangleVertices[] = {
    { -0.8f,  0.8f, 0.0f, 0, 0 },   /* st=(0,0) */
    {  0.8f,  0.8f, 0.0f, 1, 0 },   /* st=(1,0) */
    {  0.0f, -0.8f, 0.0f, 0.5f, 1 } /* st=(0.5,1)*/
  };

  if (FAILED(pDev->CreateVertexBuffer(sizeof(triangleVertices),
                                      0, D3DFVF_XYZ|D3DFVF_TEX0,
                                      D3DPOOL_DEFAULT,
                                      &myVertexBuffer, NULL))) {
    return E_FAIL;
  }

  void* pVertices;
  if (FAILED(myVertexBuffer->Lock(0, 0, /* map entire buffer */
                                  &pVertices, 0))) {
    return E_FAIL;
  }
  memcpy(pVertices, triangleVertices, sizeof(triangleVertices));
  myVertexBuffer->Unlock();

  return S_OK;
}


static HRESULT initTexture(IDirect3DDevice9* pDev)
{
  D3DLOCKED_RECT lockedRect;

  if (FAILED(pDev->CreateTexture(128, 128, 0,
                                 D3DUSAGE_AUTOGENMIPMAP, D3DFMT_X8R8G8B8,
                                 D3DPOOL_MANAGED,
                                 &myTexture, NULL))) {
    return E_FAIL;
  }

  if (FAILED(myTexture->LockRect(0, &lockedRect, 0, 0))) {
    return E_FAIL;
  }

  DWORD *texel = (DWORD*) lockedRect.pBits;

  for (int i=0; i<128*128*3; i+=3) {
    *texel++ = myDemonTextureImage[i+0] << 16 |
               myDemonTextureImage[i+1] << 8  |
               myDemonTextureImage[i+2];
  }

  myTexture->UnlockRect(0);

  myTexture->GenerateMipSubLevels();

  return S_OK;
}

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
    createCgPrograms();
    firstTime = 0;
  }

  /* false below means "with parameter shadowing" */
  cgD3D9LoadProgram(myCgVertexProgram, false, 0);
  checkForCgError("loading vertex program");

  /* false below means "with parameter shadowing" */
  cgD3D9LoadProgram(myCgFragmentProgram, false, 0);
  checkForCgError("loading fragment program");

  if (FAILED(initVertexBuffer(pDev))) {
    return E_FAIL;
  }

  if (FAILED(initTexture(pDev))) {
    return E_FAIL;
  }

  return S_OK;
}

static void CALLBACK OnFrameRender(IDirect3DDevice9* pDev,
                                   double time,
                                   float elapsedTime,
                                   void* userContext)
{
  pDev->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
              D3DXCOLOR(0.1f, 0.3f, 0.6f, 0.0f), 1.0f, 0);

  if (SUCCEEDED(pDev->BeginScene())) {
    cgD3D9BindProgram(myCgVertexProgram);
    checkForCgError("binding vertex program");

    cgD3D9SetTexture(myCgFragmentParam_decal, myTexture);  

    cgD3D9BindProgram(myCgFragmentProgram);
    checkForCgError("binding fragment program");

    /* Render the stars. */
    pDev->SetStreamSource(0, myVertexBuffer, 0, sizeof(MY_V3F_T2F));
    pDev->SetFVF(D3DFVF_XYZ|D3DFVF_TEX1);

    pDev->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
    pDev->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
    pDev->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);

    pDev->DrawPrimitive(D3DPT_TRIANGLELIST, 0, 1);

    pDev->EndScene();
  }
}

static void CALLBACK OnLostDevice(void* userContext)
{
  myVertexBuffer->Release();
  myTexture->Release();
  cgD3D9SetDevice(NULL);
}
