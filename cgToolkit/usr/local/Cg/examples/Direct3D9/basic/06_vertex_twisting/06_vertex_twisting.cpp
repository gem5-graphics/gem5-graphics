
/* 06_vertex_twisting.c - OpenGL-based very simple vertex program example
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
static CGparameter myCgVertexParam_twisting;

static LPDIRECT3DVERTEXBUFFER9 myVertexBuffer = NULL;

static const WCHAR *myProgramNameW = L"06_vertex_twisting";
static const char *myProgramName = "06_vertex_twisting",
                  *myVertexProgramFileName = "C3E4v_twist.cg",
/* Page 79 */     *myVertexProgramName = "C3E4v_twist",
                  *myFragmentProgramFileName = "C2E2f_passthru.cg",
/* Page 53 */     *myFragmentProgramName = "C2E2f_passthru";

static bool myAnimating = false;
static float myTwisting = 2.9f, /* Twisting angle in radians. */
             myTwistDirection = 0.1f; /* Animation delta for twist. */
static int myNumTriangles;

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
static void CALLBACK OnFrameMove(IDirect3DDevice9*, double, float, void*);
static void CALLBACK KeyboardProc(UINT, bool, bool, void*);

INT WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);

  DXUTSetCallbackDeviceReset(OnResetDevice);
  DXUTSetCallbackDeviceLost(OnLostDevice);
  DXUTSetCallbackFrameRender(OnFrameRender);
  DXUTSetCallbackFrameMove(OnFrameMove);
  DXUTSetCallbackKeyboard(KeyboardProc);

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

  myCgVertexParam_twisting =
    cgGetNamedParameter(myCgVertexProgram, "twisting");
  checkForCgError("could not get twisting parameter");

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
}

struct MY_V3F_C4UB {
  FLOAT x, y, z;
  DWORD color;  // RGBA
};

/* Helper function to write vertex positions into vertex buffer. */
static inline void writeVertex(MY_V3F_C4UB &v,
                               const float p[2], const float c[3])
{
  v.x = FLOAT(p[0]);
  v.y = FLOAT(p[1]);
  v.z = 0;
  v.color = D3DXCOLOR(c[0], c[1], c[2], 1);
}

static void triangleDivide(MY_V3F_C4UB v[],
                           int &n,
                           int depth,
                           const float a[2], const float b[2], const float c[2],
                           const float ca[3], const float cb[3], const float cc[3])
{
  if (depth == 0) {
    writeVertex(v[n++], a, ca);
    writeVertex(v[n++], b, cb);
    writeVertex(v[n++], c, cc);
  } else {
    const float d[2] = { (a[0]+b[0])/2, (a[1]+b[1])/2 },
                e[2] = { (b[0]+c[0])/2, (b[1]+c[1])/2 },
                f[2] = { (c[0]+a[0])/2, (c[1]+a[1])/2 };
    const float cd[3] = { (ca[0]+cb[0])/2, (ca[1]+cb[1])/2, (ca[2]+cb[2])/2 },
                ce[3] = { (cb[0]+cc[0])/2, (cb[1]+cc[1])/2, (cb[2]+cc[2])/2 },
                cf[3] = { (cc[0]+ca[0])/2, (cc[1]+ca[1])/2, (cc[2]+ca[2])/2 };

    depth -= 1;
    triangleDivide(v, n, depth, a, d, f, ca, cd, cf);
    triangleDivide(v, n, depth, d, b, e, cd, cb, ce);
    triangleDivide(v, n, depth, f, e, c, cf, ce, cc);
    triangleDivide(v, n, depth, d, e, f, cd, ce, cf);
  }
}

static inline int numDividedTriangles(int subdivisions)
{
  int tris = 1;

  for (int i=0; i<subdivisions; i++) {
    tris *= 4;
  }
  return tris;
}

static HRESULT initVertexBuffer(IDirect3DDevice9* pDev)
{
  const int subdivisions = 5;

  myNumTriangles = numDividedTriangles(subdivisions);

  if (FAILED(pDev->CreateVertexBuffer(myNumTriangles*3*sizeof(MY_V3F_C4UB),
                                      0, D3DFVF_XYZ|D3DFVF_DIFFUSE,
                                      D3DPOOL_DEFAULT,
                                      &myVertexBuffer, NULL))) {
    return E_FAIL;
  }

  void *pVertices;
  if (FAILED(myVertexBuffer->Lock(0, 0, &pVertices, 0))) {
    return E_FAIL;
  }

  const float a[2] = { -0.8f, 0.8f },
              b[2] = {  0.8f, 0.8f },
              c[2] = {  0.0f, -0.8f },
              ca[3] = { 0, 0, 1 },
              cb[3] = { 0, 0, 1 },
              cc[3] = { 0.7f, 0.7f, 1 };

  int n = 0;
  MY_V3F_C4UB *v = (MY_V3F_C4UB*) pVertices;
  triangleDivide(v, n, subdivisions, a, b, c, ca, cb, cc);

  myVertexBuffer->Unlock();

  return S_OK;
}

HRESULT CALLBACK OnResetDevice(IDirect3DDevice9* pDev, 
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

  return initVertexBuffer(pDev);
}

void CALLBACK OnFrameRender(IDirect3DDevice9* pDev,
                            double fTime, float fElapsedTime,
                            void* userContext)
{
  pDev->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
              D3DXCOLOR(1, 1, 1, 1), 1.0f, 0);

  if (SUCCEEDED(pDev->BeginScene())) {
    cgSetParameter1f(myCgVertexParam_twisting, myTwisting);

    cgD3D9BindProgram(myCgVertexProgram);
    checkForCgError("binding vertex program");

    cgD3D9BindProgram(myCgFragmentProgram);
    checkForCgError("binding fragment program");

    /* Render the stars. */
    pDev->SetStreamSource(0, myVertexBuffer, 0, sizeof(MY_V3F_C4UB));
    pDev->SetFVF(D3DFVF_XYZ|D3DFVF_DIFFUSE);

    pDev->DrawPrimitive(D3DPT_TRIANGLELIST, 0, myNumTriangles);

    pDev->EndScene();
  }
}

static void CALLBACK OnFrameMove(IDirect3DDevice9* pDev,
                                 double time,
                                 float elapsedTime, 
                                 void* userContext)
{
  if (myAnimating) {
    if (myTwisting > 3) {
      myTwistDirection = -0.05f;
    } else if (myTwisting < -3) {
      myTwistDirection = 0.05f;
    }
    myTwisting += myTwistDirection;
  }
}

static void CALLBACK KeyboardProc(UINT nChar,
                                  bool keyDown,
                                  bool altDown,
                                  void* userContext)
{
  static bool wireframe = false;

  if (keyDown) {
    switch (nChar) {
    case ' ':
      myAnimating = !myAnimating;
      break;
    case 'W':
      wireframe = !wireframe;
      if (wireframe) {
        cgD3D9GetDevice()->SetRenderState(D3DRS_FILLMODE, D3DFILL_WIREFRAME);
      } else {
        cgD3D9GetDevice()->SetRenderState(D3DRS_FILLMODE, D3DFILL_SOLID);
      }
      break;
    }
  }
}

static void CALLBACK OnLostDevice(void* userContext)
{
  myVertexBuffer->Release();
  cgD3D9SetDevice(NULL);
}
