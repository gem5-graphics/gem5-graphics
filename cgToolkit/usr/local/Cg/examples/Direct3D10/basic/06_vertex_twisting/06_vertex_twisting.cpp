
/* 06_vertex_twisting.c - Direct3D10-based vertex twisting example
   using Cg program from Chapter 2 of "The Cg Tutorial" (Addison-Wesley,
   ISBN 0321194969). */

#include <windows.h>
#include <stdio.h>
#include <d3d10_1.h>
#include <d3d10.h>     /* Can't include this?  Is DirectX SDK installed? */
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgD3D10.h>

#include "DXUT.h"  /* DirectX Utility Toolkit (part of the DirectX SDK) */

#pragma comment(lib, "d3d10.lib")

static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,
                   myCgFragmentProfile;
static CGprogram   myCgVertexProgram,
                   myCgFragmentProgram;
static CGparameter myCgVertexParam_twisting;

static const WCHAR *myProgramNameW = L"06_vertex_twisting";
static const char *myProgramName = "06_vertex_twisting",
                  *myVertexProgramFileName = "C3E4v_twist.cg",
/* Page 79 */     *myVertexProgramName = "C3E4v_twist",
                  *myFragmentProgramFileName = "C2E2f_passthru.cg",
/* Page 53 */     *myFragmentProgramName = "C2E2f_passthru";

static bool myAnimating = true;
static float myTwisting = 2.9f, /* Twisting angle in radians. */
             myTwistDirection = 0.1f; /* Animation delta for twist. */
static int myNumTriangles;

ID3D10InputLayout *      myVertexLayout = NULL;
ID3D10Buffer *           myVB = NULL;
ID3D10BlendState *       myBlendState_NoBlend     = NULL;
ID3D10RasterizerState *	 myRasterizerState_NoCull = NULL;

struct Vertex3
{
    float x;
    float y;
    float z;
    float r;
    float g;
    float b;
};

static void checkForCgError(const char *situation)
{
  char buffer[4096];
  CGerror error;
  const char *string = cgGetLastErrorString(&error);
  
  if (error != CG_NO_ERROR) {
    if (error == CG_COMPILER_ERROR) {
      sprintf_s(buffer,
              "Program: %s\n"
              "Situation: %s\n"
              "Error: %s\n\n"
              "Cg compiler output...\n",
              myProgramName, situation, string);
      OutputDebugStringA(buffer);
      OutputDebugStringA(cgGetLastListing(myCgContext));
      sprintf_s(buffer,
              "Program: %s\n"
              "Situation: %s\n"
              "Error: %s\n\n"
              "Check debug output for Cg compiler output...",
              myProgramName, situation, string);
      MessageBoxA(0, buffer,
                  "Cg compilation error", MB_OK | MB_ICONSTOP | MB_TASKMODAL);
    } else {
      sprintf_s(buffer,
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

bool    CALLBACK IsD3D10DeviceAcceptable( UINT Adapter, UINT Output, D3D10_DRIVER_TYPE DeviceType, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK OnD3D10CreateDevice( ID3D10Device * pDev, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void    CALLBACK OnD3D10FrameRender( ID3D10Device * pDev, double fTime, float fElapsedTime, void* pUserContext );
void    CALLBACK OnD3D10DestroyDevice( void* pUserContext );
void    CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext );
void    CALLBACK KeyboardProc(UINT, bool, bool, void*);

INT WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);

  DXUTSetCallbackD3D10DeviceAcceptable( IsD3D10DeviceAcceptable );
  DXUTSetCallbackD3D10DeviceCreated( OnD3D10CreateDevice );
  DXUTSetCallbackD3D10DeviceDestroyed( OnD3D10DestroyDevice );
  DXUTSetCallbackD3D10FrameRender( OnD3D10FrameRender );
  DXUTSetCallbackFrameMove( OnFrameMove );
  DXUTSetCallbackKeyboard( KeyboardProc );

  /* Parse  command line, handle  default hotkeys, and show messages. */
  DXUTInit();

  DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
  DXUTCreateWindow( L"06_vertex_twisting" );
  DXUTCreateDevice( true, 400, 400 );

  DXUTMainLoop();

  cgDestroyProgram(myCgVertexProgram);
  checkForCgError("destroying vertex program");
  cgDestroyProgram(myCgFragmentProgram);
  checkForCgError("destroying fragment program");
  cgDestroyContext(myCgContext);

  /* This must be called BEFORE the device is destroyed */
  cgD3D10SetDevice( myCgContext, 0 );

  return DXUTGetExitCode();
}


static void createCgPrograms( ID3D10Device * pDev )
{
  ID3D10Blob * pVSBuf  = NULL;
  ID3D10Blob * pPSBuf  = NULL;
  ID3D10Blob * pErrBuf = NULL;

  const char **profileOpts;

  /* Determine the best profiles once a device to be set. */
  myCgVertexProfile = cgD3D10GetLatestVertexProfile();
  checkForCgError("getting latest vertex profile");

  profileOpts = cgD3D10GetOptimalOptions(myCgVertexProfile);
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

  cgD3D10LoadProgram( myCgVertexProgram, 0 );

  // Create vertex input layout
  const D3D10_INPUT_ELEMENT_DESC layout[] =
  {
      { "POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },
      { "COLOR",     0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D10_INPUT_PER_VERTEX_DATA, 0 },      
  };
  UINT numElements = sizeof(layout)/sizeof(layout[0]);

  pVSBuf = cgD3D10GetCompiledProgram( myCgVertexProgram );

  pDev->CreateInputLayout( layout, numElements, pVSBuf->GetBufferPointer(), pVSBuf->GetBufferSize(), &myVertexLayout );

  /* Determine the best profile once a device to be set. */
  myCgFragmentProfile = cgD3D10GetLatestPixelProfile();
  checkForCgError("getting latest fragment profile");

  profileOpts = cgD3D10GetOptimalOptions(myCgFragmentProfile);
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

  cgD3D10LoadProgram( myCgFragmentProgram, 0 );
}

/* Helper function to write vertex positions into vertex buffer. */
static inline void writeVertex(Vertex3 &v,
                               const float p[2], const float c[3])
{
  v.x = FLOAT(p[0]);
  v.y = FLOAT(p[1]);
  v.z = 0;  
  v.r = c[0];
  v.g = c[1];
  v.b = c[2];
}

static void triangleDivide(Vertex3 v[],
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

static HRESULT initVertexBuffer( ID3D10Device * pDev )
{
  HRESULT hr = S_OK;

  const int subdivisions = 5;

  myNumTriangles = numDividedTriangles(subdivisions);
  int numVerts = myNumTriangles * 3;

  int n = 0;
  const float a[2] = { -0.8f, 0.8f },
              b[2] = {  0.8f, 0.8f },
              c[2] = {  0.0f, -0.8f },
              ca[3] = { 0, 0, 1 },
              cb[3] = { 0, 0, 1 },
              cc[3] = { 0.7f, 0.7f, 1 };

  Vertex3 *v = new Vertex3[numVerts];
  triangleDivide(v, n, subdivisions, a, b, c, ca, cb, cc);  

  D3D10_BUFFER_DESC vbDesc;
  vbDesc.ByteWidth      = numVerts * sizeof( Vertex3 );
  vbDesc.Usage          = D3D10_USAGE_DEFAULT;
  vbDesc.BindFlags      = D3D10_BIND_VERTEX_BUFFER;
  vbDesc.CPUAccessFlags = 0;
  vbDesc.MiscFlags      = 0;

  D3D10_SUBRESOURCE_DATA vbInitData;
  ZeroMemory( &vbInitData, sizeof( D3D10_SUBRESOURCE_DATA ) );
  
  vbInitData.pSysMem = v;

  hr = pDev->CreateBuffer( &vbDesc, &vbInitData, &myVB );
  if( hr != S_OK )
  {
      delete [] v;
      return hr;
  }

  delete [] v;

  return S_OK;
}

bool CALLBACK IsD3D10DeviceAcceptable( UINT Adapter, UINT Output, D3D10_DRIVER_TYPE DeviceType, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
   return true;
}

HRESULT CALLBACK OnD3D10CreateDevice( ID3D10Device * pDev, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr = S_OK;

    cgD3D10SetDevice( myCgContext, pDev );
    checkForCgError("setting Direct3D 10 device");

    initVertexBuffer( pDev ); 
    createCgPrograms( pDev );

    // Disable alpha blending
    D3D10_BLEND_DESC BlendState;
    ZeroMemory( &BlendState, sizeof( D3D10_BLEND_DESC ) );
    
    BlendState.BlendEnable[0] = FALSE;
    BlendState.RenderTargetWriteMask[0] = D3D10_COLOR_WRITE_ENABLE_ALL;
    
    hr = pDev->CreateBlendState( &BlendState, &myBlendState_NoBlend );
    if( hr != S_OK )
        return hr;

    // Disable culling
    D3D10_RASTERIZER_DESC RSDesc;
    RSDesc.FillMode = D3D10_FILL_SOLID;
    RSDesc.CullMode = D3D10_CULL_NONE;
    RSDesc.FrontCounterClockwise = FALSE;
    RSDesc.DepthBias = 0;
    RSDesc.DepthBiasClamp = 0;
    RSDesc.SlopeScaledDepthBias = 0;
    RSDesc.ScissorEnable = FALSE;
    RSDesc.MultisampleEnable = FALSE;
    RSDesc.AntialiasedLineEnable = FALSE;

    hr = pDev->CreateRasterizerState( &RSDesc, &myRasterizerState_NoCull );
    if( hr != S_OK )
        return hr;

    return S_OK;
}

static void Clear( ID3D10Device *pDev )
{
    float ClearColor[4] = { 1.0f, 1.0f, 1.0f, 0.0f };
    
    ID3D10RenderTargetView* pRTV = DXUTGetD3D10RenderTargetView();
    pDev->ClearRenderTargetView( pRTV, ClearColor );
    
    ID3D10DepthStencilView* pDSV = DXUTGetD3D10DepthStencilView();
    pDev->ClearDepthStencilView( pDSV, D3D10_CLEAR_DEPTH, 1.0, 0 );
}

void CALLBACK OnD3D10FrameRender( ID3D10Device * pDev, double fTime, float fElapsedTime, void* pUserContext )
{
    UINT strides[1] = { sizeof( Vertex3 ) };
    UINT offsets[1] = { 0 };
    ID3D10Buffer * pBuffers[1] = { myVB };
    
    Clear( pDev );    

    pDev->OMSetBlendState( myBlendState_NoBlend, 0, 0xffffffff );
    pDev->RSSetState( myRasterizerState_NoCull );  

    pDev->IASetVertexBuffers( 0, 1, pBuffers, strides, offsets );
    pDev->IASetInputLayout( myVertexLayout );
    pDev->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

    cgD3D10BindProgram( myCgVertexProgram );
    cgD3D10BindProgram( myCgFragmentProgram );
    
    pDev->Draw( myNumTriangles * 3, 0 ); // numVertices, startVertex
}

void CALLBACK OnD3D10DestroyDevice( void* pUserContext )
{
    SAFE_RELEASE( myVertexLayout );
    SAFE_RELEASE( myVB );
    SAFE_RELEASE( myBlendState_NoBlend );
    SAFE_RELEASE( myRasterizerState_NoCull );
}

static void CALLBACK OnFrameMove(double fTime, 
                                 float fElapsedTime, 
                                 void* pUserContext)
{
  if (myAnimating) {
    if (myTwisting > 3) {
      myTwistDirection = -0.05f;
    } else if (myTwisting < -3) {
      myTwistDirection = 0.05f;
    }
    myTwisting += myTwistDirection;

    cgSetParameter1f( myCgVertexParam_twisting, myTwisting );
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
      } 
      else {        
      }
      break;
    }
  }
}
