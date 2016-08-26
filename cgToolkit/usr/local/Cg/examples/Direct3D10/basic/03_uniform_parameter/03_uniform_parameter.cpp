
/* 03_uniform_parameter.c - Direct3D10-based uniform parameter example
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
static CGparameter myCgVertexParam_constantColor;

static const WCHAR *myProgramNameW = L"03_uniform_parameter";
static const char *myProgramName = "03_uniform_parameter",
                  *myVertexProgramFileName = "C3E1v_anycolor.cg",
/* Page 62 */     *myVertexProgramName = "C3E1v_anycolor",
                  *myFragmentProgramFileName = "C2E2f_passthru.cg",
/* Page 53 */     *myFragmentProgramName = "C2E2f_passthru";

ID3D10InputLayout *      myVertexLayout = NULL;
ID3D10Buffer *           myVB = NULL;
ID3D10BlendState *       myBlendState_NoBlend = NULL;
ID3D10RasterizerState *	 myRasterizerState_NoCull = NULL;

struct Vertex3
{
    float x;
    float y;
    float z;
};

struct SceneVertex
{
    Vertex3 pos;
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

INT WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
  myCgContext = cgCreateContext();
  checkForCgError("creating context");
  cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);

  DXUTSetCallbackD3D10DeviceAcceptable( IsD3D10DeviceAcceptable );
  DXUTSetCallbackD3D10DeviceCreated( OnD3D10CreateDevice );
  DXUTSetCallbackD3D10DeviceDestroyed( OnD3D10DestroyDevice );
  DXUTSetCallbackD3D10FrameRender( OnD3D10FrameRender );

  /* Parse  command line, handle  default hotkeys, and show messages. */
  DXUTInit();

  DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
  DXUTCreateWindow( L"03_uniform_parameter" );
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

  const char **profileOpts = NULL;

  /* Determine the best profile once a device to be set. */
  myCgVertexProfile = cgD3D10GetLatestVertexProfile();
  checkForCgError("getting latest profile");

  profileOpts = cgD3D10GetOptimalOptions(myCgVertexProfile);
  checkForCgError("getting latest profile options");

  myCgVertexProgram =
    cgCreateProgramFromFile(
      myCgContext,              /* Cg runtime context */
      CG_SOURCE,                /* Program in human-readable form */
      myVertexProgramFileName,  /* Name of file containing program */
      myCgVertexProfile,        /* Profile */
      myVertexProgramName,      /* Entry function name */
      profileOpts);             /* Pass optimal compiler options */
  checkForCgError("creating vertex program from file");

  cgD3D10LoadProgram( myCgVertexProgram, 0 );

  // Create vertex input layout
  const D3D10_INPUT_ELEMENT_DESC layout[] =
  {
      { "POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D10_INPUT_PER_VERTEX_DATA, 0 },	  
  };

  pVSBuf = cgD3D10GetCompiledProgram( myCgVertexProgram );

  pDev->CreateInputLayout( layout, 1, pVSBuf->GetBufferPointer(), pVSBuf->GetBufferSize(), &myVertexLayout );

  myCgVertexParam_constantColor = cgGetNamedParameter(myCgVertexProgram, "constantColor");
  checkForCgError("could not get constantColor parameter");

  const float green[3] = { 0.2f, 0.8f, 0.3f };
  cgSetParameter3fv(myCgVertexParam_constantColor, green);
   
  myCgFragmentProfile = cgD3D10GetLatestPixelProfile();

  profileOpts = cgD3D10GetOptimalOptions(myCgFragmentProfile);
  checkForCgError("getting latest profile options");

  myCgFragmentProgram = 
    cgCreateProgramFromFile(
      myCgContext,                /* Cg runtime context */
      CG_SOURCE,                  /* Program in human-readable form */
      myFragmentProgramFileName,  /* Name of file containing program */
      myCgFragmentProfile,        /* Profile */
      myFragmentProgramName,      /* Entry function name */
      profileOpts);               /* Pass optimal compiler options */

  cgD3D10LoadProgram( myCgFragmentProgram, 0 );
}

static HRESULT initVertexBuffer( ID3D10Device * pDev )
{
  HRESULT hr = S_OK;

  /* Initialize three vertices for rendering a triangle. */
  static const SceneVertex triangleVertices[3] = 
  {
    { -0.8f,  0.8f, 0.0f },
    {  0.8f,  0.8f, 0.0f },
    {  0.0f, -0.8f, 0.0f }
  };
  
  D3D10_BUFFER_DESC vbDesc;
  vbDesc.ByteWidth      = 3 * sizeof( SceneVertex );
  vbDesc.Usage          = D3D10_USAGE_DEFAULT;
  vbDesc.BindFlags      = D3D10_BIND_VERTEX_BUFFER;
  vbDesc.CPUAccessFlags = 0;
  vbDesc.MiscFlags      = 0;

  D3D10_SUBRESOURCE_DATA vbInitData;
  ZeroMemory( &vbInitData, sizeof( D3D10_SUBRESOURCE_DATA ) );
  
  vbInitData.pSysMem = triangleVertices;

  hr = pDev->CreateBuffer( &vbDesc, &vbInitData, &myVB );
  if( hr != S_OK )
      return hr;

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
    float ClearColor[4] = { 0.1f, 0.3f, 0.6f, 0.0f };
    
    ID3D10RenderTargetView* pRTV = DXUTGetD3D10RenderTargetView();
    pDev->ClearRenderTargetView( pRTV, ClearColor );
    
    ID3D10DepthStencilView* pDSV = DXUTGetD3D10DepthStencilView();
    pDev->ClearDepthStencilView( pDSV, D3D10_CLEAR_DEPTH, 1.0, 0 );
}

void CALLBACK OnD3D10FrameRender( ID3D10Device * pDev, double fTime, float fElapsedTime, void* pUserContext )
{
    UINT strides[1] = { sizeof( SceneVertex ) };
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

    pDev->Draw( 3, 0 ); // numVertices, startVertex
}

void CALLBACK OnD3D10DestroyDevice( void* pUserContext )
{
    SAFE_RELEASE( myVertexLayout );
    SAFE_RELEASE( myVB );
    SAFE_RELEASE( myBlendState_NoBlend );
    SAFE_RELEASE( myRasterizerState_NoCull );
}
