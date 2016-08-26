/* cgfx_simple.cpp  */

#include <windows.h>
#include <stdio.h>
#include <assert.h>
#include <d3d10_1.h>     /* Direct3D10 API: Can't include this?  Is DirectX SDK installed? */
#include <d3d10.h>       /* Direct3D10 API: Can't include this?  Is DirectX SDK installed? */
#include <Cg/cg.h>       /* Can't include this?  Is Cg Toolkit installed? */
#include <Cg/cgD3D10.h>  /* Can't include this?  Is Cg Toolkit installed? */

#pragma comment(lib, "d3d10.lib")

#ifndef SAFE_RELEASE
#define SAFE_RELEASE( p ) { if( p ) { ( p )->Release(); ( p ) = NULL; } }
#endif

//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct Vertex3
{
    float x, y, z;
};

//--------------------------------------------------------------------------------------
// Global Variables
//--------------------------------------------------------------------------------------
HINSTANCE                   g_hInst             = NULL;  
HWND                        g_hWnd              = NULL;
D3D10_DRIVER_TYPE           g_driverType        = D3D10_DRIVER_TYPE_NULL;
ID3D10Device *              g_pDevice           = NULL;
IDXGISwapChain *            g_pSwapChain        = NULL;
ID3D10RenderTargetView *    g_pRenderTargetView = NULL;

ID3D10InputLayout *         g_pVertexLayout = NULL;
ID3D10Buffer *              g_pVB           = NULL;

const int Width  = 400;
const int Height = 400;

float ClearColor[4] = { 0.0f, 0.0f, 1.0f, 1.0f }; // RGBA

/* Cg global variables */
CGcontext    myCgContext;
CGeffect     myCgEffect;
CGtechnique  myCgTechnique;

const TCHAR * myProgramName  = L"cgfx_simple";
const char *  myCgFXFileName = "cgfx_simple.cgfx"; // Use char here since that's what the Cg runtime expects

//--------------------------------------------------------------------------------------
// Forward declarations
//--------------------------------------------------------------------------------------
HRESULT             InitWindow( HINSTANCE, int );
HRESULT             InitDevice();
void                CleanupDevice();
LRESULT CALLBACK    WndProc( HWND, UINT, WPARAM, LPARAM );
void                Render();
HRESULT             InitCg();
void                CleanupCg();

HRESULT             CreateVertexBuffer();

//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI WinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow )
{
    MSG msg = { 0 };

    if( FAILED( InitWindow( hInstance, nCmdShow ) ) )
        return 0;

    if( FAILED( InitDevice() ) )
    {
        CleanupDevice();
        return 0;
    }

    if( FAILED( InitCg() ) || FAILED( CreateVertexBuffer() ) )
    {
        CleanupDevice();
        CleanupCg();
        return 0;
    }
    
    // Main message loop    
    while( WM_QUIT != msg.message )
    {
        if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
        {
            TranslateMessage( &msg );
            DispatchMessage( &msg );
        }
        else
        {
            Render();  
        }
    }

    CleanupDevice();
    CleanupCg();

    return (int)msg.wParam;
}

//--------------------------------------------------------------------------------------
// Register class and create window
//--------------------------------------------------------------------------------------
HRESULT InitWindow( HINSTANCE hInstance, int nCmdShow )
{
    g_hInst = hInstance; 
    RECT rc = { 0, 0, Width, Height };

    // Register class
    WNDCLASSEX wcex;
    wcex.cbSize         = sizeof(WNDCLASSEX); 
    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = NULL;
    wcex.hCursor        = LoadCursor( NULL, IDC_ARROW );
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName   = NULL;
    wcex.lpszClassName  = myProgramName;
    wcex.hIconSm        = NULL;

    if( !RegisterClassEx( &wcex ) )
        return E_FAIL;

    // Create window    
    AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
    g_hWnd = CreateWindow( myProgramName,          // Class name
                           myProgramName,          // Window name
                           WS_OVERLAPPEDWINDOW,    // Style
                           CW_USEDEFAULT,          // X position
                           CW_USEDEFAULT,          // Y position
                           rc.right - rc.left,     // Width
                           rc.bottom - rc.top,     // Height
                           NULL,                   // Parent HWND
                           NULL,                   // Menu
                           hInstance,              // Instance
                           NULL                    // Param
                         );
    if( !g_hWnd )
        return E_FAIL;

    ShowWindow( g_hWnd, nCmdShow );
   
    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create Direct3D device and swap chain
//--------------------------------------------------------------------------------------
HRESULT InitDevice()
{
    HRESULT hr = S_OK;

    RECT rc;
    GetClientRect( g_hWnd, &rc );
    UINT width  = rc.right  - rc.left;
    UINT height = rc.bottom - rc.top;

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D10_CREATE_DEVICE_DEBUG;
#endif

    D3D10_DRIVER_TYPE driverTypes[] = 
    {
        D3D10_DRIVER_TYPE_HARDWARE,
        D3D10_DRIVER_TYPE_REFERENCE,
    };

    UINT numDriverTypes = sizeof( driverTypes ) / sizeof( driverTypes[0] );

    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory( &sd, sizeof( sd ) );
    sd.BufferCount                        = 1;
    sd.BufferDesc.Width                   = width;
    sd.BufferDesc.Height                  = height;
    sd.BufferDesc.Format                  = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator   = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage                        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow                       = g_hWnd;
    sd.SampleDesc.Count                   = 1;
    sd.SampleDesc.Quality                 = 0;
    sd.Windowed                           = TRUE;

    for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )
    {
        g_driverType = driverTypes[driverTypeIndex];
        hr = D3D10CreateDeviceAndSwapChain( NULL, g_driverType, NULL, createDeviceFlags, 
                                            D3D10_SDK_VERSION, &sd, &g_pSwapChain, &g_pDevice );
        if( SUCCEEDED( hr ) )
            break;
    }

    if( FAILED(hr) )
        return hr;

    // Create a render target view
    ID3D10Texture2D *pBuffer;
    hr = g_pSwapChain->GetBuffer( 0, __uuidof( ID3D10Texture2D ), (LPVOID*)&pBuffer );

    if( FAILED(hr) )
        return hr;

    hr = g_pDevice->CreateRenderTargetView( pBuffer, NULL, &g_pRenderTargetView );
    pBuffer->Release();

    if( FAILED(hr) )
        return hr;

    g_pDevice->OMSetRenderTargets( 1, &g_pRenderTargetView, NULL );

    // Setup the viewport
    D3D10_VIEWPORT vp;
    vp.Width    = width;
    vp.Height   = height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pDevice->RSSetViewports( 1, &vp );      

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Utility function for reporting Cg errors
//--------------------------------------------------------------------------------------
static void checkForCgError( const char * situation, bool _exit = true )
{
    CGerror error;
    const char *string = cgGetLastErrorString(&error);

    if( error != CG_NO_ERROR ) 
    {
        if( error == CG_COMPILER_ERROR ) 
        {
            fprintf(stderr,
                "Program: %s\n"
                "Situation: %s\n"
                "Error: %s\n\n"
                "Cg compiler output...\n%s",
                myProgramName, situation, string,
                cgGetLastListing(myCgContext));
        } 
        else 
        {
            fprintf(stderr,
                "Program: %s\n"
                "Situation: %s\n"
                "Error: %s",
                myProgramName, situation, string);
        }

        if( _exit )
            exit(1);
    }
}

//--------------------------------------------------------------------------------------
// Create Cg objects
//--------------------------------------------------------------------------------------
HRESULT InitCg()
{
    HRESULT hr = S_OK;

    myCgContext = cgCreateContext();
    checkForCgError( "creating context" );

    hr = cgD3D10SetDevice( myCgContext, g_pDevice );
    checkForCgError( "setting Direct3D device", false );
    if( hr != S_OK )
        return hr;

    cgD3D10RegisterStates( myCgContext );
    checkForCgError( "registering standard CgFX states" );
    
    cgD3D10SetManageTextureParameters( myCgContext, CG_TRUE );
    checkForCgError( "manage texture parameters" );


    char buffer[128];
    myCgEffect = cgCreateEffectFromFile( myCgContext, myCgFXFileName, NULL );
    sprintf_s( buffer, "creating %s effect", myCgFXFileName );
    checkForCgError( buffer );
    assert( myCgEffect );

    myCgTechnique = cgGetFirstTechnique( myCgEffect );
    while( myCgTechnique && cgValidateTechnique( myCgTechnique ) == CG_FALSE ) 
    {
        fprintf( stderr, "%s: Technique %s did not validate.  Skipping.\n", myProgramName, cgGetTechniqueName( myCgTechnique ) );
        myCgTechnique = cgGetNextTechnique( myCgTechnique );
    }
  
    if( myCgTechnique ) 
    {
        fprintf( stderr, "%s: Use technique %s.\n", myProgramName, cgGetTechniqueName( myCgTechnique ) );
    } 
    else 
    {
        fprintf( stderr, "%s: No valid technique\n", myProgramName );
        return E_FAIL;
    }
  
    // Create vertex input layout
    const D3D10_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D10_INPUT_PER_VERTEX_DATA, 0 },	  
    };

    CGpass myPass = cgGetFirstPass( myCgTechnique );

    ID3D10Blob * pVSBuf = cgD3D10GetIASignatureByPass( myPass );

    hr = g_pDevice->CreateInputLayout( layout, 1, pVSBuf->GetBufferPointer(), pVSBuf->GetBufferSize(), &g_pVertexLayout ); 
    if( hr != S_OK )
        return E_FAIL;

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Clean up the objects we've created
//--------------------------------------------------------------------------------------
void CleanupDevice()
{
    SAFE_RELEASE( g_pVertexLayout );
    SAFE_RELEASE( g_pVB );

    if( g_pDevice )             g_pDevice->ClearState();

    if( g_pRenderTargetView )   g_pRenderTargetView->Release();
    if( g_pSwapChain )          g_pSwapChain->Release();
    if( g_pDevice )             g_pDevice->Release();
}

//--------------------------------------------------------------------------------------
// Clean up the cg objects we've created
//--------------------------------------------------------------------------------------
void CleanupCg()
{
    cgDestroyEffect( myCgEffect );
    checkForCgError( "destroying effect" );
  
    cgD3D10SetDevice( myCgContext, NULL );

    cgDestroyContext( myCgContext );   
}

//--------------------------------------------------------------------------------------
// Create vertex buffer for triangle
//--------------------------------------------------------------------------------------
HRESULT CreateVertexBuffer()
{
    HRESULT hr = S_OK;

  /* Initialize three vertices for rendering a triangle. */
  static const Vertex3 triangleVertices[3] = 
  {
    { -0.8f,  0.8f, 0.0f },
    {  0.8f,  0.8f, 0.0f },
    {  0.0f, -0.8f, 0.0f }
  };
  
  D3D10_BUFFER_DESC vbDesc;
  vbDesc.ByteWidth      = 3 * sizeof( Vertex3 );
  vbDesc.Usage          = D3D10_USAGE_DEFAULT;
  vbDesc.BindFlags      = D3D10_BIND_VERTEX_BUFFER;
  vbDesc.CPUAccessFlags = 0;
  vbDesc.MiscFlags      = 0;

  D3D10_SUBRESOURCE_DATA vbInitData;
  ZeroMemory( &vbInitData, sizeof( D3D10_SUBRESOURCE_DATA ) );
  
  vbInitData.pSysMem = triangleVertices;

  hr = g_pDevice->CreateBuffer( &vbDesc, &vbInitData, &g_pVB );
  if( hr != S_OK )
      return hr;

  return S_OK;
}

//--------------------------------------------------------------------------------------
// Render a frame
//--------------------------------------------------------------------------------------
void Render()
{   
    CGpass pass;
    UINT strides[1] = { sizeof( Vertex3 ) };
    UINT offsets[1] = { 0 };
    ID3D10Buffer * pBuffers[1] = { g_pVB };

    // Clear the back buffer        
    g_pDevice->ClearRenderTargetView( g_pRenderTargetView, ClearColor );
    
    g_pDevice->IASetVertexBuffers( 0, 1, pBuffers, strides, offsets );
    g_pDevice->IASetInputLayout( g_pVertexLayout );    
    g_pDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST );   

    pass = cgGetFirstPass( myCgTechnique );
    while( pass ) 
    {
        cgSetPassState( pass );
    
        g_pDevice->Draw( 3, 0 );
    
        cgResetPassState( pass );
        pass = cgGetNextPass( pass );
    }
     
    g_pSwapChain->Present( 0, 0 );
}

//--------------------------------------------------------------------------------------
// Called every time the application receives a message
//--------------------------------------------------------------------------------------
LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam )
{
    switch( message ) 
    {
        case WM_KEYDOWN:
            switch( wParam )
            {
                case VK_ESCAPE:
                    PostQuitMessage( 0 );
                    break;
            }
            break;

        case WM_DESTROY:
            PostQuitMessage( 0 );
            break;

        default:
            return DefWindowProc( hWnd, message, wParam, lParam );
    }

    return 0;
}
