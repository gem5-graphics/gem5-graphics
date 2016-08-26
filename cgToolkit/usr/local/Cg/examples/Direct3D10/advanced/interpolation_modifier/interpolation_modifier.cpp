
/* interpolation_modifier.cpp - interpolation modifiers (FLAT,
                              NOPERSPECTIVE, and CENTROID) */

/* This example renders the same triangle into four different viewport.
   Each viewport uses a different technique that passes through the an
   interpolated texture coordinate set as the fragment color but with
   different interpolation modifiers.  */


#include <windows.h>
#include <stdio.h>
#include <assert.h>

#include <d3d10_1.h>     /* Can't include this?  Is DirectX SDK installed? */
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgD3D10.h>

#pragma comment(lib, "d3d10.lib")

#ifndef SAFE_RELEASE
#define SAFE_RELEASE( p ) { if( p ) { ( p )->Release(); ( p ) = NULL; } }
#endif

//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct Vertex4
{
    float x;
    float y;
    float z;
    float w;
    float s;
    float t;
    float r;
    float q;
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
ID3D10DepthStencilView *    g_pDepthStencilView = NULL;

ID3D10InputLayout *         g_pVertexLayout = NULL;
ID3D10Buffer *              g_pVB           = NULL;

ID3D10BlendState *          myBlendState_NoBlend = NULL;
ID3D10RasterizerState *	    myRasterizerState_NoCull = NULL;

const int Width  = 600;
const int Height = 600;

float ClearColor[4] = { 0.0f, 0.0f, 1.0f, 1.0f }; // RGBA


CGcontext    myCgContext;
CGeffect     myCgEffect;
CGtechnique  myCgTechnique;

CGprogram myCgCombinedProgram;

static const TCHAR * myProgramName = L"interpolation_modifier";
const char *  myCgFXFileName       = "interpolation_modifier.cgfx";

const char *myCgTechniqueNames[4] =
{
  "NormalColorInterpolation",
  "FlatColorInterpolation",
  "NoPerspectiveColorInterpolation",
  "CentroidColorInterpolation"
};

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
void                DrawTriangleWithEffect( int x, int y, const char *techniqueName );
void                DrawTriangle();

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
        hr = D3D10CreateDeviceAndSwapChain( NULL,                  // Adapter
                                            g_driverType,          // Driver Type
                                            NULL,                  // Software
                                            createDeviceFlags,     // Flags
                                            D3D10_SDK_VERSION,     // SDK Version
                                            &sd,                   // Swap Chain Desc
                                            &g_pSwapChain,         // Swap Chain
                                            &g_pDevice             // Device
                                           );   // Device Context

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

    D3D10_TEXTURE2D_DESC DSDesc;
	DSDesc.ArraySize          = 1;
	DSDesc.BindFlags          = D3D10_BIND_DEPTH_STENCIL;
	DSDesc.CPUAccessFlags     = 0;
	DSDesc.Format             = DXGI_FORMAT_D24_UNORM_S8_UINT;
	DSDesc.Width              = Width;
    DSDesc.Height             = Height;	
	DSDesc.MipLevels          = 1;
	DSDesc.MiscFlags          = 0;
	DSDesc.SampleDesc.Count   = 1;
	DSDesc.SampleDesc.Quality = 0;
	DSDesc.Usage              = D3D10_USAGE_DEFAULT;

	ID3D10Texture2D * DSBuffer;
	hr = g_pDevice->CreateTexture2D( &DSDesc, NULL, &DSBuffer );

    if( FAILED(hr) )
        return hr;

    hr = g_pDevice->CreateDepthStencilView( DSBuffer, NULL, &g_pDepthStencilView );

    if( FAILED(hr) )
        return hr;

    g_pDevice->OMSetRenderTargets( 1, &g_pRenderTargetView, g_pDepthStencilView );

    // Setup the viewport
    D3D10_VIEWPORT vp;
    vp.Width    = width;
    vp.Height   = height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pDevice->RSSetViewports( 1, &vp );

    // Disable alpha blending
    D3D10_BLEND_DESC BlendState;
    ZeroMemory( &BlendState, sizeof( D3D10_BLEND_DESC ) );
    
    BlendState.BlendEnable[0] = FALSE;
    BlendState.RenderTargetWriteMask[0] = D3D10_COLOR_WRITE_ENABLE_ALL;
    
    hr = g_pDevice->CreateBlendState( &BlendState, &myBlendState_NoBlend );
    if( hr != S_OK )
        return hr;

    // Disable culling
    D3D10_RASTERIZER_DESC RSDesc;
    RSDesc.FillMode              = D3D10_FILL_SOLID;
    RSDesc.CullMode              = D3D10_CULL_NONE;
    RSDesc.FrontCounterClockwise = FALSE;
    RSDesc.DepthBias             = 0;
    RSDesc.DepthBiasClamp        = 0;
    RSDesc.SlopeScaledDepthBias  = 0;
    RSDesc.ScissorEnable         = FALSE;
    RSDesc.MultisampleEnable     = FALSE;
    RSDesc.AntialiasedLineEnable = FALSE;

    hr = g_pDevice->CreateRasterizerState( &RSDesc, &myRasterizerState_NoCull );
    if( hr != S_OK )
        return hr;

    return S_OK;
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


    myCgEffect = cgCreateEffectFromFile( myCgContext, myCgFXFileName, NULL );
    checkForCgError( "creating cgfx_texture.cgfx effect" );
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

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Clean up the objects we've created
//--------------------------------------------------------------------------------------
void CleanupDevice()
{
    SAFE_RELEASE( g_pVertexLayout );
    SAFE_RELEASE( g_pVB );

    if( g_pRenderTargetView )   g_pRenderTargetView->Release();
    if( g_pDepthStencilView )   g_pDepthStencilView->Release();
    if( g_pSwapChain )          g_pSwapChain->Release();
    if( g_pDevice )
    {
        g_pDevice->ClearState();
        g_pDevice->Release();
    }

    SAFE_RELEASE( g_pDevice );
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
    const Vertex4 triangleVertices[3] = 
    {
        { -0.7f,  0.7f,        0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f },
        {  0.7f,  0.7f,        0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f },
        {  0.0f, -0.7f * 5.0f, 0.0f, 5.0f, 0.0f, 0.0f, 1.0f, 0.0f  } // give perspective
    };

    D3D10_BUFFER_DESC vbDesc;
    vbDesc.ByteWidth      = 3 * sizeof( Vertex4 );
    vbDesc.Usage          = D3D10_USAGE_DEFAULT;
    vbDesc.BindFlags      = D3D10_BIND_VERTEX_BUFFER;
    vbDesc.CPUAccessFlags = 0;
    vbDesc.MiscFlags      = 0;

    D3D10_SUBRESOURCE_DATA vbInitData;
    ZeroMemory( &vbInitData, sizeof( D3D10_SUBRESOURCE_DATA ) );

    vbInitData.pSysMem = triangleVertices;

    // Create vertex layout
    const D3D10_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D10_INPUT_PER_VERTEX_DATA, 0 },
    };

    UINT numElements = sizeof( layout ) / sizeof( layout[0] );

    CGpass myPass = cgGetFirstPass( myCgTechnique );

    ID3D10Blob * pVSBuf = cgD3D10GetIASignatureByPass( myPass );

    hr = g_pDevice->CreateInputLayout( layout, numElements, pVSBuf->GetBufferPointer(), pVSBuf->GetBufferSize(), &g_pVertexLayout );     
    if( hr != S_OK )
    {
        return NULL;
    }

    hr = g_pDevice->CreateBuffer( &vbDesc, &vbInitData, &g_pVB );
    if( hr != S_OK )
        return hr;

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Render a frame
//--------------------------------------------------------------------------------------
void DrawTriangle()
{
    UINT strides[1] = { sizeof( Vertex4 ) };
    UINT offsets[1] = { 0 };
    ID3D10Buffer * pBuffers[1] = { g_pVB };

    g_pDevice->IASetVertexBuffers( 0, 1, pBuffers, strides, offsets );
    g_pDevice->IASetInputLayout( g_pVertexLayout );    
    g_pDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST ); 

    g_pDevice->Draw( 3, 0 ); // numVertices, startVertex
}

void DrawTriangleWithEffect( int x, int y, const char *techniqueName )
{
    CGtechnique technique = cgGetNamedTechnique( myCgEffect, techniqueName );
    CGbool isValid = cgValidateTechnique( technique );

    int viewWidth = Width / 2;
    int viewHeight = Height / 2;

    D3D10_VIEWPORT vp;
    vp.Width    = (float)viewWidth;
    vp.Height   = (float)viewHeight;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = x * viewWidth;
    vp.TopLeftY = y * viewHeight;

    g_pDevice->RSSetViewports( 1, &vp );

    if( isValid )
    {
        for( CGpass pass = cgGetFirstPass(technique); pass; pass = cgGetNextPass(pass) )
        {
            cgSetPassState(pass);
    
                DrawTriangle();

            cgResetPassState(pass);            
        }
    }
}

void Render()
{
    // Clear the back buffer        
    g_pDevice->ClearRenderTargetView( g_pRenderTargetView, ClearColor );
    g_pDevice->ClearDepthStencilView( g_pDepthStencilView, D3D10_CLEAR_DEPTH | D3D10_CLEAR_STENCIL, 1.0, 0 );

    /* Lower-left */
    DrawTriangleWithEffect( 0, 0, myCgTechniqueNames[0] );
    /* Lower-right */
    DrawTriangleWithEffect( 1, 0, myCgTechniqueNames[1] );
    /* Upper-left */
    DrawTriangleWithEffect( 0, 1, myCgTechniqueNames[2] );
    /* Upper-right */
    DrawTriangleWithEffect( 1, 1, myCgTechniqueNames[3] );
     
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