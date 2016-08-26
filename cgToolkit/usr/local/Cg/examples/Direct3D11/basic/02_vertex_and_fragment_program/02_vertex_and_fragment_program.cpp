/* 02_vertex_and_fragment_program.c - Direct3D11-based very simple vertex program and
   fragment program example using Cg program from Chapter 2 of "The Cg Tutorial" 
   (Addison-Wesley, ISBN 0321194969). */

#include <windows.h>
#include <stdio.h>
#include <d3d11.h>     /* Can't include this?  Is DirectX SDK installed? */
#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgD3D11.h>

#pragma comment(lib, "d3d11.lib")

#ifndef SAFE_RELEASE
#define SAFE_RELEASE( p ) { if( p ) { ( p )->Release(); ( p ) = NULL; } }
#endif

//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct Vertex3
{
    float x;
    float y;
    float z;
};


//--------------------------------------------------------------------------------------
// Global Variables
//--------------------------------------------------------------------------------------
HINSTANCE                   g_hInst             = NULL;  
HWND                        g_hWnd              = NULL;
D3D_DRIVER_TYPE             g_driverType        = D3D_DRIVER_TYPE_NULL;
ID3D11Device *              g_pDevice           = NULL;
ID3D11DeviceContext *       g_pDeviceContext    = NULL;
IDXGISwapChain *            g_pSwapChain        = NULL;
ID3D11RenderTargetView *    g_pRenderTargetView = NULL;
ID3D11DepthStencilView *    g_pDepthStencilView = NULL;

ID3D11InputLayout *         g_pVertexLayout = NULL;
ID3D11Buffer *              g_pVB           = NULL;

ID3D11BlendState *          myBlendState_NoBlend = NULL;
ID3D11RasterizerState *	    myRasterizerState_NoCull = NULL;

const int Width  = 400;
const int Height = 400;

float ClearColor[4] = { 0.0f, 0.0f, 1.0f, 1.0f }; // RGBA


static CGcontext myCgContext;
static CGprofile myCgVertexProfile;
static CGprogram myCgVertexProgram;
static CGprofile myCgFragmentProfile;
static CGprogram myCgFragmentProgram;

static const TCHAR * myProgramName             = L"02_vertex_and_fragment_program";
static const char  * myVertexProgramFileName   = "C2E1v_green.cg",
/* Page 38 */      * myVertexProgramName       = "C2E1v_green",
                   * myFragmentProgramFileName = "C2E2f_passthru.cg",
/* Page 53 */      * myFragmentProgramName     = "C2E2f_passthru";


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
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_DRIVER_TYPE driverTypes[] = 
    {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_REFERENCE,
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

    D3D_FEATURE_LEVEL FeatureLevels     = D3D_FEATURE_LEVEL_11_0;
    D3D_FEATURE_LEVEL *pFeatureLevel    = NULL;

    for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )
    {
        g_driverType = driverTypes[driverTypeIndex];
        hr = D3D11CreateDeviceAndSwapChain( NULL,                  // Adapter
                                            g_driverType,          // Driver Type
                                            NULL,                  // Software
                                            createDeviceFlags,     // Flags
                                            &FeatureLevels,        // Feature Levels
                                            1,                     // Num Feature Levels
                                            D3D11_SDK_VERSION,     // SDK Version
                                            &sd,                   // Swap Chain Desc
                                            &g_pSwapChain,         // Swap Chain
                                            &g_pDevice,            // Device
                                            pFeatureLevel,         // Feature Level
                                            &g_pDeviceContext );   // Device Context

        if( SUCCEEDED( hr ) )
            break;
    }

    if( FAILED(hr) )
        return hr;

    // Create a render target view
    ID3D11Texture2D *pBuffer;
    hr = g_pSwapChain->GetBuffer( 0, __uuidof( ID3D11Texture2D ), (LPVOID*)&pBuffer );

    if( FAILED(hr) )
        return hr;

    hr = g_pDevice->CreateRenderTargetView( pBuffer, NULL, &g_pRenderTargetView );
    pBuffer->Release();

    if( FAILED(hr) )
        return hr;

    D3D11_TEXTURE2D_DESC DSDesc;
	DSDesc.ArraySize          = 1;
	DSDesc.BindFlags          = D3D11_BIND_DEPTH_STENCIL;
	DSDesc.CPUAccessFlags     = 0;
	DSDesc.Format             = DXGI_FORMAT_D24_UNORM_S8_UINT;
	DSDesc.Width              = Width;
    DSDesc.Height             = Height;	
	DSDesc.MipLevels          = 1;
	DSDesc.MiscFlags          = 0;
	DSDesc.SampleDesc.Count   = 1;
	DSDesc.SampleDesc.Quality = 0;
	DSDesc.Usage              = D3D11_USAGE_DEFAULT;

	ID3D11Texture2D * DSBuffer;
	hr = g_pDevice->CreateTexture2D( &DSDesc, NULL, &DSBuffer );

    if( FAILED(hr) )
        return hr;

    hr = g_pDevice->CreateDepthStencilView( DSBuffer, NULL, &g_pDepthStencilView );

    if( FAILED(hr) )
        return hr;

    g_pDeviceContext->OMSetRenderTargets( 1, &g_pRenderTargetView, g_pDepthStencilView );

    // Setup the viewport
    D3D11_VIEWPORT vp;
    vp.Width    = width;
    vp.Height   = height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pDeviceContext->RSSetViewports( 1, &vp );

    // Disable alpha blending
    D3D11_BLEND_DESC BlendState;
    ZeroMemory( &BlendState, sizeof( D3D11_BLEND_DESC ) );
    
    BlendState.RenderTarget[0].BlendEnable = FALSE;
    BlendState.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    
    hr = g_pDevice->CreateBlendState( &BlendState, &myBlendState_NoBlend );
    if( hr != S_OK )
        return hr;

    // Disable culling
    D3D11_RASTERIZER_DESC RSDesc;
    RSDesc.FillMode              = D3D11_FILL_SOLID;
    RSDesc.CullMode              = D3D11_CULL_NONE;
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
    ID3D10Blob * pVSBuf  = NULL;
    ID3D10Blob * pPSBuf  = NULL;
    ID3D10Blob * pErrBuf = NULL;

    const char ** profileOpts = NULL;

    myCgContext = cgCreateContext();
    checkForCgError( "creating context" );

    hr = cgD3D11SetDevice( myCgContext, g_pDevice );
    checkForCgError( "setting Direct3D device", false );
    if( hr != S_OK )
        return hr;

    /* Determine the best profile once a device to be set. */
    myCgVertexProfile = cgD3D11GetLatestVertexProfile();
    checkForCgError("getting latest profile");

    profileOpts = cgD3D11GetOptimalOptions(myCgVertexProfile);
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

    cgD3D11LoadProgram( myCgVertexProgram, 0 );
    checkForCgError("loading vertex program");
 
    // Create vertex input layout
    const D3D11_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },	  
    };

    pVSBuf = cgD3D11GetCompiledProgram( myCgVertexProgram );

    hr = g_pDevice->CreateInputLayout( layout, 1, pVSBuf->GetBufferPointer(), pVSBuf->GetBufferSize(), &g_pVertexLayout ); 
    if( hr != S_OK )
        return E_FAIL;
  
    myCgFragmentProfile = cgD3D11GetLatestPixelProfile();

    profileOpts = cgD3D11GetOptimalOptions(myCgFragmentProfile);
    checkForCgError("getting latest profile options");

    myCgFragmentProgram = 
        cgCreateProgramFromFile(
            myCgContext,                /* Cg runtime context */
            CG_SOURCE,                  /* Program in human-readable form */
            myFragmentProgramFileName,  /* Name of file containing program */
            myCgFragmentProfile,        /* Profile */
            myFragmentProgramName,      /* Entry function name */
            profileOpts);               /* Pass optimal compiler options */
    checkForCgError("creating fragment program from file");

    cgD3D11LoadProgram( myCgFragmentProgram, 0 );
    checkForCgError("loading fragment program");

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Clean up the objects we've created
//--------------------------------------------------------------------------------------
void CleanupDevice()
{
    SAFE_RELEASE( g_pVertexLayout );
    SAFE_RELEASE( g_pVB );

    if( g_pDeviceContext )      g_pDeviceContext->ClearState();

    if( g_pRenderTargetView )   g_pRenderTargetView->Release();
    if( g_pDepthStencilView )   g_pDepthStencilView->Release();
    if( g_pSwapChain )          g_pSwapChain->Release();
    if( g_pDevice )             g_pDevice->Release();

    SAFE_RELEASE( g_pDevice );
    SAFE_RELEASE( g_pDeviceContext );
}

//--------------------------------------------------------------------------------------
// Clean up the cg objects we've created
//--------------------------------------------------------------------------------------
void CleanupCg()
{
    cgDestroyProgram( myCgVertexProgram );
    checkForCgError( "destroying vertex program" );
    cgDestroyProgram( myCgFragmentProgram );
    checkForCgError( "destroying fragment program" );
  
    cgD3D11SetDevice( myCgContext, NULL );

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

    D3D11_BUFFER_DESC vbDesc;
    vbDesc.ByteWidth      = 3 * sizeof( Vertex3 );
    vbDesc.Usage          = D3D11_USAGE_DEFAULT;
    vbDesc.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    vbDesc.CPUAccessFlags = 0;
    vbDesc.MiscFlags      = 0;

    D3D11_SUBRESOURCE_DATA vbInitData;
    ZeroMemory( &vbInitData, sizeof( D3D11_SUBRESOURCE_DATA ) );

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
    UINT strides[1] = { sizeof( Vertex3 ) };
    UINT offsets[1] = { 0 };
    ID3D11Buffer * pBuffers[1] = { g_pVB };

    // Clear the back buffer        
    g_pDeviceContext->ClearRenderTargetView( g_pRenderTargetView, ClearColor );
    g_pDeviceContext->ClearDepthStencilView( g_pDepthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0, 0 );

    g_pDeviceContext->OMSetBlendState( myBlendState_NoBlend, 0, 0xffffffff );
    g_pDeviceContext->RSSetState( myRasterizerState_NoCull );  
    
    g_pDeviceContext->IASetVertexBuffers( 0, 1, pBuffers, strides, offsets );
    g_pDeviceContext->IASetInputLayout( g_pVertexLayout );    
    g_pDeviceContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );   

    cgD3D11BindProgram( myCgVertexProgram );
    cgD3D11BindProgram( myCgFragmentProgram );

    g_pDeviceContext->Draw( 3, 0 ); // numVertices, startVertex
     
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