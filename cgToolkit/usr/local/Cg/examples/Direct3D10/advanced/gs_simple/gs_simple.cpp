
/* gs_simple.cpp - Direct3D10-based introductory geometry shader example
   using a pass-through geometry shader to draw a pattern of colored
   stars. */

/* Requires the Cg runtime (version 2.2 or higher). */

#include <windows.h>
#include <stdio.h>
#include <math.h>

#include <d3d10_1.h>
#include <d3d10.h>     /* Can't include this?  Is DirectX SDK installed? */

#include <Cg/cg.h>     /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgD3D10.h>

#pragma comment(lib, "d3d10.lib")

#ifndef SAFE_RELEASE
#define SAFE_RELEASE( p ) { if( p ) { ( p )->Release(); ( p ) = NULL; } }
#endif

//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct Vertex2
{
    float x, y;
    float r, g, b, a;
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

ID3D10BlendState *          g_pBlendState_NoBlend     = NULL;
ID3D10RasterizerState *	    g_pRasterizerState_NoCull = NULL;

ID3D10Buffer * redStar    = NULL;
ID3D10Buffer * greenStar  = NULL;
ID3D10Buffer * blueStar   = NULL;
ID3D10Buffer * cyanStar   = NULL;
ID3D10Buffer * yellowStar = NULL;
ID3D10Buffer * grayStar   = NULL;

const int Width  = 400;
const int Height = 400;

float ClearColor[4] = { 0.1f, 0.3f, 0.6f, 1.0f }; // RGBA

CGcontext   myCgContext;
CGprogram   myCgVertexProgram;
CGprogram   myCgGeometryProgram;
CGprogram   myCgFragmentProgram;

CGprofile   myCgVertexProfile;
CGprofile   myCgGeometryProfile;
CGprofile   myCgFragmentProfile;

const char * myProgramName             = "gs_simple",
           
           * myVertexProgramFileName   = "gs_simple.cg",
           * myVertexProgramName       = "vertex_passthru",

           * myGeometryProgramFileName = "gs_simple.cg",
           * myGeometryProgramName     = "geometry_passthru",

           * myFragmentProgramFileName = "gs_simple.cg",
           * myFragmentProgramName     = "fragment_passthru";

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

HRESULT             CreateAllStars();
void                CleanupAllStars();


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

    if( FAILED( InitCg() ) || FAILED( CreateAllStars() ) )
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

    CleanupAllStars();
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
    wcex.lpszClassName  = L"gs_simple";
    wcex.hIconSm        = NULL;

    if( !RegisterClassEx( &wcex ) )
        return E_FAIL;

    // Create window    
    AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
    g_hWnd = CreateWindow( L"gs_simple",          // Class name
                           L"gs_simple",          // Window name
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

    g_driverType = D3D10_DRIVER_TYPE_HARDWARE;  

    // uncomment this line to use the software reference driver
    // g_driverType = D3D10_DRIVER_TYPE_REFERENCE;

    hr = D3D10CreateDeviceAndSwapChain( NULL, g_driverType, NULL, createDeviceFlags, 
                                        D3D10_SDK_VERSION, &sd, &g_pSwapChain, &g_pDevice );

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

    // Disable alpha blending
    D3D10_BLEND_DESC BlendState;
    ZeroMemory( &BlendState, sizeof( D3D10_BLEND_DESC ) );
    
    BlendState.BlendEnable[0]           = FALSE;
    BlendState.RenderTargetWriteMask[0] = D3D10_COLOR_WRITE_ENABLE_ALL;
    
    hr = g_pDevice->CreateBlendState( &BlendState, &g_pBlendState_NoBlend );
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

    hr = g_pDevice->CreateRasterizerState( &RSDesc, &g_pRasterizerState_NoCull );
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
    checkForCgError( "setting Direct3D device" );
    if( hr != S_OK )
        return hr;

    myCgVertexProfile = cgD3D10GetLatestVertexProfile();
    checkForCgError("selecting vertex profile");

    myCgVertexProgram =
        cgCreateProgramFromFile(
            myCgContext,              /* Cg runtime context */
            CG_SOURCE,                /* Program in human-readable form */
            myVertexProgramFileName,  /* Name of file containing program */
            myCgVertexProfile,        /* Profile: OpenGL ARB vertex program */
            myVertexProgramName,      /* Entry function name */
            NULL );                   /* No extra compiler options */

    checkForCgError( "creating vertex program from file" );
    cgD3D10LoadProgram( myCgVertexProgram, 0 );
    checkForCgError( "loading vertex program" );

    myCgGeometryProfile = cgD3D10GetLatestGeometryProfile();
    if( myCgGeometryProfile == CG_PROFILE_UNKNOWN ) 
    {
        fprintf(stderr, "%s: geometry profile is not available.\n", myProgramName);
        exit(0);
    }    
    checkForCgError("selecting geometry profile");

    myCgGeometryProgram =
        cgCreateProgramFromFile(
            myCgContext,                /* Cg runtime context */
            CG_SOURCE,                  /* Program in human-readable form */
            myGeometryProgramFileName,  /* Name of file containing program */
            myCgGeometryProfile,        /* Profile: OpenGL ARB geometry program */
            myGeometryProgramName,      /* Entry function name */
            NULL );                      /* No extra compiler options */

    checkForCgError( "creating geometry program from file" );
    cgD3D10LoadProgram( myCgGeometryProgram, 0 );
    checkForCgError( "loading geometry program" );

    myCgFragmentProfile = cgD3D10GetLatestPixelProfile();
    checkForCgError( "selecting fragment profile" );

    myCgFragmentProgram =
        cgCreateProgramFromFile(
            myCgContext,                /* Cg runtime context */
            CG_SOURCE,                  /* Program in human-readable form */
            myFragmentProgramFileName,  /* Name of file containing program */
            myCgFragmentProfile,        /* Profile: OpenGL ARB fragment program */
            myFragmentProgramName,      /* Entry function name */
            NULL) ;                     /* No extra compiler options */

    checkForCgError( "creating fragment program from file" );
    cgD3D10LoadProgram( myCgFragmentProgram, 0 );
    checkForCgError( "loading fragment program" );
 
    // Create vertex input layout
    const D3D10_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION",  0, DXGI_FORMAT_R32G32_FLOAT,       0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",     0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 8, D3D10_INPUT_PER_VERTEX_DATA, 0 },
    };

    UINT numElements = sizeof(layout)/sizeof(layout[0]);

    ID3D10Blob * pVSBuf = cgD3D10GetCompiledProgram( myCgVertexProgram );

    hr = g_pDevice->CreateInputLayout( layout, numElements, pVSBuf->GetBufferPointer(), pVSBuf->GetBufferSize(), &g_pVertexLayout ); 
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
    SAFE_RELEASE( g_pBlendState_NoBlend );
    SAFE_RELEASE( g_pRasterizerState_NoCull );
    
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
    cgDestroyProgram( myCgVertexProgram   );
    cgDestroyProgram( myCgGeometryProgram );
    cgDestroyProgram( myCgFragmentProgram );
  
    cgD3D10SetDevice( myCgContext, NULL );

    cgDestroyContext( myCgContext );   
}

//--------------------------------------------------------------------------------------
// Create stars
//--------------------------------------------------------------------------------------
void EmitVertex( Vertex2 * vert, float x, float y, float * color )
{
    vert->x = x;
    vert->y = y;
    vert->r = color[0];
    vert->g = color[1];
    vert->b = color[2];
    vert->a = 1.0f;
}

ID3D10Buffer * createStar( float x, float y, float R, float r, float * color )
{
    ID3D10Buffer * vBuff = NULL;

    float piOverStarPoints = 3.14159f / 5.0f; // 5 points in a star
    float angle = 0.0;

    static Vertex2 verts[30];
    float white[] = { 1.0f, 1.0f, 1.0f };

    int currVert = 0;
    for( int i = 0; i < 5; ++i )
    {
        // Make two triangles per iteration

        // First triangle
        EmitVertex( &verts[currVert], x, y, white );
        ++currVert;

        EmitVertex( &verts[currVert], x + R * cos( angle ), y + R * sin( angle ), color );
        ++currVert;

        angle += piOverStarPoints;
        
        EmitVertex( &verts[currVert], x + r * cos( angle ), y + r * sin( angle ), color );
        ++currVert;

        // Second triangle
        EmitVertex( &verts[currVert], x, y, white );
        ++currVert;

        EmitVertex( &verts[currVert], x + r * cos( angle ), y + r * sin( angle ), color );
        ++currVert;

        angle += piOverStarPoints;

        EmitVertex( &verts[currVert], x + R * cos( angle ), y + R * sin( angle ), color );
        ++currVert;
    }

    D3D10_BUFFER_DESC vbDesc;
    vbDesc.ByteWidth      = 30 * sizeof( Vertex2 );
    vbDesc.Usage          = D3D10_USAGE_DEFAULT;
    vbDesc.BindFlags      = D3D10_BIND_VERTEX_BUFFER;
    vbDesc.CPUAccessFlags = 0;
    vbDesc.MiscFlags      = 0;

    D3D10_SUBRESOURCE_DATA vbInitData;
    ZeroMemory( &vbInitData, sizeof( D3D10_SUBRESOURCE_DATA ) );

    vbInitData.pSysMem = verts;

    HRESULT hr = g_pDevice->CreateBuffer( &vbDesc, &vbInitData, &vBuff );
    if( hr != S_OK )
        return NULL;

    return vBuff;
}

HRESULT CreateAllStars()
{
    HRESULT hr = S_OK;

    float red[]    = { 1, 0, 0 };
    float green[]  = { 0, 1, 0 };
    float blue[]   = { 0, 0, 1 };
    float cyan[]   = { 0, 1, 1 };
    float yellow[] = { 1, 1, 0 };
    float gray[]   = { 0.5f, 0.5f, 0.5f };

    /*                                       outer    inner          */
    /*                       x        y      radius   radius  color  */
    /*                       =====    =====  ======   ======  ====== */
    redStar = createStar(    -0.1f,   0.0f,  0.5f,    0.2f,   red    );
    if( redStar == NULL )
        return E_FAIL;

    greenStar = createStar(  -0.84f,  0.1f,  0.3f,    0.12f,  green  );
    if( greenStar == NULL )
        return E_FAIL;

    blueStar = createStar(    0.92f, -0.5f,  0.25f,   0.11f,  blue   );
    if( blueStar == NULL )
        return E_FAIL;

    cyanStar = createStar(    0.3f,   0.97f, 0.3f,    0.1f,   cyan   );
    if( cyanStar == NULL )
        return E_FAIL;

    yellowStar = createStar(  0.94f,  0.3f,  0.5f,    0.2f,   yellow );
    if( yellowStar == NULL )
        return E_FAIL;

    grayStar = createStar(   -0.97f, -0.8f,  0.6f,    0.2f,   gray   );
    if( grayStar == NULL )
        return E_FAIL;
    

    return S_OK;
}

void CleanupAllStars()
{
    SAFE_RELEASE( redStar    );
    SAFE_RELEASE( greenStar  );
    SAFE_RELEASE( blueStar   );
    SAFE_RELEASE( cyanStar   );
    SAFE_RELEASE( yellowStar );
    SAFE_RELEASE( grayStar   );
}

//--------------------------------------------------------------------------------------
// Render a frame
//--------------------------------------------------------------------------------------
void drawStars()
{
    UINT strides[] = { sizeof( Vertex2 ), sizeof( Vertex2 ), sizeof( Vertex2 ),
                       sizeof( Vertex2 ), sizeof( Vertex2 ), sizeof( Vertex2 ) };
    UINT offsets[] = { 0, 0, 0, 0, 0, 0 };
    ID3D10Buffer * pBuffers[] = { redStar, greenStar, blueStar, cyanStar, yellowStar, grayStar };

    g_pDevice->IASetInputLayout( g_pVertexLayout );
    g_pDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

    for( int i = 0; i < 6; ++i )
    {
        g_pDevice->IASetVertexBuffers( 0, 1, &pBuffers[i], &strides[i], &offsets[i] );
        g_pDevice->Draw( 30, 0 ); // 30 verts in a star
    }
}

void Render()
{   
    // Clear the back buffer        
    g_pDevice->ClearRenderTargetView( g_pRenderTargetView, ClearColor );

    g_pDevice->OMSetBlendState( g_pBlendState_NoBlend, 0, 0xffffffff );
    g_pDevice->RSSetState( g_pRasterizerState_NoCull );  
    
    cgD3D10BindProgram( myCgVertexProgram   );
    cgD3D10BindProgram( myCgGeometryProgram );
    cgD3D10BindProgram( myCgFragmentProgram );

    drawStars();   

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
