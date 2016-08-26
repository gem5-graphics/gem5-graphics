
/* cgfx_latest.c - a Cg 3.0 demo demonstrating the "latest" profile
   string and cgSetStateLatestProfile usage.  Command line options are
   used to programmatically over-ride the "latest" profile behavior. */

#include <windows.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <math.h>
using namespace std;

#include <Cg/cg.h>          /* Cg Core API: Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgD3D11.h>     /* Cg Direct3D11 API (part of Cg Toolkit) */

#include <d3dx11.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dx11.lib")

#ifndef SAFE_RELEASE
#define SAFE_RELEASE( p ) { if( p ) { ( p )->Release(); ( p ) = NULL; } }
#endif

//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct MY_V2F
{
    float x, y;
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
ID3D11Texture2D *           g_pDepthStencil     = NULL;

ID3D11InputLayout *         g_pVertexLayout = NULL;
ID3D11Buffer *              g_pVB           = NULL;

ID3D11Resource * myBrickNormalMap        = NULL;

const int Width  = 640;
const int Height = 480;

float ClearColor[4] = { 0.0f, 0.0f, 1.0f, 1.0f }; // RGBA

/* Cg global variables */
CGcontext    myCgContext;
CGeffect     myCgEffect;
CGtechnique  myCgTechnique;
CGparameter myCgEyePositionParam,
            myCgLightPositionParam,
            myCgModelViewProjParam;

CGprofile latestVertexProfile = (CGprofile)0;
CGprofile latestFragmentProfile = (CGprofile)0;

const TCHAR * myProgramName  = L"latest";
const char *  myCgFXFileName = "latest.cgfx"; // Use char here since that's what the Cg runtime expects
const TCHAR * myTextureName  = L"bumpmap.png";

/* Initial scene state */
int myAnimating = 0;
float myEyeAngle = 0;
const float myLightPosition[3] = { -8, 0, 15 };
float myProjectionMatrix[16];

const int myTorusSides = 20;
const int myTorusRings = 40;

static const double myPi = 3.14159265358979323846;

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

HRESULT             CreateScene();

void                ProcessCommandLineForLatestProfiles( LPSTR lpCmdLine );
void                RegisterLatestProfiles();

HRESULT             InitTextures();
HRESULT             InitTorusVertexBuffer( int sides, int rings );
void                BuildPerspectiveMatrix( double fieldOfView, double aspectRatio, double zNear, double zFar, float m[16] );
void                DrawFlatPatch( float rows, float columns );
void                BuildLookAtMatrix( double eyex, double eyey, double eyez,
                                       double centerx, double centery, double centerz,
                                       double upx, double upy, double upz,
                                       float m[16] );
void                MultiplyMatrix( float dst[16], const float src1[16], const float src2[16] );
HRESULT             DrawFlatPatch( int sides, int rings );

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

    ProcessCommandLineForLatestProfiles( lpCmdLine );

    if( FAILED( InitCg() ) || FAILED( CreateScene() ) )
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

        if( myAnimating )
            myEyeAngle += 0.001f;
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

    // Create depth stencil texture
    D3D11_TEXTURE2D_DESC descDepth;
    descDepth.Width              = width;
    descDepth.Height             = height;
    descDepth.MipLevels          = 1;
    descDepth.ArraySize          = 1;
    descDepth.Format             = DXGI_FORMAT_D32_FLOAT;
    descDepth.SampleDesc.Count   = 1;
    descDepth.SampleDesc.Quality = 0;
    descDepth.Usage              = D3D11_USAGE_DEFAULT;
    descDepth.BindFlags          = D3D11_BIND_DEPTH_STENCIL;
    descDepth.CPUAccessFlags     = 0;
    descDepth.MiscFlags          = 0;

    hr = g_pDevice->CreateTexture2D( &descDepth, NULL, &g_pDepthStencil );
    if( FAILED( hr ) )
        return hr;

    // Create the depth stencil view
    D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
    descDSV.Format             = descDepth.Format;
    descDSV.ViewDimension      = D3D11_DSV_DIMENSION_TEXTURE2D;
    descDSV.Texture2D.MipSlice = 0;
    descDSV.Flags              = 0;

    hr = g_pDevice->CreateDepthStencilView( g_pDepthStencil, &descDSV, &g_pDepthStencilView );
    if( FAILED( hr ) )
        return hr;

    g_pDeviceContext->OMSetRenderTargets( 1, &g_pRenderTargetView, NULL );

    // Setup the viewport
    D3D11_VIEWPORT vp;
    vp.Width    = width;
    vp.Height   = height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pDeviceContext->RSSetViewports( 1, &vp );

    double aspectRatio = (float) width / (float) height;
    double fieldOfView = 70.0; /* Degrees */

    /* Build projection matrix once. */
    BuildPerspectiveMatrix( fieldOfView, aspectRatio,
                            1.0, 20.0,  /* Znear and Zfar */
                            myProjectionMatrix );

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

void ProcessCommandLineForLatestProfiles( LPSTR lpCmdLine )
{
    int    argc;
    char** argv;
    char*  arg;
    int    index;
    int    result;

    argc = 1;
    arg  = lpCmdLine;
    
    while( arg[0] != 0 )
    {
        while( arg[0] != 0 && arg[0] == ' ' )
            ++arg;
        
        if( arg[0] != 0 )
        {        
            ++argc;        
            while( arg[0] != 0 && arg[0] != ' ' )
                ++arg;
        }    
    }    
    
    argv = (char**)malloc(argc * sizeof(char*));

    arg = lpCmdLine;
    index = 1;

    while( arg[0] != 0 )
    {
        while( arg[0] != 0 && arg[0] == ' ' )
            ++arg;
        
        if( arg[0] != 0 )
        {        
            argv[index] = arg;
            index++;
        
            while( arg[0] != 0 && arg[0] != ' ' )
                ++arg;
                    
            if( arg[0] != 0 )
            {
                arg[0] = 0;    
                arg++;
            }        
        }    
    }    

    for( int i = 1; i < argc; ++i )
    {
        if( !strcmp( "-sm4", argv[i] ) )
        {
            latestVertexProfile = CG_PROFILE_VS_4_0;
            latestFragmentProfile = CG_PROFILE_PS_4_0;
        } 
        else if( !strcmp( "-sm5", argv[i] ) )
        {
            latestVertexProfile = CG_PROFILE_VS_5_0;
            latestFragmentProfile = CG_PROFILE_PS_5_0;
        }
        else
        {
            fprintf(stderr, "%s: Unknown option %s.\n", myProgramName, argv[i]);
            fprintf(stderr, "Valid options:\n"
                            "  -nv30 :: GeForce 5 series functionality\n"
                            "  -nv40 :: GeForce 6 & 7 series functionality\n"
                            "  -gp4  :: GeForce 8 & 9 series functionality\n"
                            "  -arb  :: Multi-vendor functionality\n");
            exit(1);
        }
    }
}

//--------------------------------------------------------------------------------------
// Create Cg objects
//--------------------------------------------------------------------------------------
void RegisterLatestProfiles()
{
    CGstate vpState, vsState, fpState, psState;

    /* To be comprehensive, change both the OpenGL-style "VertexProgram"
    and Direct3D-style "VertexShader" state names. */
    vpState = cgGetNamedState( myCgContext, "VertexProgram" );
    vsState = cgGetNamedState( myCgContext, "VertexShader" );

    assert(CG_PROGRAM_TYPE == cgGetStateType(vpState));
    assert(CG_PROGRAM_TYPE == cgGetStateType(vsState));

    if( latestVertexProfile )
    {
        cgSetStateLatestProfile( vpState, latestVertexProfile );
        cgSetStateLatestProfile( vsState, latestVertexProfile );
    }

    /* To be comprehensive, change both the OpenGL-style "FragmentProgram"
    and Direct3D-style "FragmentShader" state names. */
    fpState = cgGetNamedState(myCgContext, "FragmentProgram");
    psState = cgGetNamedState(myCgContext, "PixelShader");

    assert(CG_PROGRAM_TYPE == cgGetStateType(fpState));
    assert(CG_PROGRAM_TYPE == cgGetStateType(psState));

    if( latestFragmentProfile )
    {
        cgSetStateLatestProfile( fpState, latestFragmentProfile );
        cgSetStateLatestProfile( psState, latestFragmentProfile );
    }
}

HRESULT InitCg()
{
    HRESULT hr = S_OK;

    myCgContext = cgCreateContext();
    checkForCgError( "creating context" );

    hr = cgD3D11SetDevice( myCgContext, g_pDevice );
    checkForCgError( "setting Direct3D device", false );
    if( hr != S_OK )
        return hr;

    cgD3D11RegisterStates( myCgContext );
    checkForCgError( "registering standard CgFX states" );
    
    cgD3D11SetManageTextureParameters( myCgContext, CG_TRUE );
    checkForCgError( "manage texture parameters" );

    RegisterLatestProfiles();

    myCgEffect = cgCreateEffectFromFile( myCgContext, myCgFXFileName, NULL );
    checkForCgError( "creating latest.cgfx effect" );
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
  
    myCgModelViewProjParam = cgGetEffectParameterBySemantic(myCgEffect, "ModelViewProjection");
    if (!myCgModelViewProjParam)
    {
        fprintf(stderr, "%s: must find parameter with ModelViewProjection semantic\n", myProgramName);
        exit(1);
    }

    myCgEyePositionParam = cgGetNamedEffectParameter(myCgEffect, "EyePosition");
    if (!myCgEyePositionParam)
    {
        fprintf(stderr, "%s: must find parameter named EyePosition\n", myProgramName);
        exit(1);
    }

    myCgLightPositionParam = cgGetNamedEffectParameter(myCgEffect, "LightPosition");
    if (!myCgLightPositionParam)
    {
        fprintf(stderr, "%s: must find parameter named LightPosition\n", myProgramName);
        exit(1);
    }

    const D3D11_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    CGpass myPass = cgGetFirstPass( myCgTechnique );

    ID3D10Blob * pVSBuf = cgD3D11GetIASignatureByPass( myPass );

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
    if( g_pDeviceContext )      g_pDeviceContext->ClearState();

    if( g_pRenderTargetView )   g_pRenderTargetView->Release();
    if( g_pDepthStencilView )   g_pDepthStencilView->Release();
    if( g_pDepthStencil )       g_pDepthStencil->Release();
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
  
    cgD3D11SetDevice( myCgContext, NULL );

    cgDestroyContext( myCgContext );  
}

HRESULT CreateScene()
{
    if( FAILED( InitTextures() ) )
        return E_FAIL;

    if( FAILED( InitTorusVertexBuffer( myTorusSides, myTorusRings ) ) )
        return E_FAIL;

    double fieldOfView = 60.0;  // In degrees
    double width       = Width;
    double height      = Height;
    double aspectRatio = width / height;
    double zNear       = 0.1;
    double zFar        = 100.0;

    BuildPerspectiveMatrix( fieldOfView, aspectRatio, zNear, zFar, myProjectionMatrix );

    return S_OK;
}

void UseSamplerParameter( CGeffect effect, const char *paramName, ID3D11Resource *tex )
{
    CGparameter param = cgGetNamedEffectParameter( effect, paramName );

    if (!param)
    {
        fprintf(stderr, "%s: expected effect parameter named %s\n", myProgramName, paramName);
        exit(1);
    }

    cgD3D11SetTextureParameter( param, tex );
    cgSetSamplerState( param );
}

HRESULT InitTextures()
{
    HRESULT hr = S_OK;

    D3DX11_IMAGE_INFO fileInfo;
    D3DX11GetImageInfoFromFile( myTextureName, NULL, &fileInfo, NULL );

    D3DX11_IMAGE_LOAD_INFO loadInfo;
    loadInfo.Width          = fileInfo.Width;
    loadInfo.Height         = fileInfo.Height;
    loadInfo.FirstMipLevel  = 0;
    loadInfo.MipLevels      = fileInfo.MipLevels;
    loadInfo.Usage          = D3D11_USAGE_DEFAULT;
    loadInfo.BindFlags      = D3D11_BIND_SHADER_RESOURCE;
    loadInfo.CpuAccessFlags = 0;
    loadInfo.MiscFlags      = 0;
    loadInfo.Format         = fileInfo.Format;    
    loadInfo.Filter         = D3DX11_FILTER_NONE;
    loadInfo.MipFilter      = D3DX11_FILTER_NONE;
    loadInfo.pSrcInfo       = &fileInfo;
    
    hr = D3DX11CreateTextureFromFile( g_pDevice, myTextureName, &loadInfo, NULL, &myBrickNormalMap, NULL );
    if( hr != S_OK )
        return hr;  

    UseSamplerParameter( myCgEffect, "normalMap", myBrickNormalMap );

    return S_OK;
}

HRESULT InitTorusVertexBuffer( int sides, int rings )
{
    const float m = 1.0f / float(rings);
    const float n = 1.0f / float(sides);

    const int numVertsPerStrip = 2 * sides + 2;
    const int numVertsPerPatch = numVertsPerStrip * rings;

    D3D11_BUFFER_DESC vbDesc;
    vbDesc.ByteWidth      = numVertsPerPatch * sizeof( MY_V2F );
    vbDesc.Usage          = D3D11_USAGE_DEFAULT;
    vbDesc.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    vbDesc.CPUAccessFlags = 0;
    vbDesc.MiscFlags      = 0;

    D3D11_SUBRESOURCE_DATA vbInitData;
    ZeroMemory( &vbInitData, sizeof( D3D11_SUBRESOURCE_DATA ) );

    MY_V2F * pVertices = new MY_V2F[numVertsPerPatch];

    int index = 0;
    for( int i = 0; i < rings; ++i )
    {
        for( int j = 0; j <= sides; ++j )
        {
            pVertices[index].x = (float)i * m;
            pVertices[index].y = (float)j * n;
            index++;
            pVertices[index].x = ((float)i + 1.0f) * m;
            pVertices[index].y = (float)j * n;
            index++;
        }        
    }

    vbInitData.pSysMem = pVertices;

    HRESULT hr = g_pDevice->CreateBuffer( &vbDesc, &vbInitData, &g_pVB );
    if( hr != S_OK )
    {
        delete [] pVertices;
        return hr;
    }

    delete [] pVertices;
    return S_OK;
}

void BuildPerspectiveMatrix( double fieldOfView, double aspectRatio, double zNear, double zFar, float m[16] )
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

void BuildLookAtMatrix( double eyex, double eyey, double eyez,
                        double centerx, double centery, double centerz,
                        double upx, double upy, double upz,
                        float m[16]
                      )
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

void MultiplyMatrix( float dst[16], const float src1[16], const float src2[16] )
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

HRESULT DrawFlatPatch( int sides, int rings )
{
    HRESULT hr = S_OK;

    UINT strides[1] = { sizeof( MY_V2F ) };
    UINT offsets[1] = { 0 };
    ID3D11Buffer * pBuffers[1] = { g_pVB };

    g_pDeviceContext->IASetVertexBuffers( 0, 1, pBuffers, strides, offsets );
    g_pDeviceContext->IASetInputLayout( g_pVertexLayout );
    g_pDeviceContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );

    for( int i = 0, vertStart = 0; i < rings; i++, vertStart += (2 * sides + 2))
    {        
        g_pDeviceContext->Draw( sides * 2 + 2, vertStart );
    }

    return hr;
}

//--------------------------------------------------------------------------------------
// Render a frame
//--------------------------------------------------------------------------------------
void Render()
{   
    float eyePosition[3];
    float modelViewMatrix[16];
    float modelViewProjMatrix[16];

    const float eyeRadius = 18.0f;
    const float eyeElevationRange = 8.0f;
  
    CGpass pass;

    // Clear the back buffer and depth buffer
    g_pDeviceContext->ClearRenderTargetView( g_pRenderTargetView, ClearColor );
    g_pDeviceContext->ClearDepthStencilView( g_pDepthStencilView, D3D10_CLEAR_DEPTH, 1.0f, 0 );
    
    eyePosition[0] = eyeRadius         * sin( myEyeAngle );
    eyePosition[1] = eyeElevationRange * sin( myEyeAngle );
    eyePosition[2] = eyeRadius         * cos( myEyeAngle );

    BuildLookAtMatrix( eyePosition[0], eyePosition[1], eyePosition[2], 
                       0.0 ,0.0,  0.0,   /* XYZ view center */
                       0.0, 1.0,  0.0,   /* Up is in positive Y direction */
                       modelViewMatrix
                     );

    // modelViewProj = projectionMatrix * modelViewMatrix
    MultiplyMatrix( modelViewProjMatrix, myProjectionMatrix, modelViewMatrix );

    // Row major version 
    cgSetMatrixParameterfr( myCgModelViewProjParam, modelViewProjMatrix );

    cgSetParameter3fv( myCgEyePositionParam, eyePosition );
    cgSetParameter3fv( myCgLightPositionParam, myLightPosition );

    // Iterate through rendering passes for technique (even though bumpdemo.cgfx has just one pass).    
    for( pass = cgGetFirstPass( myCgTechnique ); pass; pass = cgGetNextPass( pass ) )
    {
        cgSetPassState(pass);
    
            DrawFlatPatch( myTorusSides, myTorusRings );
    
        cgResetPassState(pass);
    }
 
    g_pSwapChain->Present( 0, 0 );
}

//--------------------------------------------------------------------------------------
// Called every time the application receives a message
//--------------------------------------------------------------------------------------
LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam )
{
    static int wireframe = 0;

    switch( message ) 
    {
        case WM_KEYDOWN:
        {
            switch( wParam )
            {
                case VK_ESCAPE:
                    PostQuitMessage( 0 );
                    break;
                
                case VK_SPACE:
                    myAnimating = !myAnimating; // Toggle
                    break;
                
                case 'W':
                {
                    wireframe = !wireframe;
                    if( wireframe )
                    {
                        // Set D3D11 state for wireframe
                    }
                    else
                    {
                        // Set D3D11 state for solid fill
                    }
                } break;
            }
        } break;

        case WM_DESTROY:
            PostQuitMessage( 0 );
            break;

        default:
            return DefWindowProc( hWnd, message, wParam, lParam );
    }

    return 0;
}
