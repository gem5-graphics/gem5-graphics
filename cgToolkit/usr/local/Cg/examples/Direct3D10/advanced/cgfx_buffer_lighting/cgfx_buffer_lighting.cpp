/* cgfx_buffer_lighting.cpp  */

#include <windows.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
using namespace std;
#include <d3d10_1.h>     /* Direct3D10 API: Can't include this?  Is DirectX SDK installed? */
#include <d3d10.h>       /* Direct3D10 API: Can't include this?  Is DirectX SDK installed? */
#include <Cg/cg.h>       /* Can't include this?  Is Cg Toolkit installed? */
#include <Cg/cgD3D10.h>  /* Can't include this?  Is Cg Toolkit installed? */

#include "materials.h"
#include "matrix.h"

#pragma comment(lib, "d3d10.lib")

#ifndef SAFE_RELEASE
#define SAFE_RELEASE( p ) { if( p ) { ( p )->Release(); ( p ) = NULL; } }
#endif

//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct vec3_t
{
    float x, y, z;
};

struct int3_t
{
    unsigned short x, y, z;
};

struct mesh_t
{
    int numVertices;
    int numIndices;
    ID3D10Buffer * vertBuffer;
    ID3D10Buffer * indBuffer;

    DXGI_FORMAT format;
    ID3D10InputLayout * vertLayout;

    D3D10_PRIMITIVE_TOPOLOGY topology;
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
ID3D10Texture2D *           g_pDepthStencil     = NULL;

const int Width  = 400;
const int Height = 400;

float ClearColor[4] = { 0.2f, 0.2f, 0.2f, 1.0f }; // RGBA

bool myAnimating = false;
int currentLight = 0;

/* Cg global variables */
CGcontext    myCgContext;
CGeffect     myCgEffect;
CGtechnique  myCgTechnique;
CGbuffer     transform_buffer, * material_buffer, lightSet_buffer, lightSetPerView_buffer;

const char * myProgramName  = "cgfx_buffer_lighting";
const char * myCgFXFileName = "buffer_lighting.cgfx";

mesh_t * bigSphere   = NULL;

float eyeAngle = 1.6f;
float myProjectionMatrix[16];
int object_material[2] = { 0, 3 };

int material_buffer_index;
int transform_buffer_offset;
int lightSetPerView_offset;

typedef float float4x4[16];
typedef struct 
{
  float4x4 modelview;
  float4x4 inverse_modelview;
  float4x4 modelview_projection;
} Transform;

#define MAX_LIGHTS 8

// __unused filler variables used to compensate for D3D10's buffer packing rules
typedef struct 
{
  float enabled;
  float ambient[3];

  float diffuse[3];
  float __unused0;

  float specular[3];
  float k0;
  
  float k1;
  float k2;
  float __unused1[2];
} LightSourceStatic;

typedef struct 
{
  float global_ambient[3];
  float __unused;
  LightSourceStatic source[MAX_LIGHTS];
} LightSet;

typedef struct 
{
  float position[4];
} LightSourcePerView;

typedef struct 
{
  LightSourcePerView source[MAX_LIGHTS];
} LightSetPerView;

LightSet lightSet;
LightSetPerView lightSetPerView_world, lightSetPerView_eye;

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
void                InitBuffers();

HRESULT             CreateScene();
mesh_t *            CreateSphere( float radius, int slices, int stacks );
void                FreeMesh( mesh_t ** sphere );
void                DrawMesh( mesh_t * sphere );
void                InitLight( LightSet * lightSet, int index );

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
            eyeAngle += 0.001f;
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
    wcex.lpszClassName  = L"cgfx_buffer_lighting";
    wcex.hIconSm        = NULL;

    if( !RegisterClassEx( &wcex ) )
        return E_FAIL;

    // Create window    
    AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
    g_hWnd = CreateWindow( L"cgfx_buffer_lighting",// Class name
                           L"cgfx_buffer_lighting",// Window name
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

    // Create depth stencil texture
    D3D10_TEXTURE2D_DESC descDepth;
    descDepth.Width              = width;
    descDepth.Height             = height;
    descDepth.MipLevels          = 1;
    descDepth.ArraySize          = 1;
    descDepth.Format             = DXGI_FORMAT_D32_FLOAT;
    descDepth.SampleDesc.Count   = 1;
    descDepth.SampleDesc.Quality = 0;
    descDepth.Usage              = D3D10_USAGE_DEFAULT;
    descDepth.BindFlags          = D3D10_BIND_DEPTH_STENCIL;
    descDepth.CPUAccessFlags     = 0;
    descDepth.MiscFlags          = 0;

    hr = g_pDevice->CreateTexture2D( &descDepth, NULL, &g_pDepthStencil );
    if( FAILED( hr ) )
        return hr;

    // Create the depth stencil view
    D3D10_DEPTH_STENCIL_VIEW_DESC descDSV;
    descDSV.Format             = descDepth.Format;
    descDSV.ViewDimension      = D3D10_DSV_DIMENSION_TEXTURE2D;
    descDSV.Texture2D.MipSlice = 0;

    hr = g_pDevice->CreateDepthStencilView( g_pDepthStencil, &descDSV, &g_pDepthStencilView );
    if( FAILED( hr ) )
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

    double aspectRatio = (float) width / (float) height;
    double fieldOfView = 70.0; /* Degrees */

    /* Build projection matrix once. */
    makePerspectiveMatrix( fieldOfView, aspectRatio,
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
  
    InitBuffers();

    CGprogram myCgVertexProgram = cgGetPassProgram( cgGetFirstPass( myCgTechnique ), CG_VERTEX_DOMAIN );

    material_buffer_index   = cgGetParameterBufferIndex(  cgGetNamedParameter( myCgVertexProgram, "cbuffer0_Material"        ) );
    transform_buffer_offset = cgGetParameterBufferOffset( cgGetNamedParameter( myCgVertexProgram, "cbuffer1_Transform"       ) );
    lightSetPerView_offset  = cgGetParameterBufferOffset( cgGetNamedParameter( myCgVertexProgram, "cbuffer3_LightSetPerView" ) );
      
    return S_OK;
}

//--------------------------------------------------------------------------------------
// Clean up the objects we've created
//--------------------------------------------------------------------------------------
void CleanupDevice()
{
    FreeMesh( &bigSphere );
   
    if( g_pDevice )             g_pDevice->ClearState();

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
  
    cgD3D10SetDevice( myCgContext, NULL );

    cgDestroyContext( myCgContext );   
}

//--------------------------------------------------------------------------------------
// Create the CGbuffers
//--------------------------------------------------------------------------------------
void InitBuffers()
{
    memset( &lightSet, 0, sizeof( lightSet ) );
    lightSet.global_ambient[0] = 0.15f;
    lightSet.global_ambient[1] = 0.15f;
    lightSet.global_ambient[2] = 0.15f;
   
    InitLight( &lightSet, 0 );
    InitLight( &lightSet, 1 );

    lightSetPerView_world.source[0].position[0] = 0.0f;
    lightSetPerView_world.source[0].position[1] = 2.0f;
    lightSetPerView_world.source[0].position[2] = 4.0f;
    lightSetPerView_world.source[0].position[3] = 1.0f;

    lightSetPerView_world.source[1].position[0] = 0.0f;
    lightSetPerView_world.source[1].position[1] = -2.0f;
    lightSetPerView_world.source[1].position[2] = 4.0f;
    lightSetPerView_world.source[1].position[3] = 1.0f;

    CGprogram myCgVertexProgram   = cgGetPassProgram( cgGetFirstPass( myCgTechnique ), CG_VERTEX_DOMAIN );
    CGprogram myCgFragmentProgram = cgGetPassProgram( cgGetFirstPass( myCgTechnique ), CG_FRAGMENT_DOMAIN );

    CGparameter bufferParam = (CGparameter)0;
    int bufferParamIndex    = -1;

    transform_buffer = cgCreateBuffer( myCgContext, 3 * 16 * sizeof( float ), NULL, CG_BUFFER_USAGE_DYNAMIC_DRAW );
    bufferParam      = cgGetNamedParameter( myCgVertexProgram, "cbuffer1_Transform" );
    bufferParamIndex = cgGetParameterBufferIndex( bufferParam );
      
    cgSetProgramBuffer( myCgVertexProgram, bufferParamIndex, transform_buffer );

    lightSet_buffer  = cgCreateBuffer( myCgContext, sizeof( lightSet ), &lightSet, CG_BUFFER_USAGE_STATIC_DRAW );  
    bufferParam      = cgGetNamedParameter( myCgFragmentProgram, "cbuffer2_LightSetStatic" );
    bufferParamIndex = cgGetParameterBufferIndex( bufferParam );
  
    cgSetProgramBuffer( myCgFragmentProgram, bufferParamIndex, lightSet_buffer );

    lightSetPerView_buffer = cgCreateBuffer( myCgContext, sizeof( LightSetPerView ), NULL, CG_BUFFER_USAGE_DYNAMIC_DRAW );
    bufferParam            = cgGetNamedParameter( myCgFragmentProgram, "cbuffer3_LightSetPerView" );
    bufferParamIndex       = cgGetParameterBufferIndex( bufferParam );

    cgSetProgramBuffer( myCgFragmentProgram, bufferParamIndex, lightSetPerView_buffer );

    // Create a set of material buffers.
    material_buffer = (CGbuffer*)malloc( sizeof( CGbuffer ) * materialInfoCount );    
    for( int i = 0; i < materialInfoCount; ++i )
        material_buffer[i] = cgCreateBuffer( myCgContext, sizeof( MaterialData ), &materialInfo[i].data, CG_BUFFER_USAGE_STATIC_DRAW );  

    checkForCgError( "InitBuffers" );
}

//--------------------------------------------------------------------------------------
// Initialize the light's properties
//--------------------------------------------------------------------------------------
void InitLight( LightSet * lightSet, int index )
{
    if( lightSet == NULL )
        return;

    lightSet->source[index].enabled     = 1.0f;
    lightSet->source[index].ambient[0]  = 0.0f;
    lightSet->source[index].ambient[1]  = 0.0f;
    lightSet->source[index].ambient[2]  = 0.0f;
    lightSet->source[index].diffuse[0]  = 0.9f;
    lightSet->source[index].diffuse[1]  = 0.9f;
    lightSet->source[index].diffuse[2]  = 0.9f;
    lightSet->source[index].specular[0] = 0.9f;
    lightSet->source[index].specular[1] = 0.9f;
    lightSet->source[index].specular[2] = 0.9f;
    
    // Inverse square law attenuation    
    lightSet->source[index].k0 = 0.7f;
    lightSet->source[index].k1 = 0.0f;
    lightSet->source[index].k2 = 0.001f;
}

//--------------------------------------------------------------------------------------
// Create the scene's spheres
//--------------------------------------------------------------------------------------
HRESULT CreateScene()
{
    bigSphere = CreateSphere( 2.0f, 20, 20 );
    if( bigSphere == NULL )
        return E_FAIL;

    return S_OK;;
}


mesh_t * CreateSphere( float radius, int slices, int stacks )
{
    HRESULT hr = S_OK;
    D3D10_BUFFER_DESC buffDesc;

    const float PI = 3.1415926f;

    vector< vec3_t > vertexList;
    vector< unsigned int > indexList;

    mesh_t * sphere = new mesh_t;

    sphere->format   = DXGI_FORMAT_R32_UINT;
    sphere->topology = D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

    float phiStep = PI / stacks;
    int rings = stacks - 1;

    for( int i = 1; i <= rings; ++i )
    {
        float phi = i * phiStep;

        float thetaStep = 2.0f * PI / slices;
        for( int j = 0; j <= slices; ++j )
        {
            vec3_t vert;
            float theta = j * thetaStep;

            vert.x = radius * sinf( phi ) * cosf( theta );
            vert.y = radius * cosf( phi );
            vert.z = radius * sinf( phi ) * sinf( theta );

            vertexList.push_back( vert );
        }
    }

    sphere->numVertices = vertexList.size();

    // Indices for the top
    unsigned int nPole = 0;    
    for( int i = 0; i < slices; ++i )
    {
        indexList.push_back( nPole );
        indexList.push_back( i + 1 );
        indexList.push_back( i );
    }

    // Indices for middle stacks
    int ringVerts = slices + 1;
    for( int i = 0; i < stacks - 2; ++i )
    {
        // Create two triangles to make a quad
        for( int j = 0; j < slices; ++j )
        {
            indexList.push_back( i * ringVerts + j );
            indexList.push_back( i * ringVerts + j + 1 );
            indexList.push_back( ( i + 1 ) * ringVerts + j );

            indexList.push_back( ( i + 1 ) * ringVerts + j );
            indexList.push_back( i * ringVerts + j + 1 );
            indexList.push_back( ( i + 1 ) * ringVerts + j + 1 );
        }
    }

    // Indices for the bottom    
    unsigned int sPole = vertexList.size() - 2;   
    int baseIndex = ( rings - 1 ) * ringVerts;
    for( int i = 0; i < slices; ++i )
    {
        indexList.push_back( sPole );
        indexList.push_back( baseIndex + i );
        indexList.push_back( baseIndex + i + 1 );
    }
    
    sphere->numIndices  = indexList.size();

    // Setup Vertex Buffer
    buffDesc.ByteWidth      = sphere->numVertices * sizeof( vec3_t );
    buffDesc.Usage          = D3D10_USAGE_DEFAULT;
    buffDesc.BindFlags      = D3D10_BIND_VERTEX_BUFFER;
    buffDesc.CPUAccessFlags = 0;
    buffDesc.MiscFlags      = 0;

    D3D10_SUBRESOURCE_DATA buffInitData;
    ZeroMemory( &buffInitData, sizeof( D3D10_SUBRESOURCE_DATA ) );

    buffInitData.pSysMem = &vertexList[0];
    hr = g_pDevice->CreateBuffer( &buffDesc, &buffInitData, &sphere->vertBuffer );
    if( hr != S_OK )
    {
        delete sphere;
        return NULL;
    }

    // Setup Index Buffer
    buffDesc.BindFlags = D3D10_BIND_INDEX_BUFFER;
    buffDesc.ByteWidth = sizeof( unsigned int ) * sphere->numIndices;

    buffInitData.pSysMem = &indexList[0];
    hr = g_pDevice->CreateBuffer( &buffDesc, &buffInitData, &sphere->indBuffer );
    if( hr != S_OK )
    {
        delete sphere;
        return NULL;
    }

    // Create vertex layout
    const D3D10_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D10_INPUT_PER_VERTEX_DATA, 0 },        
    };
    UINT numElements = sizeof( layout ) / sizeof( layout[0] );

    CGpass myPass = cgGetFirstPass( myCgTechnique );

    ID3D10Blob * pVSBuf = cgD3D10GetIASignatureByPass( myPass );

    hr = g_pDevice->CreateInputLayout( layout, numElements, pVSBuf->GetBufferPointer(), pVSBuf->GetBufferSize(), &sphere->vertLayout );     
    if( hr != S_OK )
    {
        delete sphere;
        return NULL;
    }

    return sphere;
}

//--------------------------------------------------------------------------------------
// Free a sphere
//--------------------------------------------------------------------------------------
void FreeMesh( mesh_t ** mesh )
{
    if( mesh == NULL )
        return;

    mesh_t * m = *mesh;
    if( m == NULL )
        return;

    m->numVertices = 0;
    m->numIndices  = 0;

    if( m->vertBuffer )
    {
        m->vertBuffer->Release();
        m->vertBuffer = NULL;        
    }

    if( m->indBuffer )
    {
        m->indBuffer->Release();
        m->indBuffer = NULL;
    }

    if( m->vertLayout )
    {
        m->vertLayout->Release();
        m->vertLayout = NULL;
    }

    delete m;
    m = NULL;
}

//--------------------------------------------------------------------------------------
// Draw a sphere
//--------------------------------------------------------------------------------------
void DrawMesh( mesh_t * mesh )
{
    if( mesh == NULL )
        return;
    if( mesh->numVertices == 0 || mesh->indBuffer == NULL || mesh->vertBuffer == NULL )
        return;

    UINT strides[1] = { sizeof( vec3_t ) };
    UINT offsets[1] = { 0 };
    ID3D10Buffer * pBuffers[1] = { mesh->vertBuffer };    

    g_pDevice->IASetVertexBuffers( 0, 1, pBuffers, strides, offsets );
    g_pDevice->IASetIndexBuffer( mesh->indBuffer, mesh->format, 0 );
    g_pDevice->IASetInputLayout( mesh->vertLayout );
    g_pDevice->IASetPrimitiveTopology( mesh->topology );

    g_pDevice->DrawIndexed( mesh->numIndices, 0, 0 );
}

//--------------------------------------------------------------------------------------
// Render a frame
//--------------------------------------------------------------------------------------
void BindMaterialBuffer( int object )
{
    CGprogram myCgFragmentProgram = cgGetPassProgram( cgGetFirstPass( myCgTechnique ), CG_FRAGMENT_DOMAIN );
    cgSetProgramBuffer( myCgFragmentProgram, material_buffer_index, material_buffer[object_material[object]] );
}

void UpdateTransformBuffer(Transform *transform)
{
    cgSetBufferSubData( transform_buffer, transform_buffer_offset, sizeof( Transform ), transform );
}

void DrawLitSphere( const float projectionMatrix[16], const float viewMatrix[16], int object, float xTranslate )
{
    Transform transform;
    float modelMatrix[16];

    makeTranslateMatrix( xTranslate, 0, 0, modelMatrix );
    multMatrix( transform.modelview, viewMatrix, modelMatrix );
    multMatrix( transform.modelview_projection, projectionMatrix, transform.modelview );
    invertMatrix( transform.inverse_modelview, transform.modelview );
  
    UpdateTransformBuffer( &transform );
    BindMaterialBuffer( object );

    CGpass pass = cgGetFirstPass( myCgTechnique );
    while( pass ) 
    {
        cgSetPassState( pass );
    
            DrawMesh( bigSphere );
                
        cgResetPassState( pass );
        pass = cgGetNextPass( pass );
    }
}

void Render()
{
    float eyePosition[4];
    float viewMatrix[16];

    // Clear the back buffer        
    g_pDevice->ClearRenderTargetView( g_pRenderTargetView, ClearColor );

    // Clear depth buffer
    g_pDevice->ClearDepthStencilView( g_pDepthStencilView, D3D10_CLEAR_DEPTH, 1.0f, 0 );
    
    // Update latest eye position.
    eyePosition[0] = 8.0f * cos( eyeAngle );
    eyePosition[1] = 0.0f;
    eyePosition[2] = -8.0f * sin( eyeAngle );
    eyePosition[3] = 1.0f;

    // Compute current view matrix.
    makeLookAtMatrix( eyePosition[0], eyePosition[1], eyePosition[2],   /* eye position */
                      0, 0, 0,                                          /* view center */
                      0, 1, 0,                                          /* up vector */
                      viewMatrix );

    // For each light, convert its world-space position to eye-space...
    for( int i = 0; i < 2; ++i ) 
    {
        // Le[i] = V * Lw[i]
        transformVector( lightSetPerView_eye.source[i].position, viewMatrix, lightSetPerView_world.source[i].position );
    }
  
    // Update light set per-view buffer
    cgSetBufferSubData( lightSetPerView_buffer, lightSetPerView_offset, sizeof( lightSetPerView_eye ), &lightSetPerView_eye );
    
    DrawLitSphere( myProjectionMatrix, viewMatrix, 0, 3.2f  );
    DrawLitSphere( myProjectionMatrix, viewMatrix, 1, -3.2f );   
    
 
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
                case VK_SPACE:
                    myAnimating = !myAnimating; // Toggle
                    break;
                
                case '1':
                    currentLight = 0;
                    break;
                case '2':
                    currentLight = 1;
                    break;
                case '3':
                    ++object_material[0];
                    if( object_material[0] >= materialInfoCount )
                        object_material[0] = 0;                                       
                    break;
                case '4':
                    ++object_material[1];
                    if( object_material[1] >= materialInfoCount )
                        object_material[1] = 0;
                    break;

                case VK_UP:
                    lightSetPerView_world.source[currentLight].position[1] += 0.2f;
                    break;
                case VK_DOWN:
                    lightSetPerView_world.source[currentLight].position[1] -= 0.2f;
                    break;
                case VK_SUBTRACT:
                    lightSetPerView_world.source[currentLight].position[2] += 0.2f;
                    break;
                case VK_ADD:
                    lightSetPerView_world.source[currentLight].position[2] -= 0.2f;
                    break;
                case VK_RIGHT:
                    lightSetPerView_world.source[currentLight].position[0] -= 0.2f;
                    break;
                case VK_LEFT:
                    lightSetPerView_world.source[currentLight].position[0] += 0.2f;
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
