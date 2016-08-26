#include "graphics/libOpenglRender/dll.h"
#include "graphics/graphic_calls.hh"
#include "base/misc.hh"
#include <pthread.h>
#include<ctime>

#include<X11/Xlib.h>
#include <SDL/SDL.h>
#include <SDL/SDL_syswm.h>
#include "graphics/serialize_graphics.hh"

#define RENDER_API_NO_PROTOTYPES 1
#include "graphics/libOpenglRender/render_api.h"

/* These definitions *must* match those under:
 * development/tools/emulator/opengl/host/include/libOpenglRender/render_api.h
 */
#define DYNLINK_FUNCTIONS  \
  DYNLINK_FUNC(initLibrary) \
  DYNLINK_FUNC(setStreamMode) \
  DYNLINK_FUNC(initOpenGLRenderer) \
  DYNLINK_FUNC(setPostCallback) \
  DYNLINK_FUNC(getHardwareStrings) \
  DYNLINK_FUNC(createOpenGLSubwindow) \
  DYNLINK_FUNC(destroyOpenGLSubwindow) \
  DYNLINK_FUNC(repaintOpenGLDisplay) \
  DYNLINK_FUNC(stopOpenGLRenderer) \
  DYNLINK_FUNC(gem5GetOpenGLContexts) \
  DYNLINK_FUNC(gem5CreateOpenGLContext)



#define RENDERER_LIB_NAME  "lib64OpenglRender"
static ADynamicLibrary*  rendererLib;
static int               rendererStarted =0;
static char              rendererAddress[256];



/* Define the function pointers */
#define DYNLINK_FUNC(name) \
    static name##Fn name = NULL;
DYNLINK_FUNCTIONS
#undef DYNLINK_FUNC



int initOpenglesEmulationFuncs(ADynamicLibrary* rendererLib)
{
    void*  symbol;
    char*  error;

#define DYNLINK_FUNC(name) \
    symbol = adynamicLibrary_findSymbol(rendererLib, #name, &error); \
    if (symbol != NULL) { \
        name = reinterpret_cast<name##Fn>(reinterpret_cast<long long>(symbol)); \
    } else { \
        inform("GLES emulation: Could not find required symbol (%s): %s", #name, error); \
        free(error); \
        return -1; \
    }
DYNLINK_FUNCTIONS
#undef DYNLINK_FUNC

    return 0;
}
 


int android_initOpenglesEmulation(void)
{
    char* error = NULL;

    if (rendererLib != NULL)
        return 0;

    DPRINTF(GraphicsCalls,"Initializing hardware OpenGLES emulation support\n");

    rendererLib = adynamicLibrary_open(RENDERER_LIB_NAME, &error);
    if (rendererLib == NULL) {
        fatal("Error: Could not load OpenGLES emulation library: %s", error);
    }

    /* Resolve the functions */
    if (initOpenglesEmulationFuncs(rendererLib) < 0) {
        inform("Error: OpenGLES emulation library mismatch. Be sure to use the correct version!");
        goto BAD_EXIT;
    }

    if (!initLibrary()) {
        inform("Error: OpenGLES initialization failed!");
        goto BAD_EXIT;
    }
    
    setStreamMode(STREAM_MODE_UNIX); //should not have a real effect as TCP mode will be used anyway
            
    return 0;

BAD_EXIT:
    inform("Error: OpenGLES emulation library could not be initialized!");
    adynamicLibrary_close(rendererLib);
    rendererLib = NULL;
    return -1;
}

int android_startOpenglesRenderer(int width, int height)
{
    
    if (!rendererLib) {
        inform("Can't start OpenGLES renderer without support libraries");
        return -1;
    }

    if (rendererStarted) {
        return 0;
    }

    if (!initOpenGLRenderer(width, height, rendererAddress, sizeof(rendererAddress))) {
        fatal("Can't start OpenGLES renderer?");
    } else {inform("rendererAddress=%s\n",rendererAddress);}
    
    setPostCallback(NULL,NULL); //disabling it

    rendererStarted = 1;
    return 0;
}


int android_showOpenglesWindow(FBNativeWindowType window, int x, int y, int width, int height, float rotation)
{
    if (rendererStarted) {
        int success = createOpenGLSubwindow(window, x, y, width, height, rotation);
        return success ? 0 : -1;
    } else {
        return -1;
    }
}

int android_hideOpenglesWindow(void)
{
    if (rendererStarted) {
        int success = destroyOpenGLSubwindow();
        return success ? 0 : -1;
    } else {
        return -1;
    }
}

void android_gles_server_path(char* buff, size_t buffsize)
{
    strncpy(buff, rendererAddress, buffsize);
}

SDL_Surface * surface;
void android_showWindow() {
    
#ifdef __linux__
    // some OpenGL implementations may call X functions
    // it is safer to synchronize all X calls made by all the
    // rendering threads. (although the calls we do are locked
    // in the FrameBuffer singleton object).
    XInitThreads();
#endif
    // initialize SDL window
    if (SDL_Init(SDL_INIT_NOPARACHUTE | SDL_INIT_VIDEO)) {
        fatal("SDL init failed: %s\n", SDL_GetError());
    }
    const SDL_VideoInfo* info = SDL_GetVideoInfo( );
    
    info = info; //to not complain about unused variable
    DPRINTF(GraphicsCalls,"android_showWindow: bpp is %d \n",info->vfmt->BitsPerPixel);

    surface = SDL_SetVideoMode(SCREEN_WIDTH, SCREEN_HEIGHT, 32, SDL_SWSURFACE);

    if (surface == NULL) {
        fatal("Failed to set video mode: %s\n", SDL_GetError());
    }
    
    
    FBNativeWindowType windowId = (FBNativeWindowType)NULL;
    SDL_SysWMinfo  wminfo;
    memset(&wminfo, 0, sizeof(wminfo));
    SDL_GetWMInfo(&wminfo);
    windowId = wminfo.info.x11.window;
    
    DPRINTF(GraphicsCalls,"initializing renderer process\n");
    
    if(!(0==android_initOpenglesEmulation() and 0==android_startOpenglesRenderer(SCREEN_WIDTH, SCREEN_HEIGHT)))
    {
        fatal("couldn't initialize openglesEmulation and/or starting openglesRenderer");
    }
    
    android_showOpenglesWindow(windowId,0,0,SCREEN_WIDTH,SCREEN_HEIGHT,0.0);
}

void* android_repaint_t(void * arg) {
    //repaintOpenGLDisplay();
    //SDL_Flip(surface);
    return NULL;
}

void android_repaint() {
    android_repaint_t(NULL);
}

void init_gem5_graphics(){
    static bool init= false;
    
    if(!init){
        android_showWindow();
        init=true;
    }
}

void get_android_OpenGL_contexts(void** list, int* n){
    gem5GetOpenGLContexts((gem5Ctx**)list, n);
}

void* create_android_OpenGL_context(int config, int isGl2){
    return gem5CreateOpenGLContext(config, isGl2);
}
