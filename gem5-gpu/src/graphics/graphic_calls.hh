#ifndef __GRAPHIC_CALLS_HH__
#define __GRAPHIC_CALLS_HH__


#include "graphics/libOpenglRender/render_api_platform_types.h"
#include "graphics/libOpenglRender/dll.h"
#include "graphics/mesa_gpgpusim.h"
#include "base/trace.hh"
#include "debug/GraphicsCalls.hh"

#define SCREEN_WIDTH  1024
#define SCREEN_HEIGHT 768

void init_gem5_graphics();
int android_initOpenglesEmulation(void);
int android_startOpenglesRenderer(int width, int height);
int initOpenglesEmulationFuncs(ADynamicLibrary* rendererLib);
int android_hideOpenglesWindow(void);
int android_showOpenglesWindow(FBNativeWindowType window, int x, int y, int width, int height, float rotation);
void android_showWindow();
void android_gles_server_path(char* buff, size_t buffsize);
void android_repaint();
void* android_repaint_t(void*);
bool checkAndroidWindowInit();
void get_android_OpenGL_contexts(void**, int*);
void* create_android_OpenGL_context(int, int);
#endif
