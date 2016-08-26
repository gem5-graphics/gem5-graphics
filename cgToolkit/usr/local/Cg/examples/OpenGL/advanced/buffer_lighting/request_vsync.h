
/* request_vsync.h - request buffer swap synchroization with vertical sync */
/*                   false means do NOT synchronize.                       */

static void requestSynchronizedSwapBuffers(int enableSync)
{
#if defined(__APPLE__)
#ifdef CGL_VERSION_1_2
  const GLint sync = enableSync;
#else
  const long sync = enableSync;
#endif
  CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &sync);
#elif defined(_WIN32)

  if (wglSwapIntervalEXT) {
    wglSwapIntervalEXT(enableSync);
  }
#else
  if (glXSwapIntervalSGI) {
    glXSwapIntervalSGI(enableSync);
  }
#endif
}
