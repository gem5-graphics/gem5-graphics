#define TRACE_INTERNALS
#include "trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

/* MS compiler has _snprintf rather than snprintf */

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

/* Indentation and output files */

int   indent = 0;
int   indentStep = 2;
char  indentBuffer[1024];
int   base64 = 1;
long  maxBlobSize = ~0;
int   timeStamp = 0;

FILE *traceOut = NULL;
FILE *traceErr = NULL;

#define bufferSize 1024
#define MAX_GET_PROC_ADDRESS 32

static GetProcAddressFunc getProcAddress[MAX_GET_PROC_ADDRESS];
static int numGetProcAddress = 0;

/* Utility function for environment variables */

static const char *getEnvironment(const char *var)
{
  const char *ret = NULL;

  if (var)
  {
    ret = getenv(var);

    /* Treat empty environment variable the same as non-existant */
    if (!ret || *ret=='\0')
      return NULL;
  }

  return ret;
}

/* Utility function for checking if a file exists */

static int fileExists(const char *path)
{
  FILE *f = fopen(path,"r");
  if (f)
    fclose(f);
  return f!=NULL;
}

/* Init */

void traceInit(void)
{
  static int initialized   = 0;
  const char *outFileName  = NULL;
  const char *errFileName  = NULL;
  const char *base64Env    = NULL;
  const char *maxBlobEnv   = NULL;
  const char *timeStampEnv = NULL;

  if (!initialized)
  {
    outFileName = getEnvironment("CG_TRACE_FILE");
    errFileName = getEnvironment("CG_TRACE_ERROR");
    if (outFileName)
    {
      traceOut = fopen(outFileName, "w");
      if (!traceOut)
        fprintf(stderr, "Failed to open trace file: %s\n", outFileName);
    }
    if (errFileName)
    {
      if (outFileName && !strcmp(errFileName, outFileName))
        traceErr = traceOut;
      else
      {
        traceErr = fopen(errFileName, "w");
        if (!traceErr)
          fprintf(stderr, "Failed to open error file: %s\n", errFileName);
      }
    }
    traceOut = traceOut ? traceOut : stdout;
    traceErr = traceErr ? traceErr : stderr;

    base64Env = getEnvironment("CG_TRACE_BASE64");
    if (base64Env)
      base64 = atoi(base64Env);

    maxBlobEnv = getEnvironment("CG_TRACE_BLOB_LIMIT");
    if (maxBlobEnv)
      maxBlobSize = strtoul(maxBlobEnv, NULL, 0);

    timeStampEnv = getEnvironment("CG_TRACE_TIMESTAMP");
    if (timeStampEnv)
      timeStamp = atoi(timeStampEnv);

    if (!timeInit())
      timeStamp = 0;

    initialized = 1;
  }
}

const char *traceLibraryLocation(const char *lib)
{
  /* Force MS compiler to support floating point */

#if defined(_WIN32)
  float dummy = 0.0f;
#endif

  /* Where to look for Cg runtime libraries */

  const char *ret = NULL;

  static char buffer[bufferSize];

  traceInit();

  if (!strcmp(lib, "Cg"))
  {
    /* For OSX, use libCg.dylib symlink */

#if defined(__APPLE__)
    return "libCg.dylib";
#endif

    /* First try CG_TRACE_CG_LIBRARY variable */

    ret = getEnvironment("CG_TRACE_CG_LIBRARY");

    /* Second, try CG_BIN_PATH or CG_LIB_PATH variable */

    if (!ret)
    {
#if defined(_WIN32)

      /* 64-bit libraries in CG_BIN64_PATH, 32-bit libraries in CG_BIN_PATH */

# if defined(_WIN64)
      ret = getEnvironment("CG_BIN64_PATH");
# else
      ret = getEnvironment("CG_BIN_PATH");
# endif

      if (ret)
      {
        snprintf(buffer, bufferSize, "%s\\cg.dll", ret);
        ret = buffer;
      }
#else
      ret = getEnvironment("CG_LIB_PATH");
      if (ret)
      {
        snprintf(buffer, bufferSize, "%s/libCg.so", ret);
        ret = buffer;
      }
#endif
    }

    /* Third, try default installation location */

    if (!ret)
    {
#if defined(_WIN32)
      ret = getEnvironment("PROGRAMFILES");
      if (ret)
      {
        snprintf(buffer, bufferSize, "%s\\NVIDIA Corporation\\Cg\\bin\\cg.dll", ret);
        ret = buffer;
      }
#else

      /* Linux and Solaris */

#if defined(__x86_64__) || defined(__x86_64)
      ret = "/usr/lib/amd64/libCg.so"; /* Solaris */
      if (!fileExists(ret))
        ret = "/usr/lib64/libCg.so";   /* RedHat */
      if (!fileExists(ret))
        ret = "/usr/lib/libCg.so";     /* Ubuntu */
#else
      ret = "/usr/lib32/libCg.so";     /* Ubuntu */
      if (!fileExists(ret))
        ret = "/usr/lib/libCg.so";     /* RedHat & Solaris */
#endif

#endif
    }
  }

  if (!strcmp(lib, "CgGL"))
  {
    /* For OSX, use libCg.dylib symlink */

#if defined(__APPLE__)
    return "libCg.dylib";
#endif

    /* First try CG_TRACE_CGGL_LIBRARY variable */

    ret = getEnvironment("CG_TRACE_CGGL_LIBRARY");

    /* Second, try CG_BIN_PATH or CG_LIB_PATH variable */

    if (!ret)
    {
#if defined(_WIN32)

      /* 64-bit libraries in CG_BIN64_PATH, 32-bit libraries in CG_BIN_PATH */

# if defined(_WIN64)
      ret = getEnvironment("CG_BIN64_PATH");
# else
      ret = getEnvironment("CG_BIN_PATH");
# endif

      if (ret)
      {
        snprintf(buffer, bufferSize, "%s\\cgGL.dll", ret);
        ret = buffer;
      }
#else
      ret = getEnvironment("CG_LIB_PATH");
      if (ret)
      {
        snprintf(buffer, bufferSize, "%s/libCgGL.so", ret);
        ret = buffer;
      }
#endif
    }

    /* Third, try default installation location */

    if (!ret)
    {
#if defined(_WIN32)
      ret = getEnvironment("PROGRAMFILES");
      if (ret)
      {
        snprintf(buffer, bufferSize, "%s\\NVIDIA Corporation\\Cg\\bin\\cgGL.dll", ret);
        ret = buffer;
      }
#else

      /* Linux and Solaris */

#if defined(__x86_64__) || defined(__x86_64)
      ret = "/usr/lib/amd64/libCgGL.so"; /* Solaris */
      if (!fileExists(ret))
        ret = "/usr/lib64/libCgGL.so";   /* RedHat */
      if (!fileExists(ret))
        ret = "/usr/lib/libCgGL.so";     /* Ubuntu */
#else
      ret = "/usr/lib32/libCgGL.so";     /* Ubuntu */
      if (!fileExists(ret))
        ret = "/usr/lib/libCgGL.so";     /* RedHat & Solaris */
#endif

#endif
    }
  }

  if (!strcmp(lib, "GL") || !strcmp(lib, "WGL") || !strcmp(lib, "GLX"))
  {
    /* For OSX, use libGL.dylib symlink */

#if defined(__APPLE__)
    return "libGL.dylib";
#endif

    /* First, try CG_TRACE_GL_LIBRARY variable */

    ret = getEnvironment("CG_TRACE_GL_LIBRARY");

    /* Second, try default installation location */

    if (!ret)
    {
#if defined(_WIN32)
      ret = getEnvironment("windir");
      if (ret)
      {
        /* TODO: How to choose System32 vs SysWOW64 */
        snprintf(buffer, bufferSize, "%s\\system32\\opengl32.dll", ret);
        ret = buffer;
      }
#else

      /* Linux and Solaris */

#if defined(__x86_64__) || defined(__x86_64)
      ret = "/usr/lib/amd64/libGL.so.1";              /* Solaris */
      if (!fileExists(ret))
        ret = "/usr/lib64/libGL.so.1";                /* RedHat */
      if (!fileExists(ret))
        ret = "/usr/lib/nvidia-current/libGL.so.1";   /* Ubuntu NVIDIA */
      if (!fileExists(ret))
        ret = "/usr/lib/libGL.so.1";                  /* Ubuntu */
#else
      ret = "/usr/lib32/nvidia-current/libGL.so.1";   /* Ubuntu NVIDIA */
      if (!fileExists(ret))
        ret = "/usr/lib32/libGL.so.1";                /* Ubuntu */
      if (!fileExists(ret))
        ret = "/usr/lib/libGL.so.1";                  /* RedHat & Solaris */
#endif

#endif
    }
  }

  if (!strcmp(lib, "GLUT"))
  {
    /* For OSX, use libGLUT.dylib symlink */

#if defined(__APPLE__)
    return "libGLUT.dylib";
#endif

    /* First try CG_TRACE_GLUT_LIBRARY variable */

    ret = getEnvironment("CG_TRACE_GLUT_LIBRARY");

    /* Second, try CG_BIN_PATH or CG_LIB_PATH variable */

    if (!ret)
    {
#if defined(_WIN32)

      /* 64-bit libraries in CG_BIN64_PATH, 32-bit libraries in CG_BIN_PATH */

# if defined(_WIN64)
      ret = getEnvironment("CG_BIN64_PATH");
# else
      ret = getEnvironment("CG_BIN_PATH");
# endif

      if (ret)
      {
        snprintf(buffer, bufferSize, "%s\\glut32.dll", ret);
        ret = buffer;
      }
#endif
    }

    /* Third, try default installation location */

    if (!ret)
    {
#if defined(_WIN32)
      ret = getEnvironment("PROGRAMFILES");
      if (ret)
      {
        snprintf(buffer, bufferSize, "%s\\NVIDIA Corporation\\Cg\\bin\\glut32.dll", ret);
        ret = buffer;
      }
#else

      /* Linux and Solaris */

#if defined(__x86_64__) || defined(__x86_64)
      ret = "/usr/lib/amd64/libglut.so.3"; /* Solaris */
      if (!fileExists(ret))
        ret = "/usr/lib64/libglut.so.3";   /* RedHat */
      if (!fileExists(ret))
        ret = "/usr/lib/libglut.so.3";     /* Ubuntu */
#else
      ret = "/usr/lib32/libglut.so.3";     /* Ubuntu */
      if (!fileExists(ret))
        ret = "/usr/lib/libglut.so.3";     /* RedHat & Solaris */
#endif

#endif
    }
  }

  return ret;
}

void
traceRegister(GetProcAddressFunc f)
{
  traceInit();

  if (f && numGetProcAddress < MAX_GET_PROC_ADDRESS)
    getProcAddress[numGetProcAddress++] = f;
}

void *
traceGetProcAddress(const char *name)
{
  int i;
  void *proc = NULL;

  if (!name)
    return NULL;

  for (i = 0; i < numGetProcAddress; ++i)
  {
    proc = getProcAddress[i](name);
    if (proc)
      return proc;
  }

  return NULL;
}

void updateIndent(int step)
{
  int i;

  if (step>0)
  {
    for (i=0; i<step; ++i)
      indentBuffer[indent++] = ' ';
  }
  else
    indent += step;

  indentBuffer[indent] = '\0';
}

void traceInfo(const char *format, ...)
{
  va_list argptr;

  traceInit();

  fprintf(traceErr, "info: ");
  va_start(argptr, format);
  vfprintf(traceErr, format, argptr);
  va_end(argptr);
  fprintf(traceErr, "\n");
}

void traceWarning(const char *format, ...)
{
  va_list argptr;

  traceInit();

  fprintf(traceErr, "warning: ");
  va_start(argptr, format);
  vfprintf(traceErr, format, argptr);
  va_end(argptr);
  fprintf(traceErr, "\n");
}

void traceError(const char *format, ...)
{
  va_list argptr;

  traceInit();

  fprintf(traceErr, "error: ");
  va_start(argptr, format);
  vfprintf(traceErr, format, argptr);
  va_end(argptr);
  fprintf(traceErr, "\n");
  fflush(traceErr);
}
