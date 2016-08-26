#include "loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#ifdef _WIN32
#include <windows.h>
#define strcasecmp stricmp
#else
#include <dlfcn.h>
#include <strings.h>
#endif

static void HandleOptions(int argc, char *argv[]);
static void PrintUsage(void);
static void ScanVersionString(const char* pVersionString);
static void DumpProfiles(void);
static void DumpEntries(void);

static const char* g_pLibrary = "";
static const char* g_pVersion = NULL;
static const char* g_pArch = "unknown";
static const char* g_pBuildInfo = NULL;
static const char* g_pLibraryName = NULL;
static const char* g_pExeName = NULL;
static int g_VersionMajor = 0;
static int g_VersionMinor = 0;
static int g_VersionRelease = 0;
static int g_DumpProfiles = 0;
static int g_DumpEntries = 0;

#ifdef _WIN32
static char tempSpace[2048];
#endif

int main( int argc, char *argv[] )
{
  // library name is platform specific

#ifdef _WIN32
  g_pLibraryName = "cg.dll";
#else
#ifdef __APPLE__
  g_pLibraryName = "Cg";
#else
  g_pLibraryName = "libCg.so";
#endif
#endif

  // if any arguments were passed, assume that the first
  // contains the path to the library of interest

  HandleOptions(argc, argv);

  // load the library

  if ( !lInit( g_pLibraryName ) || !cgGetString ) {
    printf( "  failed to load library: %s\n", g_pLibraryName );
#ifdef _WIN32
    FormatMessage( FORMAT_MESSAGE_FROM_SYSTEM, NULL,
                   GetLastError(), 0, tempSpace, 2048, NULL );
    printf( "  LoadLibrary error: %s", tempSpace );
#else
    printf( "  failed to load library: %s\n", g_pLibraryName );
    printf( "  dlopen error: %s\n", dlerror() );
#endif
    return 1;
  }

  // get the name of the library which satisfied this symbol.  this
  // is useful if the dynamic loader located the library by searching
  // along LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, or etc.

  g_pLibrary = lGetModuleFileName( lHandle, (void *) cgGetString );

  // get version from library

  g_pVersion = cgGetString( CG_VERSION );
  g_pBuildInfo = cgGetString( (CGenum) 4151 );

  ScanVersionString(g_pVersion);

  // Architecture for Windows, Mac OS X, Linux, Solaris

#if defined(_WIN64)
  g_pArch = "x86_64";
#elif defined(_WIN32)
  g_pArch = "i386";
#elif defined(__APPLE__) || defined(__linux) || defined(__sun)
#if defined(__i386__) || defined(__i386)
  g_pArch = "i386";
#elif defined(__x86_64__) || defined(__x86_64)
  g_pArch = "x86_64";
#elif defined(__ppc__)
  g_pArch = "ppc";
#endif
#endif

  // print results

  printf( "  Library      = %s\n", ( g_pLibrary ? g_pLibrary : "" ) );
  printf( "  Version      = %s\n", ( g_pVersion ? g_pVersion : "" ) );

#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__) || defined(__linux) || defined(__sun)
  printf( "  Architecture = %s\n", g_pArch );
#endif

  printf( "  Build Info   = %s\n", ( g_pBuildInfo ? g_pBuildInfo : "" ) );
  printf( "  cgGetString  = %p\n", cgGetString );

  DumpProfiles();
  DumpEntries();

  return 0;
}

static void PrintUsage(void)
{
  printf( "Usage: %s [-profiles] [/path/to/library]\n", g_pExeName );
}

static void HandleOptions(int argc, char *argv[])
{
  int ii;

  // first argv is program name

  g_pExeName = argv[0];

  // last argv (if it doesn't begin with '-') is library name

  if ( argc > 1 && *(argv[argc-1]) != '-') {
    g_pLibraryName = argv[argc-1];
    argc--;
  }

  // Parse remaining arguments

  for (ii = 1; ii < argc; ii++) {
    if (!strcasecmp(argv[ii], "-profiles")) {
      g_DumpProfiles = 1;
    } else if (!strcasecmp(argv[ii], "-entries")) {
      g_DumpEntries = 1;
    } else {
      PrintUsage();
      exit( 1 );
    }
  }
}

static void ScanVersionString(const char* pVersionString)
{
  char* pNext = NULL;
  g_VersionMajor = strtol( pVersionString, &pNext, 10 );
  pNext++;
  g_VersionMinor = strtol( pNext, &pNext, 10 );
  pNext++;
  g_VersionRelease = strtol( pNext, &pNext, 10 );
}

static void DumpProfiles(void)
{
  int ii;
  int nProfiles;
  CGprofile profile;

  // dump profiles when:
  //   option -profiles passed in command line, and
  //   library version is 2.2 or greater

  if ( !g_DumpProfiles ||
       (g_VersionMajor < 2) ||
       (g_VersionMajor == 2 && g_VersionMinor < 2) ) {
    return;
  }

  if ( !cgGetNumSupportedProfiles ||
       !cgGetSupportedProfile ||
       !cgGetProfileString ) {
    return;
  }

  nProfiles = cgGetNumSupportedProfiles();

  printf("\n  Supported profiles:\n\n");

  for (ii=0; ii<nProfiles; ++ii) {
    profile = cgGetSupportedProfile(ii);
    printf("    %i = %s\n", profile, cgGetProfileString(profile));
  }
}

static void DumpEntries(void)
{
  if ( !g_DumpEntries ) {
    return;
  }

  printf("\n  Available entry points:\n\n");
  lList("    ", stdout);
}
