#ifndef TRACE_H
#define TRACE_H

#ifdef _WIN32
# define DLLEXPORT __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__>=4
# define DLLEXPORT __attribute__ ((visibility("default")))
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
# define DLLEXPORT __global
#else
# define DLLEXPORT
#endif

/* Central trace library public API */

typedef void *(*GetProcAddressFunc)(const char *);

DLLEXPORT extern const char *traceLibraryLocation(const char *);
DLLEXPORT extern void traceRegister(GetProcAddressFunc);
DLLEXPORT extern void *traceGetProcAddress(const char *);

DLLEXPORT extern void traceInfo   (const char *, ...);
DLLEXPORT extern void traceWarning(const char *, ...);
DLLEXPORT extern void traceError  (const char *, ...);

DLLEXPORT extern int  traceEnabled(const char *);
DLLEXPORT extern void traceBegin(void);
DLLEXPORT extern void traceFunction(const char *);
DLLEXPORT extern void traceInputParameter(const char *, const char *, ...);
DLLEXPORT extern void tracePreCondition(void);
DLLEXPORT extern void tracePostCondition(void);
DLLEXPORT extern void traceOutputParameter(const char *, const char *, ...);
DLLEXPORT extern void traceReturn(const char *, ...);
DLLEXPORT extern void traceEnd(void);

#undef DLLEXPORT

/* Central trace library private API (not exported) */

#ifdef TRACE_INTERNALS

#include <stdio.h>

extern int   indent;
extern int   indentStep;
extern char  indentBuffer[1024];
extern int   base64;
extern long  maxBlobSize;
extern FILE *traceOut;
extern FILE *traceErr;
extern int   timeStamp;

extern void updateIndent(int step);
extern void traceInit(void);

/* traceTime.c */

extern int    timeInit(void);
extern double timeElapsed(void);

#endif

#endif
