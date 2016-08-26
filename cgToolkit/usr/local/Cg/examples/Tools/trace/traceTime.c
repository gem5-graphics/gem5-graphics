#define TRACE_INTERNALS
#include "trace.h"

static double timerRes = 0;

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static __int64 timerStart = 0;

int timeInit()
{
  __int64 freq;
  if( QueryPerformanceFrequency( (LARGE_INTEGER *) &freq ) )
  {
    QueryPerformanceCounter( (LARGE_INTEGER *) &timerStart );
    timerRes = 1000000.0/(double)freq;
    return 1;
  }
  else
  {
    traceWarning("no high performance counter found\n");
    return 0;
  }
}

double timeElapsed()
{
  double elapsed;
  __int64 time;
  QueryPerformanceCounter( (LARGE_INTEGER *) &time );
  elapsed = (double)(time - timerStart);
  return elapsed * timerRes;
}

#else /* non-Windows */

#include <sys/time.h>

long long timerStart = 0;

int timeInit()
{
  struct timeval val;
  gettimeofday( &val, NULL );
  timerRes = 1;
  timerStart = (long long) val.tv_sec * (long long) 1000000 + (long long) val.tv_usec;
  return 1;
}

double timeElapsed()
{
  double elapsed;
  struct timeval val;
  gettimeofday( &val, NULL );
  elapsed = (double)(((long long) val.tv_sec * (long long) 1000000 + (long long) val.tv_usec) - timerStart);
  return elapsed * timerRes;
}

#endif
