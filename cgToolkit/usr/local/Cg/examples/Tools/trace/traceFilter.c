#define TRACE_INTERNALS
#include "trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

int traceEnabled(const char *name)
{
  /* Customized logic goes here...              */
  /*   ...exclude packets by name               */
  /*   ...exclude packets based on call stack   */
  /*   ...exclude packets based on frame number */
  /*   ...exclude packets based on API prefix   */
  /*   ...exclude based on user-supplied script */

  return 1; /* Output everything by default */
}
