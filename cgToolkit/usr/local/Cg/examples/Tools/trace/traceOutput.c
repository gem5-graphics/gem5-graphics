#define TRACE_INTERNALS
#include "trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "b64.h"

/* MS compiler has _snprintf rather than snprintf */

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

/* Per-function output state */

enum State
{
  STATE_BEGIN,      /* Before the {  */
  STATE_SCOPE,      /* After the {   */
  STATE_INPUT,      /* After input:  */
  STATE_PRE,        /* After pre:    */
  STATE_NESTED,     /* After nested: */
  STATE_POST,       /* After post:   */
  STATE_OUTPUT,     /* After output: */
};

#define MAXDEPTH 4096

static enum State stack[MAXDEPTH];
static enum State *current = NULL;

static void stateBegin(void)
{
  if (current && *current==STATE_BEGIN)
  {
    *current = STATE_SCOPE;
    fprintf(traceOut, "%s{\n", indentBuffer);
  }
}

static void stateInput(void)
{
  stateBegin();
  if (current && *current<STATE_INPUT)
  {
    *current = STATE_INPUT;
    fprintf(traceOut, "%sinput:\n", indentBuffer);
    updateIndent(indentStep);
  }
}

static void statePreCondition(void)
{
  stateBegin();
  if (current && *current<STATE_PRE)
  {
    if (*current>STATE_SCOPE)
      updateIndent(-indentStep);
    *current = STATE_PRE;
    fprintf(traceOut, "%spre:\n", indentBuffer);
    updateIndent(indentStep);
  }
}

static void stateNested(void)
{
  stateBegin();
  if (current && *current<STATE_NESTED)
  {
    if (*current>STATE_SCOPE)
      updateIndent(-indentStep);
    *current = STATE_NESTED;
    fprintf(traceOut, "%snested:\n", indentBuffer);
    updateIndent(indentStep);
  }
}

static void statePostCondition(void)
{
  stateBegin();
  if (current && *current<STATE_POST)
  {
    if (*current>STATE_SCOPE)
      updateIndent(-indentStep);
    *current = STATE_POST;
    fprintf(traceOut, "%spost:\n", indentBuffer);
    updateIndent(indentStep);
  }
}

static void stateOutput(void)
{
  stateBegin();
  if (current && *current<STATE_OUTPUT)
  {
    if (*current>STATE_SCOPE)
      updateIndent(-indentStep);
    *current = STATE_OUTPUT;
    fprintf(traceOut, "%soutput:\n", indentBuffer);
    updateIndent(indentStep);
  }
}

static void printString(FILE *out, const char *formatNull, const char *format, const char *str)
{
  int i = 0;
  int j = 0;
  char *buffer = NULL;

  /* Special handling for NULL string */

  if (!str)
  {
    fprintf(out,formatNull);   /* gcc 4.4.5 warning: format not a string literal and no format arguments */
    return;
  }

  /* First determine the output length */

  for (i=0; str[i]!='\0'; ++i, ++j)
    if (str[i]=='"')
      ++j;

  /* Allocate the space */

  buffer = (char *) malloc(j+1);

  /* Copy and insert \ before each " */

  j = 0;
  for (i=0; str[i]!='\0'; ++i)
  {
    if (str[i]=='\r') 
      continue;
    if (str[i]=='"')
      buffer[j++] = '\\';
    buffer[j++] = str[i];
  }
  buffer[j++] = '\0';

  /* Output and free memory */

  fprintf(out, format, buffer);
  free(buffer);
}

static void printFiltered(FILE *out, const char *format, va_list argptr)
{
  const char          *str;
  const char         **strArray;
  const float         *floatArray;
  const double        *doubleArray;
  const char          *byteArray;
  const int           *intArray;
  const unsigned int  *uintArray;
  const long          *longArray;
  const unsigned long *ulongArray;
  const size_t        *size_tArray;
  float                floatVal;
  double               doubleVal;
  size_t               size_tVal;
  int i, size;
  long j, longSize;

  /* Special handling for NULL char * */

  if (!strcmp(format, "%s"))
  {
    str = va_arg(argptr, char *);
    printString(out, "NULL", "\"%s\"", str);
    return;
  }

  /* Special handing for NULL terminated string array */

  if (!strcmp(format, "%s[]"))
  {
    strArray = va_arg(argptr, const char **);
    if (strArray)
    {
      for (i=0; strArray[i]!=NULL; ++i)
        printString(out, "NULL ", "\"%s\" ", strArray[i]);
    }

    return;
  }

  /* Special handing for string array */

  if (!strcmp(format, "%s[%d]"))
  {
    strArray = va_arg(argptr, const char **);
    size     = va_arg(argptr, int);
    if (strArray)
    {
      for (i=0; i<size; ++i)
        printString(out, "NULL ", "\"%s\" ", strArray[i]);
    }

    return;
  }

  if (!strcmp(format, "%s[%ld]"))
  {
    strArray = va_arg(argptr, const char **);
    longSize = va_arg(argptr, long);
    if (strArray)
    {
      for (j=0; j<longSize; ++j)
        printString(out, "NULL ", "\"%s\" ", strArray[j]);
    }

    return;
  }

  /* Special handing for float and float array */

  if (!strcmp(format, "%f") && base64)
  {
    floatVal = (float) va_arg(argptr, double);
    b64Encode(out,&floatVal,sizeof(float));
    return;
  }

  if (!strcmp(format, "%f[%d]"))
  {
    floatArray = va_arg(argptr, const float *);
    size       = va_arg(argptr, int);
    if (floatArray)
    {
      if (base64)
        b64Encode(out,floatArray,sizeof(float)*size);
      else
        for (i=0; i<size; ++i)
          fprintf(out, "%f ", floatArray[i]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  if (!strcmp(format, "%f[%ld]"))
  {
    floatArray = va_arg(argptr, const float *);
    longSize   = va_arg(argptr, long);
    if (floatArray)
    {
      if (base64)
        b64Encode(out,floatArray,sizeof(float)*longSize);
      else
        for (j=0; j<longSize; ++j)
          fprintf(out, "%f ", floatArray[j]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  /* Special handing for double and double array */

  if (!strcmp(format, "%lf") && base64)
  {
    doubleVal = va_arg(argptr, double);
    b64Encode(out,&doubleVal,sizeof(double));
    return;
  }

  if (!strcmp(format, "%lf[%d]"))
  {
    doubleArray = va_arg(argptr, const double *);
    size        = va_arg(argptr, int);
    if (doubleArray)
    {
      if (base64)
        b64Encode(out,doubleArray,sizeof(double)*size);
      else
        for (i=0; i<size; ++i)
          fprintf(out, "%lf ", doubleArray[i]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  if (!strcmp(format, "%lf[%ld]"))
  {
    doubleArray = va_arg(argptr, const double *);
    longSize    = va_arg(argptr, long);
    if (doubleArray)
    {
      if (base64)
        b64Encode(out,doubleArray,sizeof(double)*longSize);
      else
        for (j=0; j<longSize; ++j)
          fprintf(out, "%lf ", doubleArray[j]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  /* Special handing for byte array */

  if (!strcmp(format, "%c[%d]"))
  {
    byteArray = va_arg(argptr, const char *);
    size      = va_arg(argptr, int);
    if (byteArray)
    {
      if ((long)size > maxBlobSize)
        fprintf(out, "%p", byteArray);
      else
      {
        if (base64)
          b64Encode(out,byteArray,sizeof(char)*size);
        else
          for (i=0; i<size; ++i)
            fprintf(out, "%d ", (int) byteArray[i]);
      }
    }
    else
      fprintf(out, "NULL");

    return;
  }

  if (!strcmp(format, "%c[%ld]"))
  {
    byteArray = va_arg(argptr, const char *);
    longSize  = va_arg(argptr, long);
    if (byteArray)
    {
      if (longSize > maxBlobSize)
        fprintf(out, "%p", byteArray);
      else
      {
        if (base64)
          b64Encode(out,byteArray,sizeof(char)*longSize);
        else
          for (j=0; j<longSize; ++j)
            fprintf(out, "%d ", (int) byteArray[j]);
      }
    }
    else
      fprintf(out, "NULL");

    return;
  }

  /* Special handing for int array */

  if (!strcmp(format, "%d[%d]"))
  {
    intArray = va_arg(argptr, const int *);
    size     = va_arg(argptr, int);
    if (intArray)
    {
      for (i=0; i<size; ++i)
        fprintf(out, "%d ", intArray[i]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  if (!strcmp(format, "%d[%ld]"))
  {
    intArray = va_arg(argptr, const int *);
    longSize = va_arg(argptr, long);
    if (intArray)
    {
      for (j=0; j<longSize; ++j)
        fprintf(out, "%d ", intArray[j]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  /* Special handing for unsigned int array */

  if (!strcmp(format, "%u[%d]"))
  {
    uintArray = va_arg(argptr, const unsigned int *);
    size      = va_arg(argptr, int);
    if (uintArray)
    {
      for (i=0; i<size; ++i)
        fprintf(out, "%u ", uintArray[i]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  if (!strcmp(format, "%u[%ld]"))
  {
    uintArray = va_arg(argptr, const unsigned int *);
    longSize  = va_arg(argptr, long);
    if (uintArray)
    {
      for (j=0; j<longSize; ++j)
        fprintf(out, "%u ", uintArray[j]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  /* Special handing for long array */

  if (!strcmp(format, "%ld[%d]"))
  {
    longArray = va_arg(argptr, const long *);
    size      = va_arg(argptr, int);
    if (longArray)
    {
      for (i=0; i<size; ++i)
        fprintf(out, "%ld ", longArray[i]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  if (!strcmp(format, "%ld[%ld]"))
  {
    longArray = va_arg(argptr, const long *);
    longSize  = va_arg(argptr, long);
    if (longArray)
    {
      for (j=0; j<longSize; ++j)
        fprintf(out, "%ld ", longArray[j]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  /* Special handing for unsigned long array */

  if (!strcmp(format, "%lu[%d]"))
  {
    ulongArray = va_arg(argptr, const unsigned long *);
    size       = va_arg(argptr, int);
    if (ulongArray)
    {
      for (i=0; i<size; ++i)
        fprintf(out, "%lu ", ulongArray[i]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  /* Special handing for ptrdiff_t and ptrdiff_t array */

  if (!strcmp(format, "%z"))
  {
    size_tVal = va_arg(argptr, size_t);
    fprintf(out, "%ld", (long) size_tVal);
    return;
  }

  if (!strcmp(format, "%z[%d]"))
  {
    size_tArray = va_arg(argptr, const size_t *);
    size        = va_arg(argptr, int);
    if (size_tArray)
    {
      for (i=0; i<size; ++i)
        fprintf(out, "%ld ", (long) size_tArray[i]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  /* Special handing for size_t and size_t array */

  if (!strcmp(format, "%zu"))
  {
    size_tVal = va_arg(argptr, size_t);
    fprintf(out, "%lu", (unsigned long) size_tVal);
    return;
  }

  if (!strcmp(format, "%zu[%d]"))
  {
    size_tArray = va_arg(argptr, const size_t *);
    size        = va_arg(argptr, int);
    if (size_tArray)
    {
      for (i=0; i<size; ++i)
        fprintf(out, "%lu ", (unsigned long) size_tArray[i]);
    }
    else
      fprintf(out, "NULL");

    return;
  }

  vfprintf(out, format, argptr);
}

void traceFunction(const char *name)
{
  traceInit();
  stateNested();
  stateBegin();

  fprintf(traceOut, "%s%s\n", indentBuffer, name);

  if (current)
    ++current;
  else
    current = stack;
  *current = STATE_BEGIN;
}

void traceBegin(void)
{
  traceInit();
}

void traceInputParameter(const char *name, const char *format, ...)
{
  va_list argptr;

  traceInit();
  stateInput();

  fprintf(traceOut, "%s%s = ", indentBuffer, name);
  va_start(argptr, format);
  printFiltered(traceOut, format, argptr);
  va_end(argptr);
  fprintf(traceOut, "\n");
}

void tracePreCondition(void)
{
  traceInit();

  if (timeStamp)
  {
    statePreCondition();
    if (timeStamp)
      fprintf(traceOut, "%stime = %.3lf\n",indentBuffer,timeElapsed());
  }
}

void tracePostCondition(void)
{
  traceInit();

  if (timeStamp)
  {
    statePostCondition();
    if (timeStamp)
      fprintf(traceOut, "%stime = %.3lf\n",indentBuffer,timeElapsed());
  }
}

void traceOutputParameter(const char *name, const char *format, ...)
{
  va_list argptr;

  traceInit();
  stateOutput();

  fprintf(traceOut, "%s%s = ", indentBuffer, name);
  va_start(argptr, format);
  printFiltered(traceOut, format, argptr);
  va_end(argptr);
  fprintf(traceOut, "\n");
}

void traceReturn(const char *format, ...)
{
  va_list argptr;

  traceInit();
  stateOutput();

  fprintf(traceOut, "%sreturn = ", indentBuffer);
  va_start(argptr, format);
  printFiltered(traceOut, format, argptr);
  va_end(argptr);
  fprintf(traceOut, "\n");
}

void traceEnd(void)
{
  traceInit();

  if (current)
  {
    if (*current>STATE_SCOPE)
      updateIndent(-indentStep);

    if (*current>STATE_BEGIN)
    {
      fprintf(traceOut, "%s}\n", indentBuffer);
      fflush(traceOut);
    }

    if (current==stack)
      current = NULL;
    else
      --current;
  }
}
