
/* showfps.c - OpenGL code for rendering frames per second */

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <stdio.h>
#include "showfps.h"

#ifdef _WIN32
#include <windows.h>  /* for QueryPerformanceCounter */
#endif

static int reportFPS = 1;
static GLfloat textColor[3];

static void drawFPS(double fpsRate)
{
  GLubyte dummy;
  char buffer[200], *c;

  glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
      glLoadIdentity();
      glOrtho(0, 1, 1, 0, -1, 1);
      glDisable(GL_DEPTH_TEST);
      glColor3fv(textColor);
      glRasterPos2f(1,1);
      glBitmap(0, 0, 0, 0, -10*9, 15, &dummy);
      sprintf(buffer, "fps %0.1f", fpsRate);
      for (c = buffer; *c != '\0'; c++)
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);
      glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

#ifndef _WIN32
#include <sys/time.h> /* for gettimeofday and struct timeval */
#endif

void handleFPS(void)
{
  static int frameCount = 0;     /* Number of frames for timing */
  static double lastFpsRate = 0;
#ifdef _WIN32
  /* Use Win32 performance counter for high-accuracy timing. */
  static __int64 freq = 0;
  static __int64 lastCount = 0;  /* Timer count for last fps update */
  __int64 newCount;

  if (!freq) {
    QueryPerformanceFrequency((LARGE_INTEGER*) &freq);
  }

  /* Update the frames per second count if we have gone past at least
     a second since the last update. */

  QueryPerformanceCounter((LARGE_INTEGER*) &newCount);
  frameCount++;
  if (((newCount - lastCount) > freq) && reportFPS) {
    double fpsRate;

    fpsRate = (double) (freq * (__int64) frameCount)  / (double) (newCount - lastCount);
    lastCount = newCount;
    frameCount = 0;
    lastFpsRate = fpsRate;
  }
#else
  /* Use BSD 4.2 gettimeofday system call for high-accuracy timing. */
  static struct timeval last_tp = { 0, 0 };
  struct timeval new_tp;
  double secs;
  
  gettimeofday(&new_tp, NULL);
  secs = (new_tp.tv_sec - last_tp.tv_sec) + (new_tp.tv_usec - last_tp.tv_usec)/1000000.0;
  if (secs >= 1.0) {
    lastFpsRate = frameCount / secs;
    last_tp = new_tp;
    frameCount = 0;
  }
  frameCount++;
#endif
  if (reportFPS) {
    drawFPS(lastFpsRate);
  }
}

void colorFPS(float r, float g, float b)
{
  textColor[0] = r;
  textColor[1] = g;
  textColor[2] = b;
}

void toggleFPS(void)
{
  reportFPS = !reportFPS;
}

void enableFPS(void)
{
  reportFPS = 1;
}

void disableFPS(void)
{
  reportFPS = 0;
}
