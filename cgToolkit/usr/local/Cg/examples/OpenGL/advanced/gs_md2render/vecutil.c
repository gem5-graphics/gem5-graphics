
/* vecutil.c - vector math utilities */

/* Copyright NVIDIA Corporation, 2000. */

#include <math.h>
#include "vecutil.h"

void
v3zero(float *v)
{
    v[0] = 0.0;
    v[1] = 0.0;
    v[2] = 0.0;
}

void
v3set(float *v, float x, float y, float z)
{
    v[0] = x;
    v[1] = y;
    v[2] = z;
}

void
v3sub(const float *src1, const float *src2, float *dst)
{
    dst[0] = src1[0] - src2[0];
    dst[1] = src1[1] - src2[1];
    dst[2] = src1[2] - src2[2];
}

void
v3copy(const float *v1, float *v2)
{
    register int i;
    for (i = 0 ; i < 3 ; i++)
        v2[i] = v1[i];
}

void
v3cross(const float *v1, const float *v2, float *cross)
{
    float temp[3];

    temp[0] = (v1[1] * v2[2]) - (v1[2] * v2[1]);
    temp[1] = (v1[2] * v2[0]) - (v1[0] * v2[2]);
    temp[2] = (v1[0] * v2[1]) - (v1[1] * v2[0]);
    v3copy(temp, cross);
}

float
v3sqlength(const float *v)
{
  return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

float
v3length(const float *v)
{
    return (float) sqrt(v3sqlength(v));
}

void
v3scale(float *v, float div)
{
    v[0] *= div;
    v[1] *= div;
    v[2] *= div;
}

void
v3normal(float *v)
{
    v3scale(v, 1.0f/v3length(v));
}

float
v3dot(const float *v1, const float *v2)
{
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

void
v3add(const float *src1, const float *src2, float *dst)
{
    dst[0] = src1[0] + src2[0];
    dst[1] = src1[1] + src2[1];
    dst[2] = src1[2] + src2[2];
}

float
v4dot(const float *v1, const float *v2)
{
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3];
}

#include <stdio.h>

void
v3print(const char *msg, const float *v)
{
  printf("v3: %s = [ %f, %f, %f ]\n", msg,
    v[0], v[1], v[2]);
}

void
v4print(const char *msg, const float *v)
{
  printf("v4: %s = [ %f, %f, %f, %f ]\n", msg,
    v[0], v[1], v[2], v[3]);
}
