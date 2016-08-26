
/* vecutil.h - vector math prototypes */

/* Copyright NVIDIA Corporation, 2000. */

extern void v3zero(float *v);
extern void v3set(float *v, float x, float y, float z);
extern void v3sub(const float *src1, const float *src2, float *dst);
extern void v3copy(const float *v1, float *v2);
extern void v3cross(const float *v1, const float *v2, float *cross);
extern float v3sqlength(const float *v);
extern float v3length(const float *v);
extern void v3scale(float *v, float div);
extern void v3normal(float *v);
extern float v3dot(const float *v1, const float *v2);
extern void v3add(const float *src1, const float *src2, float *dst);

extern float v4dot(const float *v1, const float *v2);

extern void v3print(const char *msg, const float *v);
extern void v4print(const char *msg, const float *v);
