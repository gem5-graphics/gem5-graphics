#ifndef __GPGPUSIM_CALLS_H_
#define __GPGPUSIM_CALLS_H_

void getBlendingMode(unsigned  * src, unsigned  * dst, unsigned* srcAlpha, unsigned * dstAlpha,
        unsigned* eqnRGB, unsigned* eqnAlpha, float * blendColor);

unsigned readMESABufferWidth();

//checks if a vertex finishes execution
void checkGraphicsThreadExit(void* kernelPtr, unsigned tid);

bool isDepthTestEnabled(void);

bool isBlendingEnabled(void);

#endif /* __GPGPUSIM_CALLS_H_ */
