#ifndef __GPGPUSIM_CALLS__
#define __GPGPUSIM_CALLS__

extern "C" {
#include "math/m_xform.h"
#include "main/mtypes.h"
}

enum shaderAttribs_t{
   FRAG_ACTIVE = FRAG_ATTRIB_MAX+1,
   VERT_ACTIVE = VERT_ATTRIB_MAX+1
};

//called by gpgpusim
bool isDepthTestEnabled();

bool isBlendingEnabled();


void getBlendingMode(unsigned  * src, unsigned  * dst, unsigned* srcAlpha, unsigned * dstAlpha,
        unsigned* eqnRGB, unsigned* eqnAlpha, float * blendColor);

void graphics_gpgpusim_init_options(unsigned int startFrame, unsigned int endFrame, int start_frame, unsigned int end_frame,
        unsigned int tile_H, unsigned int tile_W, unsigned int block_H, unsigned int block_W,
        unsigned blendMode, unsigned depthMode, unsigned cptStartFrame, unsigned cptEndFrame, unsigned cptPeriod, bool skipCpFrames, char* outdir);

unsigned readMESABufferWidth();
unsigned readMESABufferSize();

//reading fragment attributes, used by the fragment shading stage
float readFragmentAttribs(unsigned threadID, unsigned attribID, unsigned attribIndex, void* stream);
int readFragmentAttribsInt(unsigned threadID, unsigned attribID, unsigned attribIndex, void* stream);
int readVertexAttribsInt(unsigned threadID, unsigned attribID, unsigned attribIndex, void* stream);

//copy the result data to the store object, this store object will be used by the rest of the pipeline in MESA (rasterization and fragment shading)
void writeVertexResult(unsigned threadID, unsigned resAttribID, unsigned attribIndex, float data);

//checks if a vertex finishes execution
void checkGraphicsThreadExit(void* kernelPtr, unsigned tid);

bool isBlendingEnabled(void);

bool isDepthTestEnabled(void);

void checkGraphicsThreadExit(void* kernelPtr, unsigned tid);

unsigned readMESABufferWidth();

void getBlendingMode(unsigned  * src, unsigned  * dst, unsigned* srcAlpha, unsigned * dstAlpha,
        unsigned* eqnRGB, unsigned* eqnAlpha, float * blendColor);

#endif 
