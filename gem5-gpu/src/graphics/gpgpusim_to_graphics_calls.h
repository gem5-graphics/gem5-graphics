#ifndef __GPGPUSIM_CALLS__
#define __GPGPUSIM_CALLS__

#include <vector> 
extern "C" {
#include "main/mtypes.h"
#include "pipe/p_state.h"
#include "pipe/p_shader_tokens.h"
}

enum shaderAttribs_t {
     FRAG_ACTIVE = PIPE_MAX_SHADER_INPUTS+1,
     QUAD_INDEX = PIPE_MAX_SHADER_INPUTS+2,
     FRAG_UINT_POS = PIPE_MAX_SHADER_INPUTS+3,
     VERT_ACTIVE = VERT_ATTRIB_MAX+1
};

union shaderAttrib_t {
  uint32_t u32;
  float f32;
  uint32_t u64;
};

//called by gpgpusim
bool isDepthTestEnabled();
bool isBlendingEnabled();

void getBlendingMode(unsigned  * src, unsigned  * dst, unsigned* srcAlpha, unsigned * dstAlpha,
        unsigned* eqnRGB, unsigned* eqnAlpha, float * blendColor);

void graphics_gpgpusim_init_options(bool standalone_mode, unsigned int startFrame, unsigned int endFrame, int start_frame, unsigned int end_frame,
        unsigned int tile_H, unsigned int tile_W, unsigned int block_H, unsigned int block_W,
        unsigned blendMode, unsigned depthMode, unsigned cptStartFrame, unsigned cptEndFrame, unsigned cptPeriod, bool skipCpFrames, char* outdir);

unsigned readMESABufferWidth();
unsigned readMESABufferSize();

//reading fragment attributes, used by the fragment shading stage
shaderAttrib_t readFragmentAttribs(unsigned threadID, unsigned attribID,
                                   unsigned attribIndex, unsigned fileIdx, unsigned idx2D, void* stream);

uint32_t readVertexAttribs(unsigned threadID, unsigned attribID, unsigned attribIndex, void* stream);

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

std::vector<uint64_t> fetchMesaTexels(int modifier, int unit, int dim,
                                      float* coords, int num_coords, float* dst,
                                      int num_dst, unsigned tid, bool isTxq, bool isTxb);

unsigned getMesaTexelSize(int samplingUnit);

unsigned getMesaFramebufferFormat();

uint64_t getFramebufferFragmentAddr(uint64_t x, uint64_t y, uint64_t size);

#endif
