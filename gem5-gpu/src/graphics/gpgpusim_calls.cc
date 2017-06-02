#include "gpgpusim_calls.h"
#include "mesa_gpgpusim.h"

extern renderData_t g_renderData;
extern GLvector4f * vertexResultsForMesa;


//called by gpgpusim
bool isDepthTestEnabled() {
    return g_renderData.isDepthTestEnabled();
}

bool isBlendingEnabled() {
    return g_renderData.isBlendingEnabled();
}

void getBlendingMode(unsigned  * src, unsigned  * dst, unsigned* srcAlpha, unsigned * dstAlpha,
        unsigned* eqnRGB, unsigned* eqnAlpha, float * blendColor){
    g_renderData.getBlendingMode(src,  dst,  srcAlpha,  dstAlpha,  eqnRGB, eqnAlpha, blendColor);
}

void graphics_gpgpusim_init_options(unsigned int startFrame, unsigned int endFrame, int startDrawcall, unsigned int endDrawcall,
        unsigned int tile_H, unsigned int tile_W, unsigned int block_H, unsigned int block_W,
        unsigned blendMode, unsigned depthMode, unsigned cptStartFrame, unsigned cptEndFrame, unsigned cptPeriod, bool skipCpFrames, char* outdir) {
    g_renderData.initParams(startFrame, endFrame, startDrawcall, endDrawcall, tile_H, tile_W,
          block_H, block_W, blendMode, depthMode, cptStartFrame, cptEndFrame, cptPeriod, skipCpFrames, outdir);
}

unsigned readMESABufferWidth() {
    return g_renderData.getScreenWidth();
}

unsigned readMESABufferSize(){
   return g_renderData.getRBSize();
}

//reading fragment attributes, used by the fragment shading stage
float readFragmentAttribs(unsigned threadID, unsigned attribID, unsigned attribIndex, void* stream) {
    return g_renderData.getFragmentData(threadID, attribID, attribIndex, stream);
}

int readFragmentAttribsInt(unsigned threadID, unsigned attribID, unsigned attribIndex, void* stream) {
    return g_renderData.getFragmentDataInt(threadID, attribID, attribIndex, stream);
}

int readVertexAttribsInt(unsigned threadID, unsigned attribID, unsigned attribIndex, void* stream) {
    return g_renderData.getVertexDataInt(threadID, attribID, attribIndex, stream);
}
//copy the result data to the store object, this store object will be used by the rest of the pipeline in MESA (rasterization and fragment shading)
void writeVertexResult(unsigned threadID, unsigned resAttribID, unsigned attribIndex, float data) {
    g_renderData.writeVertexResult(threadID, resAttribID, attribIndex, data);
}

//checks if a vertex finishes execution
void checkGraphicsThreadExit(void* kernelPtr, unsigned tid){
    g_renderData.checkGraphicsThreadExit(kernelPtr, tid);
}
