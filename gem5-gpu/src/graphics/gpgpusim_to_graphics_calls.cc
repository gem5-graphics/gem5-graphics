#include "gpgpusim_to_graphics_calls.h"
#include "mesa_gpgpusim.h"

extern renderData_t g_renderData;


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

unsigned readMESABufferWidth() {
    return g_renderData.getScreenWidth();
}

unsigned readMESABufferSize(){
   return g_renderData.getRBSize();
}

//reading fragment attributes, used by the fragment shading stage
shaderAttrib_t readFragmentAttribs(unsigned uniqueThreadID, unsigned tid, unsigned attribID,
                                   unsigned attribIndex, unsigned fileIdx, unsigned idx2D,
                                   void* stream) {
    return g_renderData.getFragmentData(uniqueThreadID, tid, attribID, attribIndex, fileIdx, idx2D, stream);
}

uint32_t readVertexAttribs(unsigned threadID, unsigned attribID, unsigned attribIndex, void* stream) {
    return g_renderData.getVertexData(threadID, attribID, attribIndex, stream);
}

//copy the result data to the store object, this store object will be used by the rest of the pipeline in MESA (rasterization and fragment shading)
void writeVertexResult(unsigned threadID, unsigned resAttribID, unsigned attribIndex, float data) {
    g_renderData.writeVertexResult(threadID, resAttribID, attribIndex, data);
}

//checks if a vertex finishes execution
void checkGraphicsThreadExit(void* kernelPtr, unsigned tid, void* stream){
    g_renderData.checkGraphicsThreadExit(kernelPtr, tid, stream);
}

//get texel size
unsigned getMesaTexelSize(int samplingUnit){
  return g_renderData.getTexelSize(samplingUnit);
}
