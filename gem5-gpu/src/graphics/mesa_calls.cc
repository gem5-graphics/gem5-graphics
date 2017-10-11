#include "mesa_gpgpusim.h"

extern renderData_t g_renderData;


//extern "C" bool GPGPUSimSimulationActive() {
extern "C" bool gpgpusimSimulationActive() {
    return g_renderData.GPGPUSimSimulationActive();
}

//extern "C" bool GPGPUSimSkipCpFrames() {
extern "C" bool gpgpusimSkipCpFrames() {
    return g_renderData.GPGPUSimSkipCpFrames();
}

extern "C" void gpgpusimEndOfFrame() {
    g_renderData.endOfFrame();
}

extern "C" void gpgpusimInitializeCurrentDraw(struct gl_context *ctx, const char* fragPtxCode) {
    if(!gpgpusimSimulationActive()) return;
    g_renderData.initializeCurrentDraw(ctx, fragPtxCode);
}

extern "C" void gpgpusimFinilizeCurrentDraw() {
    g_renderData.finilizeCurrentDraw();
}

/*extern "C" GLboolean gpgpusim_vertexShader(GLvector4f ** inputParams, vp_stage_data * stage){
    return g_renderData.doVertexShading(inputParams, stage);
}*/

//adding the data of each span fragment so we can render them in the order we want
//instead of restricting ourselves in rendering each span one at the time

/*extern "C" void gpgpusimAddSpanFragments(SWspan *span) {
    g_renderData.addFragmentsSpan(span);
}*/

//called to mark the beginging of a new primitive
extern "C" void gpgpusimAddPrimitive(){
    if(!gpgpusimSimulationActive()) return;
    g_renderData.addPrimitive();
}

extern "C" void gpgpusimSetShaderRegs(int shader_type, int usedRegs){
    if(!gpgpusimSimulationActive()) return;
    if(shader_type == GL_VERTEX_SHADER)
      g_renderData.setVertShaderUsedRegs(usedRegs);
    else if (shader_type == GL_FRAGMENT_SHADER)
      g_renderData.setFragShaderUsedRegs(usedRegs);
}
