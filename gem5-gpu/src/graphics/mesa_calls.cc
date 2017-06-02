#include "mesa_gpgpusim.h"
#include "mesa_calls.h"

extern renderData_t g_renderData;


extern "C" bool GPGPUSimSimulationActive() {
    return g_renderData.GPGPUSimSimulationActive();
}

extern "C" bool GPGPUSimSkipCpFrames() {
    return g_renderData.GPGPUSimSkipCpFrames();
}

extern "C" void gpgpusimEndOfFrame() {
    g_renderData.endOfFrame();
}

extern "C" void gpgpusimInitializeCurrentDraw(struct gl_context *ctx) {
    if(!GPGPUSimSimulationActive()) return;
    if (ctx->Shader.ActiveProgram == NULL) {
       printf("mesa-gpgpusim: active shader found\n");
    } else {
       printf("mesa-gpgpusim: no active shader found!\n");
    }
    g_renderData.initializeCurrentDraw(ctx);
}

extern "C" void gpgpusimFinilizeCurrentDraw() {
    g_renderData.finilizeCurrentDraw();
}

extern "C" GLboolean gpgpusim_vertexShader(GLvector4f ** inputParams, vp_stage_data * stage){
    return g_renderData.doVertexShading(inputParams, stage);
}

//adding the data of each span fragment so we can render them in the order we want
//instead of restricting ourselves in rendering each span one at the time
extern "C" void addSpanFragments(SWspan *span) {
    g_renderData.addFragmentsSpan(span);
}

//called to mark the beginging of a new primitive
extern "C" void gpgpusimAddPrimitive(){
    if(!GPGPUSimSimulationActive()) return;
    g_renderData.addPrimitive();
}

