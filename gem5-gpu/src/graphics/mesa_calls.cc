#include "mesa_gpgpusim.h"
#include "base/output.hh"

extern "C" {
#include "tgsi/tgsi_exec.h"
}

extern renderData_t g_renderData;

extern "C" bool gpgpusimSimulationActive() {
    return g_renderData.GPGPUSimSimulationActive();
}

extern "C" bool gpgpusimSkipCpFrames() {
    return g_renderData.GPGPUSimSkipCpFrames();
}

extern "C" void gpgpusimEndOfFrame() {
    g_renderData.endOfFrame();
}

extern "C" void gpgpusimInitializeCurrentDraw(struct tgsi_exec_machine* tmachine, void* sp, void* mapped_indices) {
    if(!gpgpusimSimulationActive()) return;
    g_renderData.initializeCurrentDraw(tmachine, sp, mapped_indices);
}

extern "C" void gpgpusimFinalizeCurrentDraw() {
    g_renderData.finalizeCurrentDraw();
}

/*extern "C" GLboolean gpgpusim_vertexShader(GLvector4f ** inputParams, vp_stage_data * stage){
    return g_renderData.doVertexShading(inputParams, stage);
}*/

//adding the data of each span fragment so we can render them in the order we want
//instead of restricting ourselves in rendering each span one at the time
extern "C" void gpgpusimAddFragQuad(struct  tgsi_exec_machine *mach,
                                    int firstInput, int lastInput, uint32_t mask) {
    std::vector<fragmentData_t> frags(TGSI_QUAD_SIZE);

    for(int qf=0; qf < TGSI_QUAD_SIZE; qf++){
      frags[qf].isLive = mask & (1 << qf)? true: false;
      frags[qf].quadIdx = qf;
      frags[qf].uintPos[0] = (unsigned) mach->QuadPos.xyzw[0].f[qf];
      frags[qf].uintPos[1] = (unsigned) mach->QuadPos.xyzw[1].f[qf];
      assert(mach->QuadPos.xyzw[2].f[qf] < 1.0);
      frags[qf].uintPos[2] = (unsigned) (mach->QuadPos.xyzw[2].f[qf]*
            g_renderData.getMesaCtx()->DrawBuffer->_DepthMaxF);
    }

    //add input files
    for (int j = 0; j < TGSI_QUAD_SIZE; j++) {
      if(!frags[j].isLive){
        continue;
      }
      for (int i = firstInput; i <= lastInput; ++i) {
        ch4_t elm;
        elm[0] = mach->Inputs[i].xyzw[0].f[j];
        elm[1] = mach->Inputs[i].xyzw[1].f[j];
        elm[2] = mach->Inputs[i].xyzw[2].f[j];
        elm[3] = mach->Inputs[i].xyzw[3].f[j];
        frags[j].inputs.push_back(elm);
      }
    }

    g_renderData.addFragmentsQuad(frags);
}

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

extern "C" void gpgpusimGetFrameDrawcallNum(int* frameNum, int* drawcallNum){
  g_renderData.getFrameDrawcallNum(frameNum, drawcallNum);
}

extern "C" void gpgpusimGetOutputDir(const char** outputDir){
  *outputDir = g_renderData.getShaderOutputDir();
}

extern "C" void gpgpusimSetContext(struct gl_context *ctx){
  g_renderData.setMesaCtx(ctx);
}

extern "C" void gpgpusimDoFragmentShading(){
  if(!gpgpusimSimulationActive()) return;
  g_renderData.doFragmentShading();
}

extern "C" void gpgpusimAddTexelFetch(int x, int y, int level){
  if(!gpgpusimSimulationActive()) return;
  g_renderData.addTexelFetch(x, y, level);
}

extern "C" bool gpgpusimIsBusy(){
  return g_renderData.isBusy();
}


extern "C" void gpgpusimUpdateMachine(struct tgsi_exec_machine* tmachine){
  g_renderData.updateMachine(tmachine);
}

extern "C" bool gpgpusimGenerateDepthCode(FILE* inst_stream){
   g_renderData.generateDepthCode(inst_stream);
}
