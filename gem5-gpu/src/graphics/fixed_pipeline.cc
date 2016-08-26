#include "graphics/mesa_gpgpusim.h"`
#include "gpu/gpgpu-sim/zunit.hh"


void startEarlyZ(CudaGPU* cudaGPU, uint64_t depthBuffStart, uint64_t depthBuffEnd, unsigned bufWidth, RasterTiles* tiles, DepthSize dSize, GLenum depthFunc){
   cudaGPU->getZUnit()->startEarlyZ(depthBuffStart, depthBuffEnd, bufWidth, tiles, dSize, depthFunc);
}
