#include "graphics/mesa_gpgpusim.h"
#include "graphics/zunit.hh"


void startEarlyZ(CudaGPU* cudaGPU, uint64_t depthBuffStart, uint64_t depthBuffEnd, unsigned bufWidth, std::vector<RasterTile* >* tiles, DepthSize dSize, GLenum depthFunc,
      uint8_t* depthBuf, unsigned frameWidth, unsigned frameHeight, unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir){
   ZUnit* zunit = cudaGPU->getZUnit();
   //printf("zunit ptr =%x \n", zunit);
   zunit->startEarlyZ(depthBuffStart, depthBuffEnd, bufWidth, tiles, dSize, depthFunc, 
         depthBuf,frameWidth, frameHeight, tileH, tileW, blockH, blockW, dir);
}
