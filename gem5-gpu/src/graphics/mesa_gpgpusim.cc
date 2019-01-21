// Copyright (c) 2018, Ayub A. Gubran, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <assert.h>
#include <chrono>
#include <functional>
#include <fstream>
#include <GL/gl.h>
#include <iostream>
#include <math.h>
#include <map>
#include <queue>
#include <sstream>
#include <stack>
#include <sys/stat.h>
#include <time.h>
#include <thread>
#include <unistd.h>

#include "main/macros.h"
extern "C" {
#include "drivers/dri/swrast/swrast_priv.h"
#include "gallium/drivers/softpipe/sp_context.h"
}


#include "base/statistics.hh"
#include "base/trace.hh"
#include "debug/MesaGpgpusim.hh"
#include "graphics/graphics_standalone.hh"
#include "graphics/mesa_gpgpusim.h"
#include "graphics/serialize_graphics.hh"
#include "sim/simulate.hh"
#include "sim/sim_exit.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"

extern std::mutex g_gpuMutex;
extern unsigned g_active_device;

uint64_t g_startTick;
uint64_t g_totalTicks = 0;
uint64_t g_totalFrags = 0;

void startEarlyZ(CudaGPU* cudaGPU, uint64_t depthBuffStart, uint64_t depthBuffEnd, unsigned bufWidth, std::vector<RasterTile* >* tiles, DepthSize dSize, GLenum depthFunc,
      uint8_t* depthBuf, unsigned frameWidth, unsigned frameHeight, unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir);



renderData_t g_renderData;
int sizeOfEachFragmentData = PIPE_MAX_SHADER_INPUTS * sizeof (float) * 4;

const char* VERT_ATTRIB_NAMES[33] =
{
   "VERT_ATTRIB_POS",
   "VERT_ATTRIB_WEIGHT",
   "VERT_ATTRIB_NORMAL",
   "VERT_ATTRIB_COLOR0",
   "VERT_ATTRIB_COLOR1",
   "VERT_ATTRIB_FOG",
   "VERT_ATTRIB_POINT_SIZE",
   "VERT_ATTRIB_EDGEFLAG",
   "VERT_ATTRIB_TEX0",
   "VERT_ATTRIB_TEX1",
   "VERT_ATTRIB_TEX2",
   "VERT_ATTRIB_TEX3",
   "VERT_ATTRIB_TEX4",
   "VERT_ATTRIB_TEX5",
   "VERT_ATTRIB_TEX6",
   "VERT_ATTRIB_TEX7",
   "VERT_ATTRIB_GENERIC0",
   "VERT_ATTRIB_GENERIC1",
   "VERT_ATTRIB_GENERIC2",
   "VERT_ATTRIB_GENERIC3",
   "VERT_ATTRIB_GENERIC4",
   "VERT_ATTRIB_GENERIC5",
   "VERT_ATTRIB_GENERIC6",
   "VERT_ATTRIB_GENERIC7",
   "VERT_ATTRIB_GENERIC8",
   "VERT_ATTRIB_GENERIC9",
   "VERT_ATTRIB_GENERIC10",
   "VERT_ATTRIB_GENERIC11",
   "VERT_ATTRIB_GENERIC12",
   "VERT_ATTRIB_GENERIC13",
   "VERT_ATTRIB_GENERIC14",
   "VERT_ATTRIB_GENERIC15",
   "VERT_ATTRIB_MAX"
};

shaderAttrib_t primitiveFragmentsData_t::getFragmentData(unsigned utid, unsigned tid, unsigned attribID,
                                                   unsigned attribIndex, unsigned fileIdx, 
                                                   unsigned idx2D, void * stream,
                                                   stage_shading_info_t* si, bool z_unit_disabled) {
   /*
  shaderAttrib_t retVal;
  bool isRetVal = false;
  fragmentData_t* frag = NULL;
  if(z_unit_disabled){
     if(utid < m_fragments.size()){
        frag = &m_fragments[utid];
     }
  } else {
     unsigned tileId = si->cudaStreamTiles[(uint64_t)stream].tileId;
     DPRINTF(MesaGpgpusim, "querying utid=%d, tileId=%d, tid=%d\n", tileId, utid, tid);
     RasterTile& rt = (*(*(si->earlyZTiles))[tileId]);
     if(tid < rt.size()){
        frag = &rt.getFragment(tid);
     }
  }

  switch(attribID){
    case FRAG_ACTIVE: {
      if(frag == NULL){
         retVal.u32 = 0;
      } else {
         if(z_unit_disabled){
            retVal.u32  = frag->isLive? 1 : 0;
         } else {
            retVal.u32 =  frag->isLive? (frag->passedDepth? 1 : 0) : 0;
         }
      }
        isRetVal = true;
        break;
      }
    case QUAD_INDEX: {
      assert(frag->quadIdx < TGSI_QUAD_SIZE);
      retVal.u32 =  frag->quadIdx;
      isRetVal = true;
      break;
      }
    case FRAG_UINT_POS: {
      retVal.u64 =  frag->uintPos(attribIndex);
      isRetVal = true;
      break;
    }
    case TGSI_FILE_INPUT: {
      retVal.f32 = frag->inputs[fileIdx][attribIndex];
      isRetVal = true;
      break;
    }
  }

  if(isRetVal)
    return retVal;

  retVal.f32 = frag->attribs[attribID][attribIndex];
  return retVal;
  */
}

void primitiveFragmentsData_t::addFragment(fragmentData_t fd) {
    maxDepth = std::max(maxDepth, (uint64_t) fd.uintPos(2));
    minDepth = std::min(minDepth, (uint64_t) fd.uintPos(2));
    m_fragments.push_back(fd);
}


//the current code is not efficient, might be rewritten some time later
void primitiveFragmentsData_t::sortFragmentsInRasterOrder(unsigned frameHeight, unsigned frameWidth,
        const unsigned tileH, const unsigned tileW,
        const unsigned blockH, const unsigned blockW, const RasterDirection rasterDir) {

    assert(rasterDir==RasterDirection::HorizontalRaster or 
          rasterDir==RasterDirection::BlockedHorizontal); //what we do here so far
  
    //checking if a suitable block size is provided
    assert((blockH%tileH)==0);
    assert((blockW%tileW)==0);

    //DPRINTF(MesaGpgpusim, "tileW = %d, and tileH = %d\n", tileW, tileH);
    DPRINTF(MesaGpgpusim, "Current frame size WxH=%dx%d\n", frameWidth, frameHeight);

    //adding padding for rounded pixel locations
    frameHeight+= blockH;
    frameWidth += blockW;
  
    if ( (frameWidth % blockW) != 0) {
        frameWidth -= frameWidth % blockW;
        frameWidth += blockW;
        //DPRINTF(MesaGpgpusim, "Display size width padded to %d\n", frameWidth);
    }

    if ((frameHeight % blockH) != 0) {
        frameHeight -= frameHeight % blockH;
        frameHeight += blockH;
        //DPRINTF(MesaGpgpusim, "Display size height padded to %d\n", frameHeight);
    }
    
     //DPRINTF(MesaGpgpusim, "Adjusted display size is WxH=%dx%d\n", frameWidth, frameHeight);


    //we add empty sets and then we will fill them
    const unsigned fragmentsPerTile = tileH * tileW;
    assert(0 == ((frameHeight* frameWidth) % fragmentsPerTile));
    unsigned tilesCount = (frameHeight * frameWidth) / fragmentsPerTile;


    std::vector<std::vector<fragmentData_t> > fragmentTiles(tilesCount);
    /*for (unsigned tile = 0; tile < tilesCount; tile++) {
        std::vector<fragmentData_t> aSet;
        fragmentTiles.push_back(aSet);
    }*/

    assert(fragmentTiles.size() == tilesCount);
    assert((frameWidth%tileW) == 0);
    assert((frameHeight%tileH) == 0);
            
    unsigned const numberOfHorizontalTiles = frameWidth / tileW;
    
    //now we figure which tile every fragment belongs to
    for (int frag = 0; frag < m_fragments.size(); frag++) {
        unsigned xPosition = m_fragments[frag].uintPos(0);
        unsigned yPosition = m_fragments[frag].uintPos(1);
        unsigned tileXCoord = xPosition / tileW;
        unsigned tileYCoord = yPosition / tileH; //normalize this guy
        assert(tileXCoord<numberOfHorizontalTiles);
        unsigned tileIndex = tileYCoord * numberOfHorizontalTiles + tileXCoord;
        assert(tileIndex<fragmentTiles.size());
        fragmentTiles[tileIndex].push_back(m_fragments[frag]);
                
        //make sure that we do not add more fragments in each tile than we should have
        assert(fragmentTiles[tileIndex].size() <= (tileH * tileW));
    }

    unsigned originalSize = m_fragments.size();
    m_fragments.clear();

    //now adding the fragments in the raster order, tile moves horizontally

    if (rasterDir == RasterDirection::HorizontalRaster) {
         //DPRINTF(MesaGpgpusim, "raster order: HorizontalRaster\n");
        for (unsigned tile = 0; tile < fragmentTiles.size(); tile++) {
            for (unsigned frag = 0; frag < fragmentTiles[tile].size(); frag++) {
                m_fragments.push_back(fragmentTiles[tile][frag]);
            }
        }
    } else if (rasterDir == RasterDirection::BlockedHorizontal) {
         //DPRINTF(MesaGpgpusim, "raster order: BlockedHorizontal\n");
        
        std::vector<std::vector<std::vector<fragmentData_t> > >blocks;
        unsigned blocksCount = (frameHeight * frameWidth) / (blockH * blockW);
        for (unsigned block = 0; block < blocksCount; block++) {
            std::vector < std::vector<fragmentData_t> >  aSet;
            blocks.push_back(aSet);
        }
        assert(blocks.size()==blocksCount);
        
        const unsigned numberOfHorizontalBlocks = frameWidth/blockW;
        const unsigned hTilesPerBlock = blockW/tileW;
        const unsigned vTilesPerBlock = blockH/tileH;
        
        for (unsigned tile = 0; tile < fragmentTiles.size(); tile++) {
            unsigned tileX = tile%numberOfHorizontalTiles;
            unsigned tileY = tile/numberOfHorizontalTiles;
            unsigned blockXCoord = tileX/hTilesPerBlock;
            unsigned blockYCoord = tileY/vTilesPerBlock;
            unsigned blockIndex = blockYCoord * numberOfHorizontalBlocks + blockXCoord;
            assert(blockIndex<blocks.size());
            blocks[blockIndex].push_back(fragmentTiles[tile]);
        }
        

        for (unsigned blockId = 0; blockId < blocks.size(); blockId++) {
            for (unsigned tileId = 0; tileId < blocks[blockId].size(); tileId++) {
                for (unsigned frag = 0; frag < blocks[blockId][tileId].size(); frag++) {
                    m_fragments.push_back(blocks[blockId][tileId][frag]);
                }
            }
        }
        
    } else assert(0);
    
     assert(m_fragments.size() == originalSize);
}


void renderData_t::runEarlyZ(CudaGPU * cudaGPU, unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir, unsigned clusterCount) {
/*
   RasterTiles * allTiles = new RasterTiles();
   for(int prim=0; prim < drawPrimitives.size(); prim++){
      drawPrimitives[prim].sortFragmentsInTiles(
            m_bufferHeight, m_bufferWidth,
            m_tile_H, m_tile_W, 
            m_hTiles, m_wTiles,
            m_tilesCount,
            blockH, blockW, dir, clusterCount);
      RasterTiles& primTiles = drawPrimitives[prim].getRasterTiles();
      DPRINTF(MesaGpgpusim, "prim %d tiles = %ld\n", prim, primTiles.size());
      for(int tile=0; tile < primTiles.size(); tile++){
         allTiles->push_back(primTiles[tile]);
      }
   }

   DPRINTF(MesaGpgpusim, "number of tiles = %ld\n", allTiles->size());
   uint64_t depthBuffEndAddr = (uint64_t)m_deviceData + m_colorBufferByteSize + m_depthBufferSize;
   uint64_t depthBuffStartAddr = (uint64_t)m_deviceData + m_colorBufferByteSize; 
   DPRINTF(MesaGpgpusim, "depthBuffer start = %lx, end =%lx\n", depthBuffStartAddr, depthBuffEndAddr);

   m_sShading_info.doneEarlyZ = false;
   m_sShading_info.earlyZTiles = allTiles;
   m_sShading_info.completed_threads_frags = 0;
   m_sShading_info.launched_threads_frags = 0;
   assert(m_sShading_info.fragCodeAddr == NULL);

   startEarlyZ(cudaGPU, depthBuffStartAddr, depthBuffEndAddr, m_bufferWidth, allTiles, m_depthSize, m_mesaCtx->Depth.Func, 
         m_depthBuffer, m_bufferWidth, m_bufferHeight, tileH, tileW, blockH, blockW, dir);*/
}

void primitiveFragmentsData_t::sortFragmentsInTiles(
      unsigned frameHeight, unsigned frameWidth,
      const unsigned tileH, const unsigned tileW,
      const unsigned hTiles, const unsigned wTiles,
      const unsigned tilesCount,
      const unsigned blockH, const unsigned blockW, 
      const RasterDirection rasterDir,
      unsigned tcH,
      unsigned tcW,
      unsigned tcBlockDim,
      unsigned clusterCount) {
   
    assert(m_rasterTiles.size() == 0);
    assert(rasterDir==RasterDirection::HorizontalRaster);

    //checking if a suitable block size is provided
    assert((blockH%tileH)==0);
    assert((blockW%tileW)==0);
 
    /*//adding padding for rounded pixel locations
    frameHeight+= blockH;
    frameWidth += blockW;
    
    if ( (frameWidth % blockW) != 0) {
        frameWidth -= frameWidth % blockW;
        frameWidth += blockW;
    }

    if ((frameHeight % blockH) != 0) {
        frameHeight -= frameHeight % blockH;
        frameHeight += blockH;
    }*/
    


    DPRINTF(MesaGpgpusim, "Sorting %d framgents in %d tiles \n", m_fragments.size(), tilesCount);

    int minX, maxX, minY, maxY;
    minX = minY = -1;
    maxX = maxY = -1;

    std::unordered_map<unsigned, RasterTile* > fragmentTiles;
    /*std::vector<RasterTile* > fragmentTiles;
    for (unsigned tileId = 0; tileId < tilesCount; tileId++) {
        //std::vector<fragmentData_t>* aSet = new std::vector<fragmentData_t>();
        unsigned xCoord = tileId%wTiles;
        unsigned yCoord = tileId/wTiles;
        RasterTile * rtile = new RasterTile(this, primId, tileId,
              tileH, tileW, xCoord, yCoord);
        fragmentTiles.push_back(rtile);
    }
    assert(fragmentTiles.size() == tilesCount);
     */

    assert((frameWidth%tileW) == 0);
    //assert((frameHeight%tileH) == 0);
            
    //unsigned const numberOfHorizontalTiles = frameWidth / tileW;
    
    //now we figure which tile every fragment belongs to
    for (int frag = 0; frag < m_fragments.size(); frag++) {
        unsigned xPosition = m_fragments[frag].uintPos(0);
        unsigned yPosition = m_fragments[frag].uintPos(1);
        unsigned tileXCoord = xPosition / tileW;
        unsigned tileYCoord = yPosition / tileH; //normalize this guy
        minX = (minX == -1)? tileXCoord: std::min(minX, (int)tileXCoord);
        maxX = (maxX == -1)? tileXCoord: std::max(maxX, (int)tileXCoord);
        minY = (minY == -1)? tileYCoord: std::min(minY, (int)tileYCoord);
        maxY = (maxY == -1)? tileYCoord: std::max(maxY, (int)tileYCoord);
        assert(tileXCoord<wTiles);
        unsigned tileIndex = tileYCoord * wTiles + tileXCoord;
        if(fragmentTiles.find(tileIndex) == fragmentTiles.end()){
           fragmentTiles[tileIndex] = new RasterTile(this, primId, tileIndex,
              tileH, tileW, tileXCoord, tileYCoord);
        }
        fragmentTiles[tileIndex]->addFragment(&m_fragments[frag]);
        //make sure that we do not add more fragments in each tile than we should have
        assert(fragmentTiles[tileIndex]->size() <= (tileH * tileW));
    }

    //we need to set active frags count tally
    unsigned fragCount = 0;
    for(int i=0; i< tilesCount; i++){
       if(fragmentTiles.find(i)!= fragmentTiles.end())
          fragCount += fragmentTiles[i]->resetActiveCount();
    }
    assert(fragCount == m_fragments.size());

    m_simtRasterTiles.resize(clusterCount);
    unsigned tcSize = tcH*tcW;
    unsigned tcBlockInTilesH = tcBlockDim*tcH;
    unsigned tcBlockInTilesW = tcBlockDim*tcW;
    unsigned tbpw = tcBlockInTilesW * tileW;
    unsigned xTCBlocks = frameWidth%tbpw>0? 
       ((frameWidth - (frameWidth % tbpw)) + tbpw)/tbpw : frameWidth/tbpw;
    for(int tile=0; tile < tilesCount; tile++){
       if(fragmentTiles.find(tile)!= fragmentTiles.end()){
          if(fragmentTiles[tile]->xCoord >= minX and 
                fragmentTiles[tile]->xCoord <= maxX and 
                fragmentTiles[tile]->yCoord >= minY and 
                fragmentTiles[tile]->yCoord <= maxY){
             unsigned tcBlockX = fragmentTiles[tile]->xCoord/tcBlockInTilesW;
             unsigned tcBlockY = fragmentTiles[tile]->yCoord/tcBlockInTilesH;
             unsigned tcBlockId = (tcBlockY * xTCBlocks) + tcBlockX;
             m_simtRasterTiles[tcBlockId%clusterCount].push_back(fragmentTiles[tile]);
             assert(fragmentTiles[tile]->getActiveCount() != 0);
             /*printf("assigning tile %d to cluster %d\n", tile,
               tcBlockId%clusterCount);*/
             //m_rasterTiles.push_back(fragmentTiles[tile]);
          } else {
             //all tiles should have some frags
             assert(0);
             /*assert(fragmentTiles[tile]->getActiveCount() == 0);
             delete fragmentTiles[tile];*/
          }
       }
    }

    //add terminating tiles
    for(unsigned s=0; s<clusterCount; s++){
       RasterTile* rtile = new RasterTile(NULL, -1, -1, -1, -1, -1, -1);
       m_simtRasterTiles[s].push_back(rtile);
       rtile->lastPrimTile=true;
    }

    m_validTiles = true;
}

renderData_t::renderData_t():
   m_hizBuff(this)
{
    m_deviceData = NULL;
    m_currentFrame = 0;
    m_startFrame = -1;
    m_endFrame = -1;
    //callsCount = 0;
    m_drawcall_num = 0;
    m_tcPid = -1;
    m_tcTid = -1;
    m_flagEndVertexShader = false;
    m_flagEndFragmentShader = false;
    m_inShaderBlending = false;
    m_inShaderDepth = false;

    m_usedVertShaderRegs = -1;
    m_usedFragShaderRegs = -1;

    m_currentRenderBufferBytes = NULL;
}

renderData_t::~renderData_t() {
}

shaderAttrib_t renderData_t::getFragmentData(unsigned utid, unsigned tid, unsigned attribID, unsigned attribIndex,
      unsigned fileIdx, unsigned idx2D, void * stream) {
   //bool z_unit_disabled = m_inShaderDepth or !isDepthTestEnabled(); 

   assert(utid == tid);
   if( attribID == TGSI_FILE_CONSTANT){
      shaderAttrib_t retVal;
      if((fileIdx > consts.size()) or (idx2D > consts[fileIdx].size())){
         assert(0);
         retVal.u32 = 0;
      } else {
         retVal.u32 = consts[fileIdx][idx2D][attribIndex];
      }
      return retVal;
   }
   unsigned tcSize = 0;
   tileStream_t* tcTile = m_sShading_info.getTCTile(utid, &tcSize);
   tcTilePtr_t tcTilePtr = tcTile->tcTilePtr;

   shaderAttrib_t retVal;
   bool isRetVal = false;
   assert((utid%tcSize) < tcTilePtr->size());
   fragmentData_t* frag = tcTilePtr->at(utid%tcSize) == NULL? 
      NULL: tcTilePtr->at(utid%tcSize)->frag;

   switch(attribID){
      case QUAD_ACTIVE: {
         //retVal.u32 = frag->hasLiveQuad()? 1 : 0;
         unsigned startFrag = (utid/TGSI_QUAD_SIZE) * TGSI_QUAD_SIZE;
         unsigned endFrag = startFrag + TGSI_QUAD_SIZE - 1;
         retVal.u32 = 0;
         for(unsigned qf=startFrag; qf <= endFrag; qf++){
            if(getFragmentData(qf, qf, FRAG_ACTIVE, -1, -1, -1, stream).u32 > 0){
               retVal.u32 = 1;
               break;
            }
         }
         isRetVal = true;
         break;
      }
      case FRAG_ACTIVE: {
         if(frag == NULL){
            retVal.u32 = 0;
         } else {
            //retVal.u32 =  frag->isLive? (frag->passedDepth? 1 : 0) : 0;
            retVal.u32 =  frag->isLive? 1 : 0;
         }
         isRetVal = true;
         break;
      }
      case DEPTH_TEST_NOT_ACTIVE: {
        retVal.u32 = tcTilePtr->skipDepthTest? 1: 0;
        isRetVal = true;
        break;
      }
      case QUAD_INDEX: {
        assert(frag->quadIdx < TGSI_QUAD_SIZE);
        retVal.u32 =  frag->quadIdx;
        isRetVal = true;
        break;
                       }
      case FRAG_UINT_POS: {
           retVal.u32 =  frag->uintPos(attribIndex);
           isRetVal = true;
           break;
        }
      case FRAG_DEPTH_ADDR: {
           uint64_t xPos = frag->uintPos(0);
           uint64_t yPos = frag->uintPos(1);
           retVal.u64 = ((uint64_t) (m_deviceData 
                    + m_colorBufferByteSize)
                    + m_depthBufferSize) 
              - ((yPos+1) * m_depthBufferWidth * (unsigned)m_depthSize) 
              + (xPos * (unsigned)m_depthSize);
           isRetVal = true;
           break;
       }
      case TGSI_FILE_INPUT: {
             retVal.f32 = frag->inputs[fileIdx][attribIndex];
             isRetVal = true;
             break;
          }
   }

   if(isRetVal)
      return retVal;

   retVal.f32 = frag->attribs[attribID][attribIndex];
   return retVal;
}

uint32_t renderData_t::getVertexData(unsigned utid, unsigned attribID, unsigned attribIndex, void * stream) {
   switch(attribID){
      case VERT_ACTIVE: 
         if(utid >= m_sShading_info.launched_threads_verts)  return 0;
         return 1;
         break;
      default: printf("Invalid attribID: %d \n", attribID);
               abort();
   }
}

void renderData_t::setVertexAttribsCount(int inputAttribsCount, int outputAttribsCount){
   m_sShading_info.vertInputAttribs = inputAttribsCount;
   m_sShading_info.vertOutputAttribs = outputAttribsCount;
}

void renderData_t::addVertex(struct tgsi_exec_machine* mach, int pos) { 
   vertexData_t vd(m_sShading_info.vertInputAttribs, m_sShading_info.vertOutputAttribs);
   for(unsigned i=0; i<m_sShading_info.vertInputAttribs; i++){
      ch4_t c4;
      c4[0] = mach->Inputs[i].xyzw[0].f[pos];
      c4[1] = mach->Inputs[i].xyzw[1].f[pos];
      c4[2] = mach->Inputs[i].xyzw[2].f[pos];
      c4[3] = mach->Inputs[i].xyzw[3].f[pos];
      vd.inputs.push_back(c4);
   }

   for(unsigned i=0; i<m_sShading_info.vertOutputAttribs; i++){
      ch4_t c4;
      c4[0] = mach->Outputs[i].xyzw[0].f[pos];
      c4[1] = mach->Outputs[i].xyzw[1].f[pos];
      c4[2] = mach->Outputs[i].xyzw[2].f[pos];
      c4[3] = mach->Outputs[i].xyzw[3].f[pos];
      vd.outputs.push_back(c4);
   }
   m_sShading_info.vertex_data.push_back(vd);
}

void renderData_t::addFragment(fragmentData_t fragmentData) {
    DPRINTF(MesaGpgpusim, "adding a fragment to primitive %d, fragments count=%d\n", drawPrimitives.size()-1, drawPrimitives.back().size());
    //printf( "adding a fragment to primitive %d, fragments count=%d\n", drawPrimitives.size()-1, drawPrimitives.back().size());
    drawPrimitives.back().addFragment(fragmentData);
}

void renderData_t::addPrimitive() {
    if(!GPGPUSimSimulationActive()) return;
    primitiveFragmentsData_t prim(drawPrimitives.size());
    DPRINTF(MesaGpgpusim, "adding new primitive, total = %ld\n", drawPrimitives.size()+1);
    drawPrimitives.push_back(prim);
}


void renderData_t::sortFragmentsInRasterOrder(unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir) {
   for(int prim=0; prim < drawPrimitives.size(); prim++)
      drawPrimitives[prim].sortFragmentsInRasterOrder(m_bufferHeight, m_bufferWidth, tileH, tileW, blockH, blockW, dir);
}

void renderData_t::endDrawCall() {
   printf("ending drawcall tick = %ld\n", curTick());
   printf("endDrawCall: start\n");
   uint64_t ticks = curTick() - g_startTick;
   g_totalTicks+= ticks;
   printf("totalTicks = %ld, frags = %ld\n", g_totalTicks, g_totalFrags);
   CudaGPU* cudaGPU = CudaGPU::getCudaGPU(g_active_device);
   cudaGPU->endDrawCall();
   putDataOnColorBuffer();
   if(isDepthTestEnabled())
       putDataOnDepthBuffer();
    delete [] lastFatCubin->ident;
    delete [] lastFatCubin->ptx[0].gpuProfileName;
    delete [] lastFatCubin->ptx[0].ptx;
    delete [] lastFatCubin->ptx;
    delete lastFatCubin;
    if(m_sShading_info.allocAddr) graphicsFree(m_sShading_info.allocAddr);
    if(m_sShading_info.deviceVertsAttribs) graphicsFree(m_sShading_info.deviceVertsAttribs);
    if(m_sShading_info.vertCodeAddr) graphicsFree(m_sShading_info.vertCodeAddr);
    if(m_sShading_info.fragCodeAddr) graphicsFree(m_sShading_info.fragCodeAddr);
    graphicsFree(m_deviceData);

    //free textures
    for(int tex=0; tex < m_textureInfo.size(); tex++){
      texelInfo_t* ti = &m_textureInfo[tex];
      graphicsFree((void*)ti->getBaseAddr());
    }

    lastFatCubin = NULL;
    RasterTiles * tiles = m_sShading_info.earlyZTiles;
    m_sShading_info.earlyZTiles = NULL;
    if(tiles !=NULL){
       for(int i=0; i < tiles->size(); i++){
          delete (*tiles)[i];
       }
       delete tiles;
    }
    graphicsStreamDestroy(m_sShading_info.cudaStreamVert);
    graphicsStreamDestroy(m_sShading_info.cudaStreamFrag);
    m_sShading_info.clear();
    for(auto tr: textureRefs)
       delete tr;
    textureRefs.clear();
    drawPrimitives.clear();
    if(m_depthBuffer!=NULL) {
       delete [] m_depthBuffer;
       m_depthBuffer = NULL;
    }
    consts.clear();
    //Stats::dump();
    //Stats::reset();
    struct softpipe_context *sp = (struct softpipe_context *) m_sp;
    const void* mapped_indices = m_mapped_indices;
    m_sp = NULL;
    m_mapped_indices = NULL;
    finalize_softpipe_draw_vbo(sp, mapped_indices);
    incDrawcallNum();
    m_textureInfo.clear();
    m_deviceData = NULL;
    printf("endDrawCall: done\n");
}

void renderData_t::initParams(bool standaloneMode, unsigned int startFrame, unsigned int endFrame, int startDrawcall, unsigned int endDrawcall,
        unsigned int tile_H, unsigned int tile_W, unsigned int block_H, unsigned int block_W, unsigned int tc_h, unsigned int tc_w, unsigned int tc_block_dim, unsigned int wg_size,
        unsigned blendingMode, unsigned depthMode, unsigned cptStartFrame, unsigned cptEndFrame, unsigned cptPeroid, bool skipCpFrames, char* outdir) {
    m_standaloneMode = standaloneMode;
    m_startFrame = startFrame;
    m_endFrame = endFrame;
    m_startDrawcall = startDrawcall;
    m_endDrawcall = endDrawcall;
    m_tile_H = tile_H;
    m_tile_W = tile_W;
    m_block_H = block_H;
    m_block_W = block_W;
    m_tc_h = tc_h;
    m_tc_w = tc_w;
    m_tc_block_dim = tc_block_dim;
    m_wg_size = wg_size;
    m_inShaderBlending = (blendingMode != 0);
    m_inShaderDepth = (depthMode != 0);
    printf("inshader depth = %d\n", m_inShaderDepth);
    m_cptStartFrame = cptStartFrame;
    m_cptEndFrame = cptEndFrame;
    m_cptPeroid = cptPeroid;
    m_skipCpFrames = skipCpFrames;
    m_cptNextFrame = (unsigned) -1;
    m_outdir = outdir;
    
    m_intFolder = m_fbFolder = simout.directory().c_str();
    /*std::string uname = std::tmpnam(nullptr);
    uname = uname.substr(8, uname.size()-9);
    printf("getting file name = %s\n",  uname.c_str());
    char cwd[2048];
    getcwd(cwd, 2048);*/
    m_intFolder+= "gpgpusimShaders"; //+uname;
    m_fbFolder += "gpgpusimFrameDumps"; //+uname;
    //create if not exist
    system(std::string("mkdir -p " + m_intFolder).c_str());
    system(std::string("mkdir -p " + m_fbFolder).c_str());
    //clear older files if any
    system(std::string("rm -f "+ m_intFolder + "/*").c_str());
    system(std::string("rm -f " + m_fbFolder + "/*").c_str());

    vPTXPrfx = m_intFolder+"/vertex_shader";
    fPTXPrfx = m_intFolder+"/fragment_shader";
    fPtxInfoPrfx = m_intFolder+"/shader_ptxinfo";
}

bool renderData_t::useInShaderBlending() const {
    return m_inShaderBlending;
}

void renderData_t::checkExitCond(){
   if(((m_currentFrame== m_endFrame) and (m_drawcall_num > m_endDrawcall)) or (m_currentFrame > m_endFrame)){
      g_gpuMutex.lock();
      exitSimLoop("gem5 exit, end of graphics simulation", 0, curTick(), 0, true);
      g_gpuMutex.unlock();
   }
}

void renderData_t::incCurrentFrame(){
   m_currentFrame++;
   m_drawcall_num = 0;
   checkpoint();
   checkExitCond();
}

bool renderData_t::GPGPUSimActiveFrame() {
   bool isFrame = ((m_currentFrame >= m_startFrame)
          and (m_currentFrame <= m_endFrame) 
          and !checkpointGraphics::SerializeObject.isUnserializingCp());

   return isFrame;
}

bool renderData_t::GPGPUSimSimulationActive() {
   bool isFrame = GPGPUSimActiveFrame();
   bool afterStartDrawcall = ((m_currentFrame== m_startFrame) and (m_drawcall_num >= m_startDrawcall)) or (m_currentFrame > m_startFrame);
   bool beforeEndDrawcall =  ((m_currentFrame== m_endFrame) and (m_drawcall_num <= m_endDrawcall)) or (m_currentFrame < m_endFrame);

   return (isFrame and afterStartDrawcall and beforeEndDrawcall);
}

bool renderData_t::GPGPUSimSkipCpFrames(){
   bool skipCpFrames = (checkpointGraphics::SerializeObject.isUnserializingCp() and m_skipCpFrames);
   return skipCpFrames; 
}

void renderData_t::checkpoint(){
   std::string cptMsg = "graphics checkpoint";
   if(m_cptStartFrame == m_currentFrame){
      CheckPointRequest_t::Request.setCheckPoint(cptMsg);
      if(m_cptPeroid > 0){
         m_cptNextFrame = m_cptStartFrame + m_cptPeroid;
      }
   }

   if((m_cptNextFrame == m_currentFrame) and (m_currentFrame <= m_cptEndFrame)){
      CheckPointRequest_t::Request.setCheckPoint(cptMsg);
      m_cptNextFrame+= m_cptPeroid;
   }
}


void renderData_t::endOfFrame(){
    printf("gpgpusim: end of frame %u\n", getCurrentFrame());
    incCurrentFrame();
}

void renderData_t::finalizeCurrentDraw() {
    printf("gpgpusim: end of drawcall %llu, ", getDrawcallNum());
    if (!GPGPUSimSimulationActive()){
        printf("not simulated!\n");
        incDrawcallNum();
    }
    else {
      printf("simulated!\n");
    }
}

const char* renderData_t::getCurrentShaderId(int shaderType) {
    if (shaderType == VERTEX_PROGRAM)
        return (const char*)(m_drawcall_num * 2);
    if (shaderType == FRAGMENT_PROGRAM)
        return (const char*)(m_drawcall_num * 2 + 1);
    assert(0);
}

void renderData_t::generateVertexCode() {
    //TODO
    assert(0);
    /*bool programFound = false;
    
    std::stringstream fileNo;
    fileNo << getDrawcallNum();

    std::string glslPath =  vGlslPrfx + fileNo.str();
    std::string arbPath =   vARBPrfx + fileNo.str();
    
    std::ofstream vertex_glsl;
    vertex_glsl.open(glslPath);

    if (g_renderData.useDefaultShaders()) {
        vertex_glsl << g_renderData.getDefaultVertexShader();
        programFound = true;
    } else
        for (unsigned shadersCounter = 0; shadersCounter < m_mesaCtx->_Shader->ActiveProgram->NumShaders; shadersCounter++) {
            if (m_mesaCtx->_Shader->ActiveProgram->Shaders[shadersCounter]->Type == GL_VERTEX_SHADER and
                m_mesaCtx->_Shader->ActiveProgram->Shaders[shadersCounter]->Stage == MESA_SHADER_VERTEX) {
                //&& m_mesaCtx->_Shader->ActiveProgram->VertexProgram == m_mesaCtx->VertexProgram._Current)
                vertex_glsl << m_mesaCtx->_Shader->ActiveProgram->Shaders[shadersCounter]->Source;
                programFound = true;
                break;
            }
        }

    vertex_glsl.close();
    assert(programFound);

    std::string command = "cgc -oglsl -profile arbvp1 " + glslPath + " -o " + arbPath;
    system(command.c_str());
    std::string blendingMode = "disabled"; 
    std::string blending = "disabled";
    std::string depth = "disabled";
    std::string depthSize = "Z16";
    std::string depthFunc = std::to_string(m_mesaCtx->Depth.Func);
    command = "arb_to_ptx " + arbPath + " " + getIntFolder() + " " 
            + blendingMode + " " + blending + " " + depth + " " + fileNo.str() + " "
            + getCurrentShaderName(VERTEX_PROGRAM)
            + " " + depthSize + " " + depthFunc;
    DPRINTF(MesaGpgpusim, "Running command: %s\n", command);
    system(command.c_str());*/
}

void renderData_t::generateFragmentCode(DepthSize dbSize){ //, struct softpipe_context *sp) {
    assert(0);
    /*
    bool programFound = false;
    std::stringstream fileNo;
    fileNo << g_renderData.getDrawcallNum();
    std::string glslPath = fGlslPrfx + fileNo.str();
    std::string arbPath = fARBPrfx + fileNo.str();
    std::ofstream fragment_glsl;
    fragment_glsl.open(glslPath);


    if (g_renderData.useDefaultShaders()) {
        fragment_glsl << g_renderData.getDefaultFragmentShader();
        programFound = true;
    } else
        for (unsigned shadersCounter = 0; shadersCounter < m_mesaCtx->_Shader->ActiveProgram->NumShaders; shadersCounter++) {
            if (m_mesaCtx->_Shader->ActiveProgram->Shaders[shadersCounter]->Type == GL_FRAGMENT_SHADER and
                m_mesaCtx->_Shader->ActiveProgram->Shaders[shadersCounter]->Stage == MESA_SHADER_FRAGMENT) {
                    // && m_mesaCtx->_Shader->ActiveProgram->FragmentProgram == m_mesaCtx->FragmentProgram._Current)
                fragment_glsl << m_mesaCtx->_Shader->ActiveProgram->Shaders[shadersCounter]->Source;
                programFound = true;
                break;
            }
        }

    fragment_glsl.close();
    assert(programFound);

    std::string command = "cgc -oglsl -profile arbfp1 " + glslPath + " -o " + arbPath;
    system(command.c_str());
    std::string blendingMode =  "disabled";
    std::string blending = "disabled";
    std::string depth = "disabled";

    std::string depthSize = "Z16";
    if(dbSize == DepthSize::Z32){
       depthSize = "Z32";
    }
    
    if(isBlendingEnabled()){
       blending = "enabled";
        if(useInShaderBlending())
            blendingMode = "inShader";
        else blendingMode = "inZunit";
    }

    if(isDepthTestEnabled() and m_inShaderDepth){
       depth = "enabled";
    }
    
    std::string depthFunc = std::to_string(m_mesaCtx->Depth.Func);

    command = "arb_to_ptx " + arbPath + " " + g_renderData.getIntFolder() + " " 
            + blendingMode + " " + blending + " "
            + depth + " " + fileNo.str() + " "
            + getCurrentShaderName(FRAGMENT_PROGRAM)
            + " " + depthSize + " " + depthFunc;

    DPRINTF(MesaGpgpusim, "Running command: %s\n", command);
    system(command.c_str());*/
}

void renderData_t::addFragmentsQuad(std::vector<fragmentData_t>& quad) {
    assert(GPGPUSimSimulationActive());
    assert(quad.size() == TGSI_QUAD_SIZE);

    for(int i=0; i < quad.size(); i++)
      addFragment(quad[i]);
}

std::string getFile(const char *filename)
{
     std::ifstream in(filename, std::ios::in | std::ios::binary);
     if (in) {
        std::ostringstream contents;
        contents << in.rdbuf();
        in.close();
        return(contents.str());
     }
    panic("Unable to open file: %s\n", filename);
}

void* renderData_t::getShaderFatBin(std::string vertexShaderFile,
                                    std::string fragmentShaderFile){
    const unsigned charArraySize = 200;
    std::string vcode = getFile(vertexShaderFile.c_str());
    std::string fcode = getFile(fragmentShaderFile.c_str());

    std::string vfCode = vcode + "\n\n" + fcode;
    
    __cudaFatCudaBinary* fatBin = new __cudaFatCudaBinary();
    
    char* shaderIdent = new char[charArraySize];
    snprintf(shaderIdent,charArraySize,"shader_%llu", getDrawcallNum());
    fatBin->ident = shaderIdent;
    fatBin->version = 3;
    
    char* computeCap = new char[charArraySize];    
    fatBin->ptx = new __cudaFatPtxEntry[2];
    snprintf(computeCap,charArraySize,"compute_10");
    fatBin->ptx[0].gpuProfileName = computeCap;
    size_t codeSize = vfCode.size() + 1;
    fatBin->ptx[0].ptx = new char[codeSize];
    snprintf(fatBin->ptx[0].ptx, codeSize, "%s", vfCode.c_str());
    fatBin->ptx[1].gpuProfileName = NULL;
    
    return fatBin;
}

std::string renderData_t::getShaderPTXInfo(int usedRegs, std::string functionName) {
    assert(usedRegs >= 0);
    std::stringstream ptxInfo;
    ptxInfo <<"ptxas info    : Compiling entry function '"<<functionName<<"' for 'sm_10' " <<std::endl;
    ptxInfo <<"ptxas info    : Used "<<usedRegs<<" registers, 0 bytes smem"<<std::endl;
    return ptxInfo.str();
}

void renderData_t::writeTexture(byte* data, unsigned size, unsigned texNum, unsigned h, unsigned w, std::string typeEx) {
    //image file for the result buffer, used for testing
    std::ofstream bufferImage;
    std::stringstream ss;
    ss << getFbFolder() << "/frame" << getCurrentFrame() <<
            "_drawcall" << getDrawcallNum() << "_texture"<<texNum<<"_"<<w<<"x"<<h<<
            "_" << m_tcPid << "." << m_tcTid << "." << typeEx;
    bufferImage.open(ss.str(), std::ios::binary | std::ios::out);
    for (int i = 0; i < size; i++) {
        bufferImage << data[i];
    }
    bufferImage.close();
    std::string convertCommand = "convert -depth 8 -size " + std::to_string(w) + "x" + std::to_string(h) 
                                 + " " + ss.str() + " " + ss.str() + ".jpg";
    system(convertCommand.c_str());
    system(std::string("rm " + ss.str()).c_str());
}

void renderData_t::writeDrawBuffer(std::string time, byte * buffer, int bufferSize, unsigned w, unsigned h, std::string extOrder, int depth) {
    //copying the result render buffer to mesa
    bool diffFileNames = true;
    //image file for the result buffer, used for testing
    std::ofstream bufferImage;
    std::stringstream ss;

    if (diffFileNames) ss << getFbFolder()
            << "/gpgpusimBuffer_"+ time +"_frame" << getCurrentFrame() << "_drawcall" << getDrawcallNum()
            << "_" << w << "x" << h << "_" << m_tcPid << "." << m_tcTid << "." << extOrder;
    else ss << getFbFolder() << "gpgpusimBuffer." << extOrder;

    bufferImage.open(ss.str(), std::ios::binary | std::ios::out);

    if(!bufferImage.is_open()){
      printf("Error opening file: %s\n", ss.str().c_str());
      abort();
    }

    for (int i = 0; i < bufferSize; i++) {
        bufferImage << buffer[i];
    }

    bufferImage.close();
    std::string convertCommand = "convert -flip -depth " + std::to_string(depth) + " -size " + std::to_string(w) + "x" + std::to_string(h) 
                                 + " " + ss.str() + " " + ss.str() + ".jpg";
    system(convertCommand.c_str());
    system(std::string("rm " + ss.str()).c_str());
}

unsigned renderData_t::getFramebufferFormat(){
    return m_mesaColorBuffer->InternalFormat;
}

void renderData_t::setPixelSize(){
    /*unsigned justFormat = rb->Format;
    unsigned baseFormat = rb->_BaseFormat;
    unsigned internalFormat = rb->InternalFormat;
    unsigned bufferFormat = rb->_BaseFormat;
    m_fbPixelSize = -1;
    unsigned bf = 0;
    switch(bufferFormat){
      case GL_RGBA:
      case GL_RGBA8:
        m_fbPixelSize = 4;
        bf = GL_RGBA;
        break;
      case GL_RGB8:
      case GL_RGB:
        m_fbPixelSize = 3;
        bf = GL_RGB;
        break;
      default:
        printf("Error: unsupported buffer format %x \n", bufferFormat);
        abort();
    }*/

   unsigned fbFormat = getFramebufferFormat();
   if(fbFormat==GL_RGBA or fbFormat==GL_RGBA8 
         or fbFormat==GL_SRGB8_ALPHA8){
      m_fbPixelSizeSim = 4;
   } else if(fbFormat==GL_RGB8 or fbFormat==GL_RGB){
      m_fbPixelSizeSim = 4;
   } else assert(0);
}

unsigned renderData_t::getPixelSizeSim(){
   return m_fbPixelSizeSim;
}

uint64_t renderData_t::getFramebufferFragmentAddr(uint64_t x, uint64_t y, uint64_t size){
  uint64_t buffWidthByte = size*m_bufferWidth;
  x = x%m_bufferWidth; //FIXME
  y = y%m_bufferHeight; //FIXME
  int64_t fbAddr = ((uint64_t) m_deviceData);
  fbAddr += (m_bufferHeight - y -1)*buffWidthByte + (x*size);
  assert(fbAddr >= (uint64_t) m_deviceData);
  assert(fbAddr < ((uint64_t) m_deviceData + m_colorBufferByteSize));
  return fbAddr;
} 

byte* renderData_t::setRenderBuffer(){
    //gl_renderbuffer *rb = m_mesaCtx->DrawBuffer->_ColorDrawBuffers[0];
    gl_renderbuffer *rb = m_mesaCtx->DrawBuffer->_ColorReadBuffer;
    m_mesaColorBuffer = rb;
    m_bufferWidth = rb->Width;
    m_bufferHeight = rb->Height;
    m_bufferWidth = m_mesaCtx->DrawBuffer->Width;
    m_bufferHeight = m_mesaCtx->DrawBuffer->Height;
    setPixelSize();
    m_colorBufferByteSize = m_bufferHeight * m_bufferWidth * m_fbPixelSizeSim;
    DPRINTF(MesaGpgpusim, "gpgpusim-graphics: fb height=%d width=%d\n", m_bufferHeight, m_bufferWidth);

    byte * tempBuffer2 = new byte [m_colorBufferByteSize];
    byte* renderBuf;
    int rbStride;
    m_mesaCtx->Driver.MapRenderbuffer_base(m_mesaCtx, m_mesaColorBuffer,
                                      0, 0, m_bufferWidth, m_bufferHeight,
                                      GL_MAP_READ_BIT,
                                      &renderBuf, &rbStride);

      unsigned pixelSizeMesa = std::abs(rbStride)/m_bufferWidth;
      byte* tempBufferEnd = tempBuffer2 + m_colorBufferByteSize;
      for(int h=0; h < m_bufferHeight; h++)
        for(int w=0; w< m_bufferWidth; w++){
          int srcPixel = ((m_bufferHeight - h - 1) * rbStride)
              + (w * pixelSizeMesa);
          int dstPixel = ((m_bufferHeight - h) * m_bufferWidth * m_fbPixelSizeSim*-1)
              + (w * m_fbPixelSizeSim);
          tempBufferEnd[dstPixel + 0] = renderBuf[srcPixel + 0];
          tempBufferEnd[dstPixel + 1] = renderBuf[srcPixel + 1];
          tempBufferEnd[dstPixel + 2] = renderBuf[srcPixel + 2];
          if(pixelSizeMesa > 3){
             tempBufferEnd[dstPixel + 3] = renderBuf[srcPixel + 3];
          } else {
             assert(m_fbPixelSizeSim > 3);
             tempBufferEnd[dstPixel + 3] = 0xff;
          }
        }

      m_mesaCtx->Driver.UnmapRenderbuffer_base(m_mesaCtx, m_mesaColorBuffer);
      return tempBuffer2;
}


byte* renderData_t::setDepthBuffer(){
    DepthSize activeDepthSize;
    DepthSize trueDepthSize;
    //gl_renderbuffer *rb = m_mesaCtx->DrawBuffer->_DepthBuffer;
    gl_renderbuffer *rb = m_mesaCtx->ReadBuffer->Attachment[BUFFER_DEPTH].Renderbuffer;

    if(isDepthTestEnabled()){
       if(rb->Format==MESA_FORMAT_Z_UNORM32
             or rb->Format==MESA_FORMAT_Z24_UNORM_S8_UINT 
             or rb->Format==MESA_FORMAT_Z24_UNORM_X8_UINT){
          activeDepthSize = trueDepthSize = DepthSize::Z32;
       } else if(rb->Format==MESA_FORMAT_Z_UNORM16){
          if(m_inShaderDepth){
             //in-shader depth test uses atomics that only support 32 bit 
             activeDepthSize = DepthSize::Z32;
             trueDepthSize = DepthSize::Z16;
          } else {
             activeDepthSize = trueDepthSize = DepthSize::Z16;
          }
       } else {
          printf("GPGPUSIM: Unsupported depth format %x \n", rb->Format);
          abort();
       }
    }

    unsigned buffSize = rb->Height * rb->Width;
    m_depthBufferSize = buffSize * sizeof (byte)* ((uint32_t) activeDepthSize);
    m_depthBufferWidth = rb->Width;
    m_depthBufferHeight = rb->Height;
    m_depthSize = activeDepthSize;
    m_mesaDepthSize = trueDepthSize;
    m_mesaDepthBuffer = rb;
    unsigned depthSize = (unsigned) m_depthSize;
    uint32_t mesaDepthBufferSize = buffSize * depthSize;

    assert(m_depthBufferWidth  == m_bufferWidth);
    assert(m_depthBufferHeight == m_bufferHeight);

    //assert our assumptions
    assert(DepthSize::Z32==m_depthSize);
    assert(DepthSize::Z32==m_mesaDepthSize);
    assert(rb->Format==MESA_FORMAT_Z24_UNORM_S8_UINT
          or rb->Format==MESA_FORMAT_Z24_UNORM_X8_UINT);
    //assert(rb->InternalFormat == GL_DEPTH24_STENCIL8);
    //assert(rb->_BaseFormat == GL_DEPTH_STENCIL);
    DPRINTF(MesaGpgpusim, 
          "gpgpusim-graphics: fb height=%d width=%d\n",m_bufferHeight, m_bufferWidth);
    byte *tempBuffer  = new byte [mesaDepthBufferSize];
    //std::memset(tempBuffer, 0, mesaDepthBufferSize);
    byte* renderBuf;
    int rbStride;
    m_mesaCtx->Driver.MapRenderbuffer_base(m_mesaCtx, rb,
                                      0, 0, m_depthBufferWidth,
                                      m_depthBufferHeight,
                                      GL_MAP_READ_BIT,
                                      &renderBuf, &rbStride);

    //_mesa_unpack_ubyte_stencil_row(rb->Format, 3, NULL, NULL);

    byte* tempBufferEnd = tempBuffer + mesaDepthBufferSize;
    for(int h=0; h < m_bufferHeight; h++)
       for(int w=0; w< m_bufferWidth; w++){
          int srcPixel = ((m_bufferHeight - h - 1) * rbStride)
             + (w * depthSize);
          int dstPixel = ((m_bufferHeight - h) * m_bufferWidth * depthSize*-1)
             + (w * depthSize);

          tempBufferEnd[dstPixel + 0] = renderBuf[srcPixel + 0];
          tempBufferEnd[dstPixel + 1] = renderBuf[srcPixel + 1];
          if(depthSize > 2)
             tempBufferEnd[dstPixel + 2] = renderBuf[srcPixel + 2];
          if(depthSize > 3)
             tempBufferEnd[dstPixel + 3] = 0;

       }
    return tempBuffer;
}

void renderData_t::setMesaCtx(struct gl_context * ctx){
   m_mesaCtx=ctx;
   assert(m_currentRenderBufferBytes == NULL);
   m_currentRenderBufferBytes = setRenderBuffer();
}


void renderData_t::registerPtxCode(){
    std::string frame_drawcall = std::to_string(m_currentFrame) + "_" + std::to_string(m_drawcall_num);
    std::string vertexPTXFile = vPTXPrfx +frame_drawcall+".ptx";
    std::string fragmentPTXFile = fPTXPrfx +frame_drawcall+".ptx"; 
    void* cudaFatBin = getShaderFatBin(vertexPTXFile, fragmentPTXFile);

    std::string vertexPtxInfo = getShaderPTXInfo(m_usedVertShaderRegs, getCurrentShaderName(VERTEX_PROGRAM));
    std::string fragmentPtxInfo = getShaderPTXInfo(m_usedFragShaderRegs, getCurrentShaderName(FRAGMENT_PROGRAM));

    std::string ptxInfoFileName = fPtxInfoPrfx +
        std::to_string(m_currentFrame) + "_" + std::to_string(getDrawcallNum());
    std::ofstream ptxInfoFile(ptxInfoFileName.c_str());
    assert(ptxInfoFile.is_open());
    ptxInfoFile<< vertexPtxInfo + fragmentPtxInfo; 
    ptxInfoFile.close();

    void ** fatCubinHandle = graphicsRegisterFatBinary(cudaFatBin, ptxInfoFileName.c_str(), &m_sShading_info.allocAddr);

    //assert(m_sShading_info.allocAddr != NULL); //we always have some constants in the shaders
    lastFatCubin = (__cudaFatCudaBinary*)cudaFatBin;
    lastFatCubinHandle = fatCubinHandle;

    graphicsRegisterFunction(fatCubinHandle,
            getCurrentShaderId(VERTEX_PROGRAM),
            (char*)getCurrentShaderName(VERTEX_PROGRAM).c_str(),
            getCurrentShaderName(VERTEX_PROGRAM).c_str(),
            -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0);

    graphicsRegisterFunction(fatCubinHandle,
            getCurrentShaderId(FRAGMENT_PROGRAM),
            (char*)getCurrentShaderName(FRAGMENT_PROGRAM).c_str(),
            getCurrentShaderName(FRAGMENT_PROGRAM).c_str(),
            -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0);
}

void renderData_t::initializeCurrentDraw(struct tgsi_exec_machine* tmachine, void* sp, void* mapped_indices) {
    g_gpuMutex.lock();
    assert(getDeviceData() == NULL);
    m_deviceData = (byte*)0xDEADBEEF; //flags that a render operation is active
    m_tmachine = tmachine;
    m_sp = sp;
    m_mapped_indices = mapped_indices;
    gl_context * ctx = m_mesaCtx;
    if (!GPGPUSimSimulationActive()) {
        std::cerr << "gpgpusim-graphics: Error, initializeCurrentDraw called when simulation is not active " << std::endl;
        exit(-1);
    }

    DPRINTF(MesaGpgpusim, "starting drawcall at tick = %ld\n", curTick());
    g_startTick = curTick();

    //pipe_context* pcontext = (softpipe_context*) st->cso_context;

    DPRINTF(MesaGpgpusim, "initializing a draw call \n");


    setAllTextures(lastFatCubinHandle);

    struct softpipe_context *softpipe = (struct softpipe_context *) m_sp;
    const void** constBufs = softpipe->mapped_constants[PIPE_SHADER_FRAGMENT];
    const unsigned* constBufSizes = softpipe->const_buffer_size[PIPE_SHADER_FRAGMENT];

    int ci = 0;
    while(constBufs[ci]){
      consts.push_back(std::vector<ch4_t>());
      assert(constBufSizes[ci]%TGSI_QUAD_SIZE ==  0);
      int constCount = constBufSizes[ci]/TGSI_QUAD_SIZE;
      for(int j=0; j < constCount; j++){
        ch4_t elm;
        for(int ch=0; ch < TGSI_NUM_CHANNELS; ch++){
          const uint *buf = (const uint*) constBufs[ci];
          const int pos = j * TGSI_NUM_CHANNELS + ch;
          elm[ch] = buf[pos];
        }
        consts.back().push_back(elm);
      }
      ci++;
    }



    unsigned fragmentsPerTile = m_tile_H * m_tile_W;
    m_wTiles = (m_bufferWidth + m_tile_W -1)/ m_tile_W;
    m_hTiles = (m_bufferHeight + m_tile_H -1)/ m_tile_H;
    assert(0 == ((m_bufferHeight* m_bufferWidth) % fragmentsPerTile));
    m_tilesCount = m_wTiles * m_hTiles;

    m_depthBuffer = NULL;
    if(isDepthTestEnabled()){
        m_depthBuffer = setDepthBuffer();
        graphicsMalloc((void**) &m_deviceData, m_colorBufferByteSize + m_depthBufferSize); 
        modeMemcpy(m_deviceData + m_colorBufferByteSize,
                 m_depthBuffer, m_depthBufferSize, graphicsMemcpyHostToSim);
        setHizTiles(RasterDirection::HorizontalRaster);
    } else {
        graphicsMalloc((void**) &m_deviceData, m_colorBufferByteSize);
    }

    modeMemcpy(m_deviceData, m_currentRenderBufferBytes, 
          getColorBufferByteSize(), graphicsMemcpyHostToSim);
    assert(m_fbPixelSizeSim == 4);
    writeDrawBuffer("pre", m_currentRenderBufferBytes,  m_colorBufferByteSize,
          m_bufferWidth, m_bufferHeight, "bgra", 8);

    delete [] m_currentRenderBufferBytes;
    m_currentRenderBufferBytes = NULL;

    if(m_depthBuffer!=NULL) {
       writeDrawBuffer("pre_depth", m_depthBuffer,  m_depthBufferSize, m_bufferWidth, m_bufferHeight, "gray", 8*(int)m_mesaDepthSize);
    }

    if(isBlendingEnabled()){
      DPRINTF(MesaGpgpusim, "blending enabled\n");
    } else {
      DPRINTF(MesaGpgpusim, "blending disabled\n");
    }
}

void renderData_t::addTexelFetch(int x, int y, int level){
   texelInfo_t* ti = &m_textureInfo[m_currSamplingUnit];
   uint64_t texelAddr = ti->getTexelAddr(x, y, level);
   m_texelFetches.push_back(texelAddr);
}

std::vector<uint64_t> renderData_t::fetchTexels(
      int modifier, int unit, int dim,
      float* coords,
      int num_coords, float* dst, int num_dst, 
      unsigned utid, void* stream,
      texModifier tmodifier){
  m_currSamplingUnit = unit;
  texelInfo_t* ti = &m_textureInfo[m_currSamplingUnit];

  unsigned  quadIdx = getFragmentData(utid, utid, QUAD_INDEX, -1, -1, -1, stream).u32;
  if(tmodifier==texModifier::NONE) {
    //FIXME: use txf
    //mesaFetchTxf(m_tmachine, modifier, unit, dim, coords, num_coords , dst, num_dst, quadIdx);
    modifier = 0;
    mesaFetchTexture(m_tmachine, modifier, unit, 1 /*sampler*/, dim, coords, num_coords , dst, num_dst, quadIdx);
  } else if(tmodifier==texModifier::PROJECTED){
    modifier = 1; //TEX_MODIFIER_PROJECTED
    mesaFetchTexture(m_tmachine, modifier, unit, 1/*sampler*/, dim, coords, num_coords , dst, num_dst, quadIdx);
  } else if(tmodifier==texModifier::LOD_BIAS){
    modifier = 2; //TEX_MODIFIER_LOD_BIAS
    mesaFetchTexture(m_tmachine, modifier, unit, 1/*sampler*/, dim, coords, num_coords , dst, num_dst, quadIdx);
  } else {
    modifier = 0; //TEX_MODIFIER_NONE
    mesaFetchTexture(m_tmachine, modifier, unit, 1/*sampler*/, dim, coords, num_coords , dst, num_dst, quadIdx);
  }
  std::vector<uint64_t> texelFetches;
  //fetches for all quad fragments are included, filter out relevant ones based
  //on the quadIdx
  assert(m_texelFetches.size() >= TGSI_QUAD_SIZE); //at least 1 texel per fragment
  int texelsPerFragment = (m_texelFetches.size()+TGSI_QUAD_SIZE-1)/TGSI_QUAD_SIZE;
  int startTexel = texelsPerFragment*quadIdx;
  int endTexel = (texelsPerFragment*(quadIdx +1)) - 1;
  int curTexel = 0;
  std::unordered_set<uint64_t> checker;
  for(auto it= m_texelFetches.begin(); it != m_texelFetches.end(); it++){
    if(curTexel >= startTexel and curTexel <= endTexel){
      if(checker.find(*it) == checker.end()){
        texelFetches.push_back(*it);
        checker.insert(*it);
      }
    }
    curTexel++;
  }
  m_texelFetches.clear();
  return texelFetches;
}


unsigned renderData_t::getTexelSize(int samplingUnit){
  const struct sp_tgsi_sampler *sp_samp = (const struct sp_tgsi_sampler*) (m_tmachine->Sampler);
  const struct sp_sampler_view* sp_view;
  sp_view = &sp_samp->sp_sview[samplingUnit];
  const struct pipe_resource* tex = sp_view->base.texture;
  GLenum datatype;
  GLuint comps;
  _mesa_uncompressed_format_to_type_and_comps((mesa_format)tex->format, &datatype, &comps);
  switch(datatype){
    case GL_BYTE:
    case GL_UNSIGNED_BYTE:
      return 1;
      break;
    case GL_SHORT:
    case GL_UNSIGNED_SHORT:
    case GL_UNSIGNED_SHORT_1_5_5_5_REV:
    case GL_HALF_FLOAT:
      return 2;
      break;
    case GL_FLOAT:
    case GL_INT:
    case GL_UNSIGNED_INT:
      return 4;
      break;

    case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
    case GL_UNSIGNED_BYTE_2_3_3_REV:
    case GL_UNSIGNED_BYTE_3_3_2:
    case GL_UNSIGNED_INT_10_10_10_2:
    case GL_UNSIGNED_INT_10F_11F_11F_REV:
    case GL_UNSIGNED_INT_2_10_10_10_REV:
    case GL_UNSIGNED_INT_24_8_MESA:
    case GL_UNSIGNED_INT_5_9_9_9_REV:
    case GL_UNSIGNED_INT_8_24_REV_MESA:
    case GL_UNSIGNED_SHORT_4_4_4_4:
    case GL_UNSIGNED_SHORT_5_5_5_1:
    case GL_UNSIGNED_SHORT_5_6_5:
    case MESA_UNSIGNED_BYTE_4_4:
    default:
      assert(0 and "Error: unsupported texture format");
      break;
  }
  return -1;
}

void renderData_t::setAllTextures(void ** fatCubinHandle){
    const struct sp_tgsi_sampler *sp_samp = (const struct sp_tgsi_sampler*) (m_tmachine->Sampler);
    const struct sp_sampler_view* sp_view;
    int tidx = 0;
    while(true){
      sp_view = &sp_samp->sp_sview[tidx];
      if(sp_view->base.format == PIPE_FORMAT_NONE)
        break;
      const struct pipe_resource* tex = sp_view->base.texture;
      printf("texture %d, format = %d, size= %dx%d, last_level=%d\n", 
            tidx, tex->format, tex->width0, tex->height0, tex->last_level);
      /*
      TODO: support array formats
      check sp_texture.c:softpipe_transfer_map
      check sp_tex_tile_cache.c:sp_find_cached_tile_tex
       */
      assert(tex->target != PIPE_TEXTURE_CUBE
            and tex->target != PIPE_TEXTURE_1D_ARRAY 
            and tex->target != PIPE_TEXTURE_2D_ARRAY 
            and tex->target != PIPE_TEXTURE_CUBE_ARRAY);
      unsigned curLevel = 0;
      std::vector<texelInfo_t::mipmapInfo_t> mmOffsets;
      unsigned width, height;
      uint64_t mmTexels = 0;
      while(true){
         width = u_minify(tex->width0, curLevel);
         height = u_minify(tex->height0, curLevel);
         mmOffsets.push_back(texelInfo_t::mipmapInfo_t(mmTexels, width, height));
         mmTexels+= width*height;
         if(tex->last_level == curLevel)
            break;
         curLevel++;
      }
      unsigned texelSize = getTexelSize(tidx);
      unsigned texturesSize =  texelSize * mmTexels;
      void* texturesBuffer;
      graphicsMalloc((void**) &texturesBuffer, texturesSize);
      m_textureInfo.push_back(
            texelInfo_t((uint64_t)texturesBuffer, texelSize, 
               mmOffsets, tex));
      printf("adding textures to buff %lx\n", (uint64_t)texturesBuffer);
      tidx++;
    }
}


/*
GLboolean renderData_t::doVertexShading(GLvector4f ** inputParams, vp_stage_data * stage){
    if (!GPGPUSimSimulationActive()) {
        std::cerr<<"gpgpusim-graphics: Error, doVertexShading called when simulation is not active "<<std::endl;
        exit(-1);
    }

    vertexStageData = stage;
    setAllTextures(lastFatCubinHandle);
    copyStateData(lastFatCubinHandle);
    
    //we want to map the attribs names to position in ARB
    std::string arbPath = vARBPrfx + std::to_string(getDrawcallNum());
    std::map<std::string, int> generalAttribPosition;
    std::ifstream arb_file;
    arb_file.open(arbPath);
    assert(arb_file.is_open());
    while (arb_file.good()) {
        std::string arbLine;
        getline(arb_file, arbLine);
        std::string inAttrib("$vin.ATTR");
        size_t found = arbLine.find(inAttrib);
        if (found != std::string::npos) {
            int attribPos = arbLine[found + inAttrib.size()] - '0';
            size_t secondSpace = arbLine.find(" ", arbLine.find(" ") + 1);
            size_t thirdSpace = arbLine.find(" ", secondSpace + 1);
            std::string varName = arbLine.substr(secondSpace + 1, thirdSpace - secondSpace - 1);
            generalAttribPosition[varName] = attribPos;
        }
    }

    arb_file.close();
    
    TNLcontext *tnl = TNL_CONTEXT(m_mesaCtx);
    vertex_buffer *VB = &tnl->vb;
    unsigned vertsCount = VB->Count;
    printf("starting vertex shading: vertices = %d\n", vertsCount);
    const unsigned verticesAttribsFloatsCount = vertsCount * VERT_ATTRIB_MAX * 4;
    float hostVertsAttribs[verticesAttribsFloatsCount];

    // the vertex array case
    for (unsigned i = 0; i < vertsCount; i++) {
        for (unsigned attr = 0; attr < VERT_ATTRIB_MAX; attr++) {
            if (m_mesaCtx->VertexProgram._Current->Base.InputsRead & (1 << attr)) {
                const GLubyte *ptr = (const GLubyte*) inputParams[attr]->data;
                const GLuint size = inputParams[attr]->size;
                const GLuint stride = inputParams[attr]->stride;
                const GLfloat *data = (GLfloat *) (ptr + stride * i);
                float * place;
              
                if (attr < VERT_ATTRIB_GENERIC0){
                    //if not generic copy in the corresponding location
                    place = &hostVertsAttribs[(i * VERT_ATTRIB_MAX * 4) + (attr * 4)];
                } else {
                    //else if general attrib we should map the locations from mesa to the nvidia arb code
                    int mesaNumber = attr - VERT_ATTRIB_GENERIC0;
                    std::string mesaName;
                    //if(m_mesaCtx->VertexProgram._Current->Base.NumAttributes == 0) break;
                    for (int l = 0; l < m_mesaCtx->VertexProgram._Current->Base.Attributes->NumParameters; l++)
                        if (m_mesaCtx->VertexProgram._Current->Base.Attributes->Parameters[l].StateIndexes[0] == mesaNumber) {
                            mesaName = m_mesaCtx->VertexProgram._Current->Base.Attributes->Parameters[l].Name;
                            break;
                        }

                    int arbNumber = generalAttribPosition[mesaName];
                    place = &hostVertsAttribs[i * VERT_ATTRIB_MAX * 4 + (VERT_ATTRIB_GENERIC0 + arbNumber)*4];
                }
                COPY_CLEAN_4V(place, size, data);
            }
        }
    }

    //DPRINTF(MesaGpgpusim, "vertex shader input attribs, verticesAttribsFloatsCount = %d \n", verticesAttribsFloatsCount);
    //for (int vert = 0; vert < vertsCount; vert++) {
        //for (int attrib = 0; attrib < VERT_ATTRIB_MAX; attrib++) {
         //   for (int i = 0; i < 4; i++) {
                //int index = (vert * VERT_ATTRIB_MAX * 4) + (attrib * 4) + i;
                //DPRINTF(MesaGpgpusim, "hostVertsAttribs[%d][%s][%d] = %f \n", vert, VERT_ATTRIB_NAMES[attrib], i, hostVertsAttribs[index]);
            //}
        //}
    //}

    assert(m_sShading_info.deviceVertsAttribs == NULL);
    graphicsMalloc((void**) &m_sShading_info.deviceVertsAttribs, verticesAttribsFloatsCount * sizeof (float));
    graphicsMemcpy(m_sShading_info.deviceVertsAttribs, hostVertsAttribs, verticesAttribsFloatsCount * sizeof (float), graphicsMemcpyHostToSim);
    
    m_sShading_info.primCountMap = new unsigned [VB->PrimitiveCount];
    memset(m_sShading_info.primCountMap, 0, sizeof(unsigned)*VB->PrimitiveCount);
    
    
    m_sShading_info.render_init = false;
    assert(m_sShading_info.vertCodeAddr == NULL);

    unsigned threadsPerBlock = m_wg_size; 
    unsigned numberOfBlocks = (vertsCount + threadsPerBlock -1) / threadsPerBlock;
    assert(graphicsConfigureCall(numberOfBlocks, threadsPerBlock, 0, 0) == cudaSuccess);
    assert(graphicsSetupArgument((void*)&m_sShading_info.deviceVertsAttribs, sizeof(float*), 0) == cudaSuccess);
    assert(graphicsLaunch(getCurrentShaderId(VERTEX_PROGRAM), &m_sShading_info.vertCodeAddr) == cudaSuccess); 
    
    m_sShading_info.launched_threads_frags = numberOfBlocks * threadsPerBlock;
    printf("launched vertex threads = %d\n", m_sShading_info.launched_threads_frags);
    m_sShading_info.completed_threads_frags = 0;
    m_sShading_info.currentPass = stage_shading_info_t::GraphicsPass::Vertex;
   
    printf("do vertex inc ready sockets\n");fflush(stdout);
    //SocketStream::incReadySockets(-1, true);
    //waiting the fragment shader to finish
    printf("first vertex lock\n");fflush(stdout);
    vertexFragmentLock.lock();
    printf("second vertex lock\n");fflush(stdout);
    vertexFragmentLock.lock(); //wait till end of fragment shader to unlock
    printf("unlock vertex lock\n");fflush(stdout);
    vertexFragmentLock.unlock();
    fflush(stdout);
    //int dum = 0;
    //while(m_sShading_info.currentPass != stage_shading_info_t::GraphicsPass::NONE){
       //dum++;
    //}
    endDrawCall();
    return GL_TRUE;
}
*/

void renderData_t::setHizTiles(RasterDirection rasterDir) {
   const unsigned frameDim = m_bufferHeight * m_bufferWidth;
   std::vector<uint64_t> depthValues(frameDim);

   if(m_depthSize == DepthSize::Z16) {
      uint16_t* p = (uint16_t*) m_depthBuffer;
      for(unsigned i =0; i< frameDim; i++)
         depthValues[i] = p[i];
   } else if(m_depthSize == DepthSize::Z32){
      uint32_t* p = (uint32_t*) m_depthBuffer;
      for(unsigned i =0; i< frameDim; i++)
         depthValues[i] = p[i];
   } else assert(0);

   std::vector<bool> touchedTiles(m_tilesCount, false);
   m_hizBuff.setSize(m_tilesCount);
   //assert((m_bufferWidth%m_tile_W) == 0);
   //assert((m_bufferHeight%m_tile_H) == 0);
   const unsigned tileRow = (m_bufferWidth+m_tile_W-1) / m_tile_W;
   for(unsigned i=0; i < depthValues.size(); i++){
      unsigned tileIdx = -1;
      unsigned xPos = i%m_bufferWidth;
      unsigned yPos = i/m_bufferWidth;
      unsigned tileXCoord = xPos/m_tile_W;
      unsigned tileYCoord = yPos/m_tile_H;
      if(rasterDir == RasterDirection::HorizontalRaster){
         tileIdx = tileYCoord*tileRow + tileXCoord;
      } else {
         assert(0);
      } 
      m_hizBuff.setDepth(tileIdx, tileXCoord, tileYCoord, depthValues[i]);
   }
}

bool renderData_t::testHiz(RasterTile* tile){
   if(not isDepthTestEnabled()){
      tile->setSkipFineDepth();
      return true;
   }
   unsigned const tileId = tile->m_tilePos;
   assert(tile->xCoord == m_hizBuff.m_hizEntries[tileId].xCoord);
   assert(tile->yCoord == m_hizBuff.m_hizEntries[tileId].yCoord);

   if(m_mesaCtx->Depth.Func==GL_NOTEQUAL 
         or m_mesaCtx->Depth.Func==GL_EQUAL){ 
      warn_once("Unsupported depth test (GL_NOTEQUAL or GL_EQUAL), skipping HiZ\n");
      return true;
   } else if(depthTest(m_hizBuff.m_hizEntries[tileId].frontDepth,
            tile->backDepth())){
      m_hizBuff.m_hizEntries[tileId].frontDepth = tile->frontDepth();
      if(tile->fullyCovered()){
         m_hizBuff.m_hizEntries[tileId].backDepth = tile->backDepth();
      }
      tile->setSkipFineDepth();
      return true;
   } else if(depthTest(m_hizBuff.m_hizEntries[tileId].backDepth, tile->frontDepth())){
      tile->setHizThresh(m_hizBuff.m_hizEntries[tileId].backDepth);
      return true;
   } 
   return false;
}

bool renderData_t::depthTest(uint64_t oldDepthVal, uint64_t newDepthVal){
   bool returnVal = false;
   switch(m_mesaCtx->Depth.Func){
      case GL_LESS: 
         if(newDepthVal < oldDepthVal)
            returnVal = true;
         break;
      case GL_LEQUAL:
         if(newDepthVal <= oldDepthVal)
            returnVal = true;
         break;
      case GL_GEQUAL: 
         if(newDepthVal >= oldDepthVal)
            returnVal = true;
         break;
      case GL_GREATER: 
         if(newDepthVal > oldDepthVal)
            returnVal = true;
         break;
      case GL_NOTEQUAL:
         if(newDepthVal != oldDepthVal)
            returnVal = true;
         break;
      case GL_EQUAL:
         if(newDepthVal == oldDepthVal)
            returnVal = true;
         break;
      case GL_NEVER:
         returnVal = false;
         break;
      case GL_ALWAYS:
         returnVal = true;
      default: 
         panic("Unsupported depth function %x\n", m_mesaCtx->Depth.Func);
   }
   return returnVal;
}

bool renderData_t::runNextPrim(){
   CudaGPU* cudaGPU = CudaGPU::getCudaGPU(g_active_device);
   gpgpu_sim* gpu =  cudaGPU->getTheGPU();
   simt_core_cluster** simt_clusters = gpu->getSIMTCluster();

   if(m_sShading_info.current_prim == drawPrimitives.size()) 
      return false;

   unsigned prim = m_sShading_info.current_prim++;
   if(drawPrimitives[prim].size() == 0){
      return true;
   }
   printf("adding prim %d from %d primitives\n", prim, drawPrimitives.size());
   drawPrimitives[prim].sortFragmentsInTiles(
         m_bufferHeight, m_bufferWidth, 
         m_tile_H, m_tile_W, 
         m_hTiles, m_wTiles,
         m_tilesCount,
         m_block_H, m_block_W, 
         RasterDirection::HorizontalRaster, 
         m_tc_h, m_tc_w,
         m_tc_block_dim,
         m_numClusters);
   for(unsigned clusterId=0; clusterId < m_numClusters; clusterId++){
      bool res = simt_clusters[clusterId]->getGraphicsPipeline()->add_primitive(&drawPrimitives[prim]);
      assert(res);
   }

   m_sShading_info.sent_simt_prims += m_numClusters;
   return true;
}

unsigned int renderData_t::startShading() {
   registerPtxCode();

   m_sShading_info.completed_threads_verts = 0;
   m_sShading_info.launched_threads_verts = 0;
   m_sShading_info.completed_threads_frags = 0;
   m_sShading_info.launched_threads_frags = 0;

   CudaGPU* cudaGPU = CudaGPU::getCudaGPU(g_active_device);
   gpgpu_sim* gpu =  cudaGPU->getTheGPU();
   m_numClusters = gpu->get_config().num_cluster(); // TODO: move me
   m_coresPerCluster= gpu->get_config().num_cores_per_cluster(); //TODO: move me
   assert(m_sShading_info.fragCodeAddr == NULL);

   while(runNextPrim()){}
   cudaGPU->activateGPU();

   /*if(m_inShaderDepth or not(isDepthTestEnabled())){
      //now sorting them in raster order
      sortFragmentsInRasterOrder(getTileH(), getTileW(),getBlockH(), getBlockW(), HorizontalRaster); //BlockedHorizontal);
      noDepthFragmentShading();
   } else {
      runEarlyZ(cudaGPU, getTileH(), getTileW(),getBlockH(), getBlockW(), HorizontalRaster, num_clusters); // BlockedHorizontal);
   }*/
   g_gpuMutex.unlock();
}

unsigned int renderData_t::noDepthFragmentShading() {
    /*if(!GPGPUSimSimulationActive()){
        std::cerr<<"Error: noDepthFragmentShading called when simulation is not active "<<std::endl;
        exit(-1);
    }
   
    unsigned totalFragsCount = 0;
    for(int prim =0; prim < drawPrimitives.size(); prim++){
       totalFragsCount+= drawPrimitives[prim].size();
    }

    if (totalFragsCount == 0){
      endFragmentShading();
      m_flagEndFragmentShader = false;
      return 0;
    }

    g_totalFrags+= totalFragsCount;

    m_sShading_info.completed_threads_frags = 0;
    assert(m_sShading_info.fragCodeAddr == NULL);
    for(int prim =0; prim < drawPrimitives.size(); prim++){
       unsigned fragmentsCount =  drawPrimitives[prim].size();
       if(fragmentsCount == 0) continue;
       unsigned threadsPerBlock = 512;
       unsigned numberOfBlocks = (fragmentsCount + threadsPerBlock -1) / threadsPerBlock;
       m_sShading_info.launched_threads_frags += numberOfBlocks*threadsPerBlock;

       m_sShading_info.cudaStreams.push_back(cudaStream_t());
       graphicsStreamCreate(&m_sShading_info.cudaStreams.back());

       DPRINTF(MesaGpgpusim, "starting fragment shader, fragments = %d on stream 0x%lx\n", fragmentsCount,  m_sShading_info.cudaStreams.back());
       printf("starting fragment shader, fragments = %d, total=%d\n", fragmentsCount, m_sShading_info.launched_threads_frags);

       tileStream_t map;
       map.primId =  prim;
       map.pendingFrags = numberOfBlocks*threadsPerBlock;
       m_sShading_info.cudaStreamTiles[(uint64_t)(m_sShading_info.cudaStreams.back())] = map;

       assert(numberOfBlocks);
       byte* arg= getDeviceData() + getColorBufferByteSize();

       assert( graphicsConfigureCall(numberOfBlocks, threadsPerBlock, 0, m_sShading_info.cudaStreams.back())
          == cudaSuccess);
       assert(graphicsSetupArgument((void*) &arg, sizeof (byte*), 0) == cudaSuccess);
       assert(graphicsLaunch(getCurrentShaderId(FRAGMENT_PROGRAM), &m_sShading_info.fragCodeAddr) == cudaSuccess);
    }

    assert(m_sShading_info.fragCodeAddr != NULL);

    m_sShading_info.currentPass = stage_shading_info_t::GraphicsPass::Fragment;
    return m_sShading_info.launched_threads_frags;*/
}

void renderData_t::putDataOnColorBuffer() {
    //copying the result render buffer to mesa
    byte * tempBuffer = new byte [getColorBufferByteSize()];
    modeMemcpy(tempBuffer, m_deviceData, getColorBufferByteSize(), graphicsMemcpySimToHost);

    assert(m_fbPixelSizeSim == 4);
    writeDrawBuffer("post", (byte*)tempBuffer, getColorBufferByteSize(), m_bufferWidth, m_bufferHeight, "bgra", 8);

    byte* renderBuf;
    int rbStride;
    m_mesaCtx->Driver.MapRenderbuffer_base(m_mesaCtx, m_mesaColorBuffer,
                                      0, 0, m_bufferWidth, m_bufferHeight,
                                      GL_MAP_WRITE_BIT
                                      | GL_MAP_INVALIDATE_RANGE_BIT,
                                      //| GL_MAP_INVALIDATE_BUFFER_BIT
                                      &renderBuf, &rbStride);
      assert(std::abs(rbStride)%m_bufferWidth == 0);
      unsigned pixelSizeMesa = std::abs(rbStride)/m_bufferWidth;
      byte* tempBufferEnd = tempBuffer + m_colorBufferByteSize;
      for(int h=0; h < m_bufferHeight; h++)
        for(int w=0; w< m_bufferWidth; w++){
          int srcPixel = ((m_bufferHeight - h - 1) * rbStride) + (w * pixelSizeMesa);
          int dstPixel = ((m_bufferHeight - h) * m_bufferWidth * m_fbPixelSizeSim*-1)
              + (w * m_fbPixelSizeSim);
          renderBuf[srcPixel + 0] = tempBufferEnd[dstPixel + 0];
          renderBuf[srcPixel + 1] = tempBufferEnd[dstPixel + 1];
          renderBuf[srcPixel + 2] = tempBufferEnd[dstPixel + 2];
          if(pixelSizeMesa > 3){
             assert(m_fbPixelSizeSim > 3);
             renderBuf[srcPixel + 3] = tempBufferEnd[dstPixel + 3];
          }
        }

      m_mesaCtx->Driver.UnmapRenderbuffer_base(m_mesaCtx, m_mesaColorBuffer);
      m_mesaCtx->Driver.UpdateState_base(m_mesaCtx);
      m_mesaCtx->Driver.Flush_base(m_mesaCtx);

    delete [] tempBuffer;
}

//copying the result depth buffer to mesa
void renderData_t::putDataOnDepthBuffer(){
    byte * tempBuffer = new byte [m_depthBufferSize];
    modeMemcpy(tempBuffer, m_deviceData + m_colorBufferByteSize,
          m_depthBufferSize, graphicsMemcpySimToHost);
    writeDrawBuffer("post_depth", tempBuffer,  m_depthBufferSize, m_bufferWidth, m_bufferHeight, "gray", 8*(int)m_depthSize);

    assert((m_depthSize == m_mesaDepthSize) or ((m_mesaDepthSize == DepthSize::Z16) and (m_depthSize == DepthSize::Z32)));
    byte* readDepth = tempBuffer;
    if((m_mesaDepthSize == DepthSize::Z16) and (m_depthSize == DepthSize::Z32)){
       int pixelBufferSize = getPixelBufferSize();
       uint16_t* depth16 = new uint16_t[pixelBufferSize];
       readDepth = (byte*) depth16;
       uint32_t* depth32 = (uint32_t*) tempBuffer;
       for(int i=0; i<pixelBufferSize; i++){
          depth16[i] = (uint16_t) (depth32[i] >> 16); //loose precision 
       }
       delete [] tempBuffer;
    }

    delete [] readDepth;
}


gl_state_index renderData_t::getParamStateIndexes(gl_state_index index) {
    assert(0);
    /*gl_program_parameter_list * paramList = m_mesaCtx->VertexProgram._Current->Base.Parameters;
    for (int i = 0; i < paramList->NumParameters; i++) {
        if (paramList->Parameters[i].Type == PROGRAM_STATE_VAR) {
            //DPRINTF(MesaGpgpusim, "state index %d = %d and the requested index is %d\n",i,paramList->Parameters[i].StateIndexes[0], index);
            if(paramList->Parameters[i].StateIndexes[0]==index)
                return paramList->Parameters[i].StateIndexes[0];
        }
    }*/
    return gl_state_index(NULL);
}

void renderData_t::copyStateData(void** fatCubinHandle) {
    //todo: double check that the generated ptx code should match the sizes here (maybe send it as a param)
    assert(sizeof(GLfloat)==CUDA_FLOAT_SIZE);
        const unsigned float_4vectorSize = VECTOR_SIZE* CUDA_FLOAT_SIZE;
    const unsigned float_4x4matrixSize = VECTOR_SIZE * VECTOR_SIZE * CUDA_FLOAT_SIZE;
    std::hash<std::string> strHash;


    //STATE_MVP_MATRIX
    {
        std::string varName = "state_matrix_mvp";
        graphicsRegisterVar(fatCubinHandle, (char*) strHash(varName), (char*) varName.c_str(),
                varName.c_str(), 0, float_4x4matrixSize, 1, 0);
        gl_state_index mvpState = getParamStateIndexes(STATE_MVP_MATRIX);
        if (mvpState != (gl_state_index)NULL) {
            gl_state_index state[5];
            state[0] = mvpState;
            state[1] = (gl_state_index) 0;
            state[2] = (gl_state_index) 0;
            state[3] = (gl_state_index) 3;
            state[4] = STATE_MATRIX_TRANSPOSE;

            //GLfloat mvpMatrix [VECTOR_SIZE * VECTOR_SIZE];

            gl_constant_value mvpMatrix [VECTOR_SIZE * VECTOR_SIZE];

            _mesa_fetch_state(m_mesaCtx, state, mvpMatrix);
            graphicsMemcpyToSymbol((char*) strHash(varName),
                    mvpMatrix,
                    sizeof (GLfloat) * VECTOR_SIZE* VECTOR_SIZE,
                    0,
                    graphicsMemcpyHostToSim);
        }
    }

    //STATE_MODELVIEW_INVERSE_MATRIX
    {
        std::string varName = "state_matrix_modelview_inverse";
        graphicsRegisterVar(fatCubinHandle, (char*)strHash(varName), (char*) varName.c_str(),
            varName.c_str(), 0, float_4x4matrixSize, 1, 0);
        gl_state_index modelViewState = getParamStateIndexes(STATE_MODELVIEW_MATRIX);
        if (modelViewState != (gl_state_index)NULL) {
            gl_state_index state[5];
            state[0] = modelViewState;
            state[1] = (gl_state_index) 0;
            state[2] = (gl_state_index) 0;
            state[3] = (gl_state_index) 3;
            state[4] = STATE_MATRIX_TRANSPOSE;
            //GLfloat modelViewMatrix [VECTOR_SIZE * VECTOR_SIZE];
            gl_constant_value modelViewMatrix [VECTOR_SIZE * VECTOR_SIZE];
            _mesa_fetch_state(m_mesaCtx, state, modelViewMatrix);
            graphicsMemcpyToSymbol((char*) strHash(varName),
                    modelViewMatrix,
                    sizeof (GLfloat) * VECTOR_SIZE* VECTOR_SIZE,
                    0,
                    graphicsMemcpyHostToSim);
        }
    }

    
    //STATE_PROJECTION_INVERSE_MATRIX
    {
        std::string varName = "state_matrix_projection_inverse";
        graphicsRegisterVar(fatCubinHandle, (char*)strHash(varName), (char*) varName.c_str(),
            varName.c_str(), 0, float_4x4matrixSize, 1, 0);
        const gl_state_index projectoinState = getParamStateIndexes(STATE_PROJECTION_MATRIX);
        if (projectoinState != (gl_state_index)NULL) {
            gl_state_index state[5];
            state[0] = projectoinState;
            state[1] = (gl_state_index) 0;
            state[2] = (gl_state_index) 0;
            state[3] = (gl_state_index) 3;
            state[4] = STATE_MATRIX_TRANSPOSE;
            //GLfloat projectionMatrix [VECTOR_SIZE * VECTOR_SIZE];
            gl_constant_value projectionMatrix [VECTOR_SIZE * VECTOR_SIZE];
            _mesa_fetch_state(m_mesaCtx, state, projectionMatrix);
            graphicsMemcpyToSymbol((char*) strHash(varName),
                    projectionMatrix,
                    sizeof (GLfloat) * VECTOR_SIZE* VECTOR_SIZE,
                    0,
                    graphicsMemcpyHostToSim);
        }
    }
    
    
    //STATE_LIGHT
    {
        std::string varName = "state_light_position";
        graphicsRegisterVar(fatCubinHandle, (char*)strHash(varName), (char*) varName.c_str(),
            varName.c_str(), 0, MAX_LIGHTS * float_4vectorSize, 1, 0);
        GLfloat vectorValue[VECTOR_SIZE];
        for (int light = 0; light < MAX_LIGHTS; light++) {
            COPY_4V(vectorValue, m_mesaCtx->Light.Light[light].EyePosition);
            graphicsMemcpyToSymbol((char*)strHash(varName),
                    vectorValue,
                    sizeof (GLfloat)*VECTOR_SIZE,
                    sizeof (GLfloat)*(light * VECTOR_SIZE),
                    graphicsMemcpyHostToSim);
        }
    }

    //UNIFORM VARIABLES
    assert(0);
    /*
    if (m_mesaCtx->_Shader->ActiveProgram) {
            std::string varName = "vertex_program_locals";
            graphicsRegisterVar(fatCubinHandle, (char*)strHash(varName), (char*) varName.c_str(),
            varName.c_str(), 0, MAX_UNIFORMS* float_4vectorSize, 1, 0);
            
            int fillIndex = 0;
            //we do this as parameters contains state parameters which we copy above in isolation
            for (int i = m_mesaCtx->_Shader->ActiveProgram->Uniforms->NumUniforms - 1; i >= 0; i--) {
            //for (int i = 0; i< m_mesaCtx->_Shader->ActiveProgram->Uniforms->NumUniforms; i++) {
                if (m_mesaCtx->_Shader->ActiveProgram->Uniforms->Uniforms[i].VertPos >= 0) {
                    struct gl_program *prog = &m_mesaCtx->_Shader->ActiveProgram->VertexProgram->Base;
                    int paramPos = m_mesaCtx->_Shader->ActiveProgram->Uniforms->Uniforms[i].VertPos;

                    const struct gl_program_parameter *param = &prog->Parameters->Parameters[paramPos];
                    if (param->Type != PROGRAM_UNIFORM) continue; //we deal with different types in a different way (as texture of type "PROGRAM_SAMPLER")
                    int numberOfElements = param->Size / VECTOR_SIZE;
                    if (numberOfElements == 0) numberOfElements = 1; //at least 1 vector should be copied for smaller uniforms
                    
                    //DPRINTF(MesaGpgpusim, "placing uniform:%d, size:%d at location:%d with values:\n", i, param->Size, fillIndex);
                    //for (int r = 0; r < numberOfElements; r++){
                    //    DPRINTF(MesaGpgpusim, "%s: ", prog->Parameters->Parameters[paramPos + r].Name);
                    //    for(int e = 0; e < VECTOR_SIZE; e++){
                    //        DPRINTF(MesaGpgpusim, "%f ", prog->Parameters->ParameterValues[paramPos + r][e]);
                    //    }
                    //    DPRINTF(MesaGpgpusim, "\n");
                    //}
                    
                    for (int r = 0; r < numberOfElements; r++) {
                        graphicsMemcpyToSymbol((char*)strHash(varName), //name
                                prog->Parameters->ParameterValues[paramPos + r], //src addr
                                sizeof (GLfloat)*VECTOR_SIZE, //byte count
                                (sizeof (GLfloat)*VECTOR_SIZE) * fillIndex, //offset
                                graphicsMemcpyHostToSim);
                        fillIndex++;
                    }
                }

            }
        }

        {
            std::string varName = "fragment_program_locals";
            graphicsRegisterVar(fatCubinHandle, (char*)strHash(varName), (char*) varName.c_str(),
            varName.c_str(), 0, MAX_UNIFORMS* float_4vectorSize, 1, 0);
            int fillIndex = 0;
            //we do this as parameters contains state parameters which we copy above in isolation
            for (int i = m_mesaCtx->_Shader->ActiveProgram->Uniforms->NumUniforms - 1; i >= 0; i--) {
                if (m_mesaCtx->_Shader->ActiveProgram->Uniforms->Uniforms[i].FragPos >= 0) {
                    struct gl_program *prog = &m_mesaCtx->_Shader->ActiveProgram->FragmentProgram->Base;
                    int paramPos = m_mesaCtx->_Shader->ActiveProgram->Uniforms->Uniforms[i].FragPos;

                    const struct gl_program_parameter *param = &prog->Parameters->Parameters[paramPos];
                    if (param->Type != PROGRAM_UNIFORM) continue; //we deal with different types in a different way (as texture of type "PROGRAM_SAMPLER")
                    int numberOfElements = param->Size / VECTOR_SIZE;
                    if (numberOfElements == 0) numberOfElements = 1; //at least 1 vector should be copied for smaller uniforms
                    for (int r = 0; r < numberOfElements; r++) {
                        graphicsMemcpyToSymbol((char*)strHash(varName),
                                prog->Parameters->ParameterValues[paramPos + r], 
                                sizeof (GLfloat)*VECTOR_SIZE, 
                                (sizeof (GLfloat)*VECTOR_SIZE) * fillIndex, 
                                graphicsMemcpyHostToSim);
                        fillIndex++;
                    }
                }

            }
        }
    }
    */
}

bool renderData_t::isDepthTestEnabled(){
   if(isBlendingEnabled())
      return false;
    if(g_renderData.m_mesaCtx->Depth.Test != 0)
        return true;
    return false;
}

bool renderData_t::isBlendingEnabled() {
   if (m_mesaCtx->Color.BlendEnabled & 1) {
      return true;
   }
   return false;
}

void renderData_t::getBlendingMode(GLenum * src, GLenum * dst, GLenum* srcAlpha, GLenum * dstAlpha, GLenum* eqnRGB, GLenum* eqnAlpha, GLfloat * blendColor){
    *src = m_mesaCtx->Color.Blend[0].SrcRGB;
    *dst = m_mesaCtx->Color.Blend[0].DstRGB;
    *srcAlpha = m_mesaCtx->Color.Blend[0].SrcA;
    *dstAlpha = m_mesaCtx->Color.Blend[0].DstA;
    *eqnRGB = m_mesaCtx->Color.Blend[0].EquationRGB;
    *eqnAlpha = m_mesaCtx->Color.Blend[0].EquationA;
    memcpy(blendColor,&m_mesaCtx->Color.BlendColor,sizeof(GLfloat)*VECTOR_SIZE);
}

void renderData_t::writeVertexResult(unsigned threadID, unsigned resAttribID, unsigned attribIndex, float data){
    assert(0);
    //DPRINTF(MesaGpgpusim, "writing vs result at thread=%d attrib=[%d][%d]=%f\n", threadID, resAttribID, attribIndex, data);
   //vertexStageData->results[resAttribID].data[threadID][attribIndex] = data;
}

void renderData_t::endFragmentShading() {
    endDrawCall(); 
}

void renderData_t::checkGraphicsThreadExit(void * kernelPtr, unsigned tid, void* stream){
   if(stream == m_sShading_info.cudaStreamVert){
       m_sShading_info.completed_threads_verts++;
       assert(m_sShading_info.completed_threads_verts <= m_sShading_info.launched_threads_verts);
       if(m_sShading_info.completed_threads_verts == m_sShading_info.launched_threads_verts){
          m_flagEndVertexShader = true;
       }
       if(m_sShading_info.completed_threads_verts%10000 == 0)
         DPRINTF(MesaGpgpusim, "completed threads = %d out of %d\n", m_sShading_info.completed_threads_frags,  m_sShading_info.launched_threads_verts);
   } else  if(stream == m_sShading_info.cudaStreamFrag){
      m_sShading_info.completed_threads_frags++;
      assert(m_sShading_info.completed_threads_frags <= m_sShading_info.launched_threads_frags);
      tileStream_t* tst = m_sShading_info.getTCTile(tid);
      assert(tst->pendingFrags>0);
      tst->pendingFrags--;
      if(tst->pendingFrags==0){
         assert(!tst->tcTilePtr->done);
         tst->tcTilePtr->done = true;
         //m_sShading_info.cudaStreamTiles.erase((uint64_t)stream);
      }

       if(m_sShading_info.completed_threads_frags%10000 == 0)
         DPRINTF(MesaGpgpusim, "completed threads = %d out of %d\n", m_sShading_info.completed_threads_frags,  m_sShading_info.launched_threads_frags);

      if (m_sShading_info.completed_threads_frags == m_sShading_info.launched_threads_frags
            and m_sShading_info.completed_threads_verts == m_sShading_info.launched_threads_verts){
         
         m_flagEndFragmentShader = (m_sShading_info.sent_simt_prims == 0);
         printf("done verts = %d\n", m_sShading_info.completed_threads_verts);
         printf("done frags = %d\n", m_sShading_info.completed_threads_frags);
         /*if(m_inShaderDepth or !isDepthTestEnabled())
         {
            m_flagEndFragmentShader = true;
         } else {
            //only done if early-Z is also done
            m_flagEndFragmentShader = m_sShading_info.doneEarlyZ;
         }*/
      }
   } else {
      //non graphics kernel
   }
}

void renderData_t::checkEndOfShader(CudaGPU * cudaGPU){
   assert(m_sShading_info.pending_kernels > 0);
   m_sShading_info.pending_kernels--;
   if(m_flagEndFragmentShader and m_flagEndVertexShader 
         and (m_sShading_info.pending_kernels==0)){
         endFragmentShading();
         m_flagEndVertexShader = false;
         m_flagEndFragmentShader = false;
   }

}

void renderData_t::doneEarlyZ(){
   m_sShading_info.doneEarlyZ = true;

   if(m_sShading_info.completed_threads_frags == m_sShading_info.launched_threads_frags){
      endFragmentShading();
      m_flagEndFragmentShader = false;
   } 
}



void renderData_t::launchFragmentTile(RasterTile * rasterTile, unsigned tileId){
   /*
   unsigned fragsCount = rasterTile->setActiveFragmentsIndices();

   DPRINTF(MesaGpgpusim, "Launching tile %d of fragments, active count=%d of of %d\n", tileId, fragsCount, rasterTile->size());

   //no active fragments
   if(fragsCount == 0){
      return;
   }

   DPRINTF(MesaGpgpusim, "Launching a tile of fragments, active count=%d of of %d\n", fragsCount, rasterTile->size());
   printf("Launching a tile of fragments, active count=%d of of %d\n", fragsCount, rasterTile->size());


   unsigned threadsPerBlock = m_wg_size; 
   unsigned numberOfBlocks = (rasterTile->size() + threadsPerBlock -1 ) / threadsPerBlock;

   m_sShading_info.cudaStreams.push_back(cudaStream_t());
   graphicsStreamCreate(&m_sShading_info.cudaStreams.back()); 


   byte* arg= getDeviceData() + getColorBufferByteSize();
   tileStream_t map;
   map.tileId = tileId;
   map.primId = rasterTile->primId;
   map.pendingFrags = numberOfBlocks*threadsPerBlock;

   m_sShading_info.cudaStreamTiles[(uint64_t)(m_sShading_info.cudaStreams.back())] = map;

   uint64_t streamId = (uint64_t)m_sShading_info.cudaStreams.back();
   DPRINTF(MesaGpgpusim, "running %d threads for  tile %d with %d fragments on stream %ld\n", rasterTile->size() , tileId, rasterTile->size(), streamId );
   assert( graphicsConfigureCall(numberOfBlocks, threadsPerBlock, 0, m_sShading_info.cudaStreams.back()) == cudaSuccess);
   assert(graphicsSetupArgument((void*) &arg, sizeof (byte*), 0) == cudaSuccess);
   assert(graphicsLaunch(getCurrentShaderId(FRAGMENT_PROGRAM), &m_sShading_info.fragCodeAddr) == cudaSuccess);
   assert(m_sShading_info.fragCodeAddr != NULL);

   m_sShading_info.launched_threads_frags+= numberOfBlocks*threadsPerBlock;
   DPRINTF(MesaGpgpusim, "total launched threads = %d\n", m_sShading_info.launched_threads_frags);

   m_sShading_info.currentPass = stage_shading_info_t::GraphicsPass::Fragment;
   */
}

void renderData_t::launchVRTile(){
   const unsigned vertsCount = m_sShading_info.vertex_data.size();
   assert(m_sShading_info.launched_threads_verts <= vertsCount);
   //all vertices have been launched done here
   if(m_sShading_info.launched_threads_verts == vertsCount)
      return;
   //all cores running vert shaders, try next cycle
   if(m_busyVertsCores.size() == (m_coresPerCluster*m_numClusters))
      return;
   const unsigned remainingVerts = 
      vertsCount - m_sShading_info.launched_threads_verts;
   /*if(tcTile == NULL){
      assert(donePrims > 0);
      assert(m_sShading_info.sent_simt_prims >= donePrims);
      m_sShading_info.sent_simt_prims-=donePrims;

      if(m_sShading_info.sent_simt_prims == 0){
         assert(m_sShading_info.fragKernel!=NULL);
         m_sShading_info.fragKernel->setDrawCallDone();
         if(m_sShading_info.completed_threads_frags == m_sShading_info.launched_threads_frags){
            m_flagEndFragmentShader = true;
         }
      }
      return;
   }*/

   //gather how many verts shaders we are going to launch 
   unsigned selectedCoreSid = (unsigned)-1;
   for(unsigned c=0; c<(m_numClusters*m_coresPerCluster); c++){
      if(m_busyVertsCores.count(c) == 0){
         m_busyVertsCores.insert(c);
         selectedCoreSid = c;
         break;
      }
   }
   assert(selectedCoreSid != (unsigned) -1);
  
   unsigned threadsPerBlock = std::max(remainingVerts, (unsigned)256); //TODO: make size configurable
   unsigned numberOfBlocks = 1;


   if(m_sShading_info.launched_threads_verts == 0){
      graphicsStreamCreate(&m_sShading_info.cudaStreamVert); 
      byte* arg= getDeviceData() + getColorBufferByteSize();

      DPRINTF(MesaGpgpusim, "running tile %d vertices on stream %ld\n", threadsPerBlock, (unsigned long) m_sShading_info.cudaStreamVert);
      assert( graphicsConfigureCall(numberOfBlocks, threadsPerBlock, 0, m_sShading_info.cudaStreamVert) == cudaSuccess);
      assert(graphicsSetupArgument((void*) &arg, sizeof (byte*), 0) == cudaSuccess);
      assert(graphicsLaunch(getCurrentShaderId(VERTEX_PROGRAM), &m_sShading_info.vertCodeAddr, &m_sShading_info.vertKernel) == cudaSuccess);
      assert(m_sShading_info.vertKernel != NULL);
      assert(m_sShading_info.vertCodeAddr != NULL);
      m_sShading_info.pending_kernels++;
   } else {
      m_sShading_info.vertKernel->add_blocks(numberOfBlocks, m_sShading_info.launched_threads_verts);
   }

   m_sShading_info.vertKernel->assignCtaToCore(
         numberOfBlocks, 
         selectedCoreSid,
         m_sShading_info.launched_threads_verts);
   m_sShading_info.launched_threads_verts+= threadsPerBlock;
   DPRINTF(MesaGpgpusim, "total launched threads = %d\n", m_sShading_info.launched_threads_verts);
}

void renderData_t::launchTCTile(
      unsigned clusterId,
      tcTilePtr_t tcTile, unsigned donePrims){
   if(tcTile == NULL){
      assert(donePrims > 0);
      assert(m_sShading_info.sent_simt_prims >= donePrims);
      m_sShading_info.sent_simt_prims-=donePrims;

      if(m_sShading_info.sent_simt_prims == 0){
         assert(m_sShading_info.fragKernel!=NULL);
         m_sShading_info.fragKernel->setDrawCallDone();
         if(m_sShading_info.completed_threads_frags == m_sShading_info.launched_threads_frags){
            m_flagEndFragmentShader = true;
         }
      }
      return;
   }

   assert(tcTile->size() > 0);
   DPRINTF(MesaGpgpusim, "launching a TC tile with %d fragments\n", tcTile->size());
   unsigned threadsPerBlock = m_wg_size; 
   unsigned numberOfBlocks = (tcTile->size() + threadsPerBlock -1 ) / threadsPerBlock;

      /*printf("launching a TC tile with (%d) active fragments with %d threads on %d\n",
          tcTile->getActiveFrags(), threadsPerBlock*numberOfBlocks, clusterId);*/

   if(m_sShading_info.launched_threads_frags == 0){
      graphicsStreamCreate(&m_sShading_info.cudaStreamFrag);
      byte* arg= getDeviceData() + getColorBufferByteSize();
      tileStream_t* tst = new tileStream_t();
      tst->tcTilePtr = tcTile;
      tst->pendingFrags = numberOfBlocks*threadsPerBlock;
      tst->t_start = m_sShading_info.launched_threads_frags;
      tst->t_end = m_sShading_info.launched_threads_frags + tst->pendingFrags -1;
      //printf("launching a TC tile with (%d to %d) fragments on %d\n", tst.t_start, tst.t_end, clusterId);
      m_sShading_info.cudaStreamTiles.push_back(tst);
      for(unsigned tid =tst->t_start; tid <= tst->t_end; tid++){
         m_sShading_info.threadTileMap[tid] = tst;
      }
      DPRINTF(MesaGpgpusim, "running tile %lp with %d fragments on stream %ld\n", tcTile, tcTile->size(), (unsigned long)m_sShading_info.cudaStreamFrag);
      assert( graphicsConfigureCall(numberOfBlocks, threadsPerBlock, 0, m_sShading_info.cudaStreamFrag) == cudaSuccess);
      assert(graphicsSetupArgument((void*) &arg, sizeof (byte*), 0) == cudaSuccess);
      assert(graphicsLaunch(getCurrentShaderId(FRAGMENT_PROGRAM), &m_sShading_info.fragCodeAddr, &m_sShading_info.fragKernel) == cudaSuccess);
      assert(m_sShading_info.fragKernel != NULL);
      assert(m_sShading_info.fragCodeAddr != NULL);
      m_sShading_info.pending_kernels++;
      unsigned tcId = m_sShading_info.cudaStreamTiles.size();
      unsigned sid = (clusterId*m_coresPerCluster) + ((tcId/m_numClusters)%m_coresPerCluster);
      m_sShading_info.fragKernel->assignCtaToCore(
            numberOfBlocks, 
            sid,
            tst->t_start);
   } else {
      tileStream_t* tst = new tileStream_t();
      tst->tcTilePtr = tcTile;
      tst->pendingFrags = numberOfBlocks*threadsPerBlock;
      tst->t_start = m_sShading_info.launched_threads_frags;
      tst->t_end = m_sShading_info.launched_threads_frags + tst->pendingFrags - 1;
      /*printf("launching a TC tile with (%d to %d) fragments on %d\n", tst->t_start, tst->t_end, clusterId);*/
      m_sShading_info.cudaStreamTiles.push_back(tst);
      for(unsigned tid =tst->t_start; tid <= tst->t_end; tid++){
         m_sShading_info.threadTileMap[tid] = tst;
      }
      m_sShading_info.fragKernel->add_blocks(numberOfBlocks, tst->t_start);
      unsigned tcId = m_sShading_info.cudaStreamTiles.size(); 
      unsigned sid = (clusterId*m_coresPerCluster) + ((tcId/m_numClusters)%m_coresPerCluster);
      m_sShading_info.fragKernel->assignCtaToCore(
            numberOfBlocks, 
            sid,
            tst->t_start);
   }

   m_sShading_info.launched_threads_frags+= numberOfBlocks*threadsPerBlock;
   DPRINTF(MesaGpgpusim, "total launched threads = %d\n", m_sShading_info.launched_threads_frags);
}

byte* Utils::RGB888_to_RGBA888(byte* rgb, int size, byte alpha){
   const int rgb_chs = 3;
   const int rgba_chs = 4;
   byte* rgba = new byte[size*rgba_chs];

   for(int c=0; c < size; c++){
      rgba[c*rgba_chs + 0] = alpha; 
      rgba[c*rgba_chs + 1] = rgb[c*rgb_chs + 0];
      rgba[c*rgba_chs + 2] = rgb[c*rgb_chs + 1];
      rgba[c*rgba_chs + 3] = rgb[c*rgb_chs + 2];
   }

   return rgba;
}

void RasterTile::addFragment(fragmentData_t* frag){ 
   unsigned fragX = frag->uintPos(0)%tileW;
   unsigned fragY = frag->uintPos(1)%tileH;
   const unsigned qd = QUAD_SIZE/2;
   unsigned tidx = (fragY/qd)*(tileW/qd) + fragX/qd;
   unsigned quadIdx = fragY%qd == 0? 
      (fragX%qd == 0? 0 : 1): 
      (fragX%qd == 0? 2 : 3);
   assert(tidx < tileH*tileW/QUAD_SIZE);
   assert(not m_fragmentsQuads[tidx][quadIdx].alive);
   m_fragmentsQuads[tidx][quadIdx].frag = frag;
   m_fragmentsQuads[tidx][quadIdx].alive = true;
   m_fragmentsQuads[tidx][quadIdx].tile = this;
   if(m_addedFragsCount==0){
      //first frag, set front and back depths
      m_frontDepth = frag->uintPos(2);
      m_backDepth = frag->uintPos(2);
   } else {
      if(g_renderData.depthTest(m_frontDepth, frag->uintPos(2))){
         m_frontDepth = frag->uintPos(2);
      }
      if(!g_renderData.depthTest(m_backDepth, frag->uintPos(2))){
         m_backDepth = frag->uintPos(2);
      }
   }
   m_addedFragsCount++;
}

void RasterTile::testHizThresh(){
   assert(m_hizThreshSet);
   for(unsigned quadId=0; quadId < m_fragmentsQuads.size(); quadId++)
      for(unsigned fragId=0; fragId < m_fragmentsQuads[quadId].size(); fragId++){
         rasterFragment_t* frag = &m_fragmentsQuads[quadId][fragId];
         if(frag->alive){
            if(g_renderData.depthTest(frag->frag->uintPos(2), m_hizThresh)){
               frag->alive = false;
               m_activeCount--;
            }
         } 
      }
}


void renderData_t::generateDepthCode(FILE* inst_stream){
   if(not isDepthTestEnabled()) return;
   //FIXME set depth size status before code generation
   //const char* depthSize = m_depthSize==DepthSize::Z32? "u32" : "u16";
   const char* depthSize = "u32";
   fprintf(inst_stream, ".reg .pred testDepth, passedDepth;\n");
   fprintf(inst_stream, ".reg .u32 depthTestRes;\n");
   fprintf(inst_stream, "setp.eq.u32 passedDepth, !fflag, 0;\n");
   fprintf(inst_stream, "setp.eq.u32 testDepth, 0, %%skip_depth_test;\n");
   fprintf(inst_stream, "setp.eq.u32 testDepth, !fflag, !testDepth;\n");
   fprintf(inst_stream, "@testDepth ztest.global.%s depthTestRes;\n", depthSize);
   //after depth testing exit dead quads
   fprintf(inst_stream, "setp.ne.u32 qflag, 0, %%quad_active;\n");
   fprintf(inst_stream, "@!qflag exit;\n");
   fprintf(inst_stream, "@testDepth setp.ne.u32 passedDepth, 0, depthTestRes;\n");
   fprintf(inst_stream, "@passedDepth zwrite.global.%s;\n", depthSize);
}

void renderData_t::generateBlendCode(FILE* inst_stream){
   assert(m_fbPixelSizeSim == 4);
   fprintf(inst_stream, "setp.ne.u32 fflag, 0, %%fragment_active;\n");
   if(isBlendingEnabled()){
      fprintf(inst_stream, "@fflag blend.global.u32 %%color, COLOR0;\n");
      fprintf(inst_stream, "@fflag stp.global.u32 %%color, %%color;\n");
   } else {
      fprintf(inst_stream, "@fflag mov.u32 %%color, COLOR0;\n");
      fprintf(inst_stream, "@fflag stp.global.u32 %%color, %%color;\n");
   }
}

void renderData_t::modeMemcpy(byte* dst, byte *src, 
      unsigned count, enum cudaMemcpyKind kind){
   if(m_standaloneMode){
      CudaGPU* cg = CudaGPU::getCudaGPU(g_active_device);
      assert(cg->standaloneMode);
      GraphicsStandalone* gs = cg->getGraphicsStandalone();
      assert(gs != NULL);
      if(kind == graphicsMemcpyHostToSim)
         gs->physProxy.writeBlob((Addr)dst, src, count);
      else if(kind == graphicsMemcpySimToHost)
         gs->physProxy.readBlob((Addr)src, dst, count);
      else assert(0);
   } else {
      graphicsMemcpy(dst, src, count, kind);
   }
}

float* renderData_t::getTexCoords(unsigned utid, void* stream){
   unsigned tcSize = 0;
   tileStream_t* tst =  m_sShading_info.getTCTile(utid, &tcSize);
   unsigned qid = (utid%tcSize)/TGSI_QUAD_SIZE;
   if(tst->quadCoords.find(qid) == tst->quadCoords.end()){
      return NULL;
   } else {
      float* fcoords = tst->quadCoords[qid].getCoords();
      assert(tst->quadCoords[qid].remainingAccesses > 0);
      tst->quadCoords[qid].remainingAccesses--;
      if(tst->quadCoords[qid].remainingAccesses == 0)
         tst->quadCoords.erase(qid);
      return fcoords;
   }
}

void renderData_t::setTexCoords(unsigned utid, void* stream, float* coords){
   unsigned tcSize = 0;
   tileStream_t* tst = m_sShading_info.getTCTile(utid, &tcSize);
   unsigned qid = (utid%tcSize)/TGSI_QUAD_SIZE;
   assert(tst->quadCoords.find(qid) == tst->quadCoords.end());
   tst->quadCoords[qid] = quadTexCoords_t();
   tst->quadCoords[qid].setCoords(coords);
}


void renderData_t::setFragLiveStatus(unsigned utid, void* stream, bool status){
   unsigned tcSize = 0;
   tileStream_t* tst = m_sShading_info.getTCTile(utid, &tcSize);
   tcTilePtr_t tcTilePtr = tst->tcTilePtr;
   assert(utid%tcSize < tcTilePtr->size());
   fragmentData_t* frag = tcTilePtr->at(utid%tcSize) == NULL? 
      NULL: tcTilePtr->at(utid%tcSize)->frag;
   assert(frag);
   frag->isLive = status;
}
