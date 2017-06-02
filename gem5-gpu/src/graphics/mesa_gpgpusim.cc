/* 
 * Author: Ayub Gubran
 *
 * Created on December 5, 2011, 5:47 PM
 */


#include <time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <queue>
#include <map>
#include <stack>
#include <sys/stat.h>
#include <GL/gl.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/extensions/XShm.h>
#include <math.h>
#include <assert.h>
#include <functional>
#include <chrono>
#include <thread>


extern "C" {
#include "drivers/x11/xmesaP.h"
#include "main/config.h"
#include "main/glheader.h"
#include "main/colormac.h"
#include "main/macros.h"
#include "main/imports.h"
#include "program/prog_instruction.h"
#include "program/prog_statevars.h"
#include "program/prog_execute.h"
#include "program/prog_uniform.h"
#include "program/prog_parameter.h"
#include "swrast/s_context.h"
#include "swrast/s_fragprog.h"
#include "tnl/tnl.h"
#include "tnl/t_pipeline.h"
}


#include "graphics/mesa_gpgpusim.h"
#include "sim/simulate.hh"
#include "graphics/serialize_graphics.hh"
#include "graphics/libOpenglRender/SocketStream.hh"
#include "debug/MesaGpgpusim.hh"
#include "sim/sim_exit.hh"
#include "base/statistics.hh"


void startEarlyZ(CudaGPU* cudaGPU, uint64_t depthBuffStart, uint64_t depthBuffEnd, unsigned bufWidth, std::vector<RasterTile* >* tiles, DepthSize dSize, GLenum depthFunc,
      uint8_t* depthBuf, unsigned frameWidth, unsigned frameHeight, unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir);


renderData_t g_renderData;
int sizeOfEachFragmentData = FRAG_ATTRIB_MAX * sizeof (float) * 4;
#define DRAW_NON_SIM_FRAME true

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

float primitiveFragmentsData_t::getFragmentData(unsigned threadID, unsigned attribID, unsigned attribIndex, void * stream,
      stage_shading_info_t* si, bool z_unit_disabled) {

   float result = 0.0;
   if(z_unit_disabled){
      assert(threadID < m_fragments.size());
      result =  m_fragments[threadID].attribs[attribID][attribIndex];
      if(attribID == FRAG_ATTRIB_COL0) return primId;
   } else {
      unsigned tileId = si->cudaStreamTiles[(uint64_t)stream].tileId;
      printf("get data for tile %d, thread %d, attrib %d, idx %d\n", tileId, threadID, attribID, attribIndex);
      assert(threadID < (*(*(si->earlyZTiles))[tileId]).size());
      result =  (*(*(si->earlyZTiles))[tileId])[threadID].attribs[attribID][attribIndex];
      printf("data = %f\n", result);
   }
   return result;
}

int primitiveFragmentsData_t::getFragmentDataInt(unsigned threadID, unsigned attribID,
      unsigned attribIndex, void * stream, stage_shading_info_t* si, bool z_unit_disabled) {
   switch(attribID){
      case FRAG_ACTIVE: 
         {
            if(z_unit_disabled){
               if(threadID >= m_fragments.size()) return 0; //no fragment, padding lane
               return 1;
            } else {
               unsigned tileId = si->cudaStreamTiles[(uint64_t)stream].tileId;
               printf("querying tileId=%d, threadID=%d\n", tileId, threadID);
               if(threadID >= (*(*(si->earlyZTiles))[tileId]).size()) {
                  printf("tile %d, thread %d inactive\n", tileId, threadID);
                  return 0; //padding thread
               }
               printf("tile %d, thread %d active\n", tileId, threadID);
               return (*(*(si->earlyZTiles))[tileId])[threadID].passedDepth == true? 1 : 0;
            }
         }
         break;
      default: printf("Invalid attribID: %d \n", attribID);
               abort();
   }
   return 0;
}

void primitiveFragmentsData_t::addFragment(fragmentData_t fd) {
    maxDepth = std::max(maxDepth, (uint64_t) fd.intPos[2]);
    minDepth = std::min(minDepth, (uint64_t) fd.intPos[2]);
    m_fragments.push_back(fd);
}


//the current code is not efficient, might be rewritten some time later
void primitiveFragmentsData_t::sortFragmentsInRasterOrder(unsigned frameHeight, unsigned frameWidth,
        const unsigned tileH, const unsigned tileW,
        const unsigned blockH, const unsigned blockW, const RasterDirection rasterDir) {
    
    assert(rasterDir==HorizontalRaster or rasterDir==BlockedHorizontal); //what we do here so far
    
    //checking if a suitable block size is provided
    assert((blockH%tileH)==0);
    assert((blockW%tileW)==0);
   
    //DPRINTF(MesaGpgpusim, "tileW = %d, and tileH = %d\n", tileW, tileH);
    printf("Current frame size WxH=%dx%d\n", frameWidth, frameHeight);
 
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

    std::vector<std::vector<fragmentData_t> > fragmentTiles;

    //we add empty sets and then we will fill them
    const unsigned fragmentsPerTile = tileH * tileW;
    assert(0 == ((frameHeight* frameWidth) % fragmentsPerTile));
    unsigned tilesCount = (frameHeight * frameWidth) / fragmentsPerTile;


    for (unsigned tile = 0; tile < tilesCount; tile++) {
        std::vector<fragmentData_t> aSet;
        fragmentTiles.push_back(aSet);
    }

    assert(fragmentTiles.size() == tilesCount);
    assert((frameWidth%tileW) == 0);
    assert((frameHeight%tileH) == 0);
            
    unsigned const numberOfHorizontalTiles = frameWidth / tileW;
    
    //now we figure which tile every fragment belongs to
    for (int frag = 0; frag < m_fragments.size(); frag++) {
        unsigned xPosition = m_fragments[frag].intPos[0];
        unsigned yPosition = m_fragments[frag].intPos[1];
        unsigned tileXCoord = xPosition / tileW;
        unsigned tileYCoord = yPosition / tileH; //normalize this guy
        assert(tileXCoord<numberOfHorizontalTiles);
        unsigned tileIndex = tileYCoord * numberOfHorizontalTiles + tileXCoord;
        assert(tileIndex<fragmentTiles.size());
        fragmentTiles[tileIndex].push_back(m_fragments[frag]);
                
        //make sure that we do not add more fragments in each tile than we should have
        //assert(fragmentTiles[tileIndex].size() <= (tileH * tileW));
    }

    unsigned originalSize = m_fragments.size();
    m_fragments.clear();

    //now adding the fragments in the raster order, tile moves horizontally

    if (rasterDir == HorizontalRaster) {
         //DPRINTF(MesaGpgpusim, "raster order: HorizontalRaster\n");
        for (unsigned tile = 0; tile < fragmentTiles.size(); tile++) {
            for (unsigned frag = 0; frag < fragmentTiles[tile].size(); frag++) {
                m_fragments.push_back(fragmentTiles[tile][frag]);
            }
        }
    } else if (rasterDir == BlockedHorizontal) {
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


void renderData_t::runEarlyZ(CudaGPU * cudaGPU, unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir) {
   RasterTiles * allTiles = new RasterTiles();
   for(int prim=0; prim < drawPrimitives.size(); prim++){
      RasterTiles * primTiles = drawPrimitives[prim].sortFragmentsInTiles(m_bufferHeight, m_bufferWidth, tileH, tileW, blockH, blockW, dir);
      printf("prim %d tiles = %d\n", prim, primTiles->size());
      for(int tile=0; tile < primTiles->size(); tile++){
         if((*primTiles)[tile]->size() == 0) continue;
         allTiles->push_back((*primTiles)[tile]);
      }
      delete primTiles;
   }

   printf("number of tiles = %d\n", allTiles->size());
   uint64_t depthBuffEndAddr = (uint64_t)m_deviceData + m_colorBufferByteSize + m_depthBufferSize;
   uint64_t depthBuffStartAddr = (uint64_t)m_deviceData + m_colorBufferByteSize; 
   printf("depthBuffer start = %x, end =%x\n", depthBuffStartAddr, depthBuffEndAddr);
   m_sShading_info.doneEarlyZ = false;
   m_sShading_info.earlyZTiles = allTiles;
   m_sShading_info.completed_threads = 0;
   m_sShading_info.launched_threads = 0;
   assert(m_sShading_info.fragCodeAddr == NULL);
   //ZUnit* zunit = cudaGPU->getZUnit();
   //printf("zunit ptr =%x \n", zunit);
   //zunit->startEarlyZ(depthBuffStart, depthBuffEnd, bufWidth, allTiles, dSize, depthFunc);
   startEarlyZ(cudaGPU, depthBuffStartAddr, depthBuffEndAddr, m_bufferWidth, allTiles, m_depthSize, m_mesaCtx->Depth.Func, 
         m_depthBuffer, m_bufferWidth, m_bufferHeight, tileH, tileW, blockH, blockW, dir);
}

RasterTiles* primitiveFragmentsData_t::sortFragmentsInTiles(unsigned frameHeight, unsigned frameWidth,
        const unsigned tileH, const unsigned tileW,
        const unsigned blockH, const unsigned blockW, const RasterDirection rasterDir) {
   

    assert(rasterDir==HorizontalRaster or rasterDir==BlockedHorizontal);
    printf("Current frame size WxH=%dx%d\n", frameWidth, frameHeight);


    //checking if a suitable block size is provided
    assert((blockH%tileH)==0);
    assert((blockW%tileW)==0);
 
    //adding padding for rounded pixel locations
    frameHeight+= blockH;
    frameWidth += blockW;
    
    if ( (frameWidth % blockW) != 0) {
        frameWidth -= frameWidth % blockW;
        frameWidth += blockW;
    }

    if ((frameHeight % blockH) != 0) {
        frameHeight -= frameHeight % blockH;
        frameHeight += blockH;
    }
    
    std::vector<RasterTile* > * fragmentTiles = new std::vector<RasterTile* >();

    const unsigned fragmentsPerTile = tileH * tileW;
    assert(0 == ((frameHeight* frameWidth) % fragmentsPerTile));
    unsigned tilesCount = (frameHeight * frameWidth) / fragmentsPerTile;

    DPRINTF(MesaGpgpusim, "Sorting %d framgents in %d tiles \n", m_fragments.size(), tilesCount);

    for (unsigned tile = 0; tile < tilesCount; tile++) {
        //std::vector<fragmentData_t>* aSet = new std::vector<fragmentData_t>();
        RasterTile * rtile = new RasterTile(primId, tile);
        fragmentTiles->push_back(rtile);
    }

    assert(fragmentTiles->size() == tilesCount);
    assert((frameWidth%tileW) == 0);
    assert((frameHeight%tileH) == 0);
            
    unsigned const numberOfHorizontalTiles = frameWidth / tileW;
    
    //now we figure which tile every fragment belongs to
    for (int frag = 0; frag < m_fragments.size(); frag++) {
        unsigned xPosition = m_fragments[frag].intPos[0];
        unsigned yPosition = m_fragments[frag].intPos[1];
        unsigned tileXCoord = xPosition / tileW;
        unsigned tileYCoord = yPosition / tileH; //normalize this guy
        assert(tileXCoord<numberOfHorizontalTiles);
        unsigned tileIndex = tileYCoord * numberOfHorizontalTiles + tileXCoord;
        assert(tileIndex < fragmentTiles->size());
        (*fragmentTiles)[tileIndex]->push_back(m_fragments[frag]);
                
        //make sure that we do not add more fragments in each tile than we should have
        assert((*fragmentTiles)[tileIndex]->size() <= (tileH * tileW));
    }

    unsigned fragCount = 0;

    for(int i=0; i< fragmentTiles->size(); i++){
       fragCount += (*fragmentTiles)[i]->size();
    }

    assert(fragCount == m_fragments.size());

    return fragmentTiles;
}

void primitiveFragmentsData_t::clear() {
    m_fragments.clear();
}

renderData_t::renderData_t() {
    m_deviceData = NULL;
    m_currentFrame = 0;
    //callsCount = 0;
    m_drawcall_num = 0;
    m_tcPid = -1;
    m_tcTid = -1;
    m_flagEndVertexShader = false;
    m_flagEndFragmentShader = false;

    
    m_inShaderBlending = false;
    m_inShaderDepth = false;
    m_useDefaultShaders = false;
    for(unsigned i=0; i<MAX_COMBINED_TEXTURE_IMAGE_UNITS; i++)
        textureArray[i]=NULL;
    //get_GPGPUSim_Context(); //a hack to force initializing gpgpusim library
}

renderData_t::~renderData_t() {
}

float renderData_t::getFragmentData(unsigned threadID, unsigned attribID, unsigned attribIndex, void * stream) {
    bool z_unit_disabled = m_inShaderDepth or !isDepthTestEnabled(); 
    unsigned primId = m_sShading_info.cudaStreamTiles[(uint64_t)stream].primId;
    assert(primId < drawPrimitives.size());
    return drawPrimitives[primId].getFragmentData(threadID, attribID, attribIndex, stream, &m_sShading_info, z_unit_disabled);

}

int renderData_t::getFragmentDataInt(unsigned threadID, unsigned attribID, unsigned attribIndex, void * stream) {
    bool z_unit_disabled = m_inShaderDepth or !isDepthTestEnabled(); 

    unsigned primId = m_sShading_info.cudaStreamTiles[(uint64_t)stream].primId;
    assert(primId < drawPrimitives.size());
    return drawPrimitives[primId].getFragmentDataInt(threadID, attribID, attribIndex, stream, &m_sShading_info, z_unit_disabled);

}

int renderData_t::getVertexDataInt(unsigned threadID, unsigned attribID, unsigned attribIndex, void * stream) {
   switch(attribID){
      case VERT_ACTIVE: 
         if(threadID >= m_sShading_info.launched_threads)  return 0;
         return 1;
         break;
      default: printf("Invalid attribID: %d \n", attribID);
               abort();
   }
}

void renderData_t::addFragment(fragmentData_t fragmentData) {
    DPRINTF(MesaGpgpusim, "adding a fragment to primitive %d, fragments count=%d\n", drawPrimitives.size()-1, drawPrimitives.back().size());
    //printf( "adding a fragment to primitive %d, fragments count=%d\n", drawPrimitives.size()-1, drawPrimitives.back().size());
    drawPrimitives.back().addFragment(fragmentData);
}

void renderData_t::addPrimitive() {
    if(!GPGPUSimSimulationActive()) return;
    primitiveFragmentsData_t prim(drawPrimitives.size());
    drawPrimitives.push_back(prim);
    printf("adding primitive, total = %d\n", drawPrimitives.size());
}


void renderData_t::sortFragmentsInRasterOrder(unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir) {
   for(int prim=0; prim < drawPrimitives.size(); prim++)
      drawPrimitives[prim].sortFragmentsInRasterOrder(m_bufferHeight, m_bufferWidth, tileH, tileW, blockH, blockW, dir);
}

void renderData_t::endDrawCall() {
   printf("endDrawCall: start\n");
    putDataOnColorBuffer();
    if(isDepthTestEnabled())
       putDataOnDepthBuffer();
    delete [] lastFatCubin->ident;
    delete [] lastFatCubin->ptx[0].gpuProfileName;
    delete [] lastFatCubin->ptx[0].ptx;
    delete [] lastFatCubin->ptx;
    delete lastFatCubin;
    graphicsFree(m_sShading_info.allocAddr);
    graphicsFree(m_sShading_info.deviceVertsAttribs);
    graphicsFree(m_sShading_info.vertCodeAddr);
    graphicsFree(m_deviceData);
    if(m_sShading_info.fragCodeAddr) graphicsFree(m_sShading_info.fragCodeAddr);
    printf("endDrawCall: clearing textures\n");
    for(int tex=0; tex<MAX_COMBINED_TEXTURE_IMAGE_UNITS; tex++){
        if(textureArray[tex]!=NULL){
            graphicsFreeArray(textureArray[tex]);
            textureArray[tex] = NULL;
        }
    }
    m_deviceData = NULL;
    for (int i = 0; i < drawPrimitives.size(); i++) {
        drawPrimitives[i].clear();
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
    for(int st=0; st < m_sShading_info.cudaStreams.size(); st++){
       graphicsStreamDestroy(m_sShading_info.cudaStreams[st]);
    }
    m_sShading_info.cudaStreams.clear();
    m_sShading_info.clear();
    for(auto tr: textureRefs)
       delete tr;
    textureRefs.clear();
    drawPrimitives.clear();
    if(m_depthBuffer!=NULL) {
       delete [] m_depthBuffer;    
       m_depthBuffer = NULL;
    }
    printf("endDrawCall: done\n");
    Stats::dump();
    Stats::reset();
}

void renderData_t::initParams(unsigned int startFrame, unsigned int endFrame, int startDrawcall, unsigned int endDrawcall,
        unsigned int tile_H, unsigned int tile_W, unsigned int block_H, unsigned int block_W,
        unsigned blendingMode, unsigned depthMode, unsigned cptStartFrame, unsigned cptEndFrame, unsigned cptPeroid, bool skipCpFrames, char* outdir) {
    m_startFrame = startFrame;
    m_endFrame = endFrame;
    m_startDrawcall = startDrawcall;
    m_endDrawcall = endDrawcall;
    m_tile_H = tile_H;
    m_tile_W = tile_W;
    m_block_H = block_H;
    m_block_W = block_W;
    m_inShaderBlending = (blendingMode != 0);
    m_inShaderDepth = (depthMode != 0);
    printf("inshader depth = %d\n", m_inShaderDepth);
    m_cptStartFrame = cptStartFrame;
    m_cptEndFrame = cptEndFrame;
    m_cptPeroid = cptPeroid;
    m_skipCpFrames = skipCpFrames;
    m_cptNextFrame = (unsigned) -1;
    m_outdir = outdir;
    
    m_intFolder = m_fbFolder = outdir;
    m_intFolder+= "/gpgpusimShaders";
    m_fbFolder += "/gpgpusimFrameDumps";
    //create if not exist
    system(std::string("mkdir -p " + m_intFolder).c_str());
    system(std::string("mkdir -p " + m_fbFolder).c_str());
    //clear older files if any
    system(std::string("rm -f "+ m_intFolder + "/*").c_str());
    system(std::string("rm -f " + m_fbFolder + "/*").c_str());

    vGlslPrfx = m_intFolder+"/vertex_glsl";
    vARBPrfx = m_intFolder+"/vertex_arb";
    vPTXPrfx = m_intFolder+"/vertex_shader";
    fGlslPrfx = m_intFolder+"/fragment_glsl";
    fARBPrfx = m_intFolder+"/fragment_arb";
    fPTXPrfx = m_intFolder+"/fragment_shader";
    fPtxInfoPrfx = m_intFolder+"/shader_ptxinfo";
}

bool renderData_t::useInShaderBlending() const {
    return m_inShaderBlending;
}

void renderData_t::checkExitCond(){
   if(((m_currentFrame== m_endFrame) and (m_drawcall_num > m_endDrawcall)) or (m_currentFrame > m_endFrame)){
      exitSimLoop("gem5 exit, end of graphics simulation", 0, curTick(), 0, true);
   }
}

void renderData_t::incCurrentFrame(){
   bool wasActive = GPGPUSimActiveFrame();
   m_currentFrame++;

   /*
   if(GPGPUSimActiveFrame()){
      if(!wasActive) SetSoftProg(1);
   } else if(wasActive){
      SetSoftProg(0);
   }*/

   m_drawcall_num = 0;
   checkpoint();
   checkExitCond();
}

bool renderData_t::GPGPUSimActiveFrame(){
   bool isFrame = ((m_currentFrame >= m_startFrame)
          and (m_currentFrame <= m_endFrame) 
          and !checkpointGraphics::isUnserializingCp());

   return isFrame;
}

bool renderData_t::GPGPUSimSimulationActive(){    
   bool isFrame = GPGPUSimActiveFrame(); 
   bool afterStartDrawcall = ((m_currentFrame== m_startFrame) and (m_drawcall_num >= m_startDrawcall)) or (m_currentFrame > m_startFrame);
   bool beforeEndDrawcall =  ((m_currentFrame== m_endFrame) and (m_drawcall_num <= m_endDrawcall)) or (m_currentFrame < m_endFrame);

   return (isFrame and afterStartDrawcall and beforeEndDrawcall);
}

bool renderData_t::GPGPUSimSkipCpFrames(){
   bool skipCpFrames = (checkpointGraphics::isUnserializingCp() and m_skipCpFrames);
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

//todo: this is a hack to work with Anrdoid composition calls
//should be replaced with a better solution
std::string renderData_t::getDefaultVertexShader() {
    return "#version 110 \n"
    "void main() \n"
    "{ \n"
    "gl_TexCoord[0] = gl_MultiTexCoord0; \n"
    "gl_Position = ftransform(); \n}"
    ;
}

std::string renderData_t::getDefaultFragmentShader() {
    return "#version 110 \n"
    "uniform sampler2D baseSampler; \n"
    "void main() { \n"
    "gl_FragColor = texture2D(baseSampler, gl_TexCoord[0].st); \n}"
    ;
}

void renderData_t::endOfFrame(){
    printf("gpgpusim: end of frame %u\n", getCurrentFrame());
    incCurrentFrame();
}

void renderData_t::finilizeCurrentDraw() {
    printf("gpgpusim: end of drawcall %u, ", getDrawcallNum());
    if (!GPGPUSimSimulationActive()){
       if(DRAW_NON_SIM_FRAME){ 

          // writeDrawBuffer("post");
       }
        printf("not simulated!\n");
    }
    else printf("simulated!\n");
    incDrawcallNum(); 
}

const char* renderData_t::getCurrentShaderId(int shaderType) {
    if (shaderType == VERTEX_PROGRAM)
        return (const char*)(m_drawcall_num * 2);
    if (shaderType == FRAGMENT_PROGRAM)
        return (const char*)(m_drawcall_num * 2 + 1);
    assert(0);
}

void renderData_t::generateVertexCode() {
    bool programFound = false;
    
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
        for (unsigned shadersCoutner = 0; shadersCoutner < getMesaCtx()->Shader.ActiveProgram->NumShaders; shadersCoutner++) {
            if (getMesaCtx()->Shader.ActiveProgram->Shaders[shadersCoutner]->Type == GL_VERTEX_SHADER
                    && getMesaCtx()->Shader.ActiveProgram->VertexProgram == getMesaCtx()->VertexProgram._Current) {
                vertex_glsl << getMesaCtx()->Shader.ActiveProgram->Shaders[shadersCoutner]->Source;
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
    system(command.c_str());
}

void renderData_t::generateFragmentCode(DepthSize dbSize) {
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
        for (unsigned shadersCoutner = 0; shadersCoutner < getMesaCtx()->Shader.ActiveProgram->NumShaders; shadersCoutner++) {
            if (getMesaCtx()->Shader.ActiveProgram->Shaders[shadersCoutner]->Type == GL_FRAGMENT_SHADER
                    && getMesaCtx()->Shader.ActiveProgram->FragmentProgram == getMesaCtx()->FragmentProgram._Current) {
                fragment_glsl << getMesaCtx()->Shader.ActiveProgram->Shaders[shadersCoutner]->Source;
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
    system(command.c_str());
}

void renderData_t::addFragmentsSpan(SWspan* span) {
    if (!GPGPUSimSimulationActive()) {
        std::cerr<<"Error: addFragmentsSpan called when simulation is not active "<<std::endl;
        exit(-1);
    }

    //DPRINTF(MesaGpgpusim, "Buffering span fragments of the current primitive, fragments count=%d\n", span->end);
    
    for (int frag = 0; frag < span->end; frag++) {
        fragmentData_t fragment;
        for (int attrib = 0; attrib < FRAG_ATTRIB_MAX; attrib++) {
            fragment.attribs[attrib][0] = span->array->attribs[attrib][frag][0];
            fragment.attribs[attrib][1] = span->array->attribs[attrib][frag][1];
            fragment.attribs[attrib][2] = span->array->attribs[attrib][frag][2];
            fragment.attribs[attrib][3] = span->array->attribs[attrib][frag][3];
        }
        fragment.intPos[0] = (unsigned) round(fragment.attribs[FRAG_ATTRIB_WPOS][0]);
        fragment.intPos[1] = (unsigned) round(fragment.attribs[FRAG_ATTRIB_WPOS][1]);
        fragment.intPos[2] = (unsigned) round(fragment.attribs[FRAG_ATTRIB_WPOS][2]*(pow(2, (int)m_depthSize*8)) - 1);
        addFragment(fragment);
    }
}

/*void renderData_t::startPrimitive(){
    addPrimitive();
}*/

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

std::string renderData_t::getShaderPTXInfo(std::string arbFileName, std::string functionName) {
    //static long long unsigned ptxInfoFileNum =0;
    std::ifstream arbFile(arbFileName);
    assert(arbFile.is_open());
    std::string arbLine;
    unsigned usedRegs=0;
    while (arbFile.good()) {
        getline(arbFile, arbLine);
        size_t rRegsLocation = arbLine.find("R-regs");
        if (rRegsLocation != std::string::npos) {
            size_t commaLocation = arbLine.find(",");
            std::string numberOfRegistersString = arbLine.substr(commaLocation + 2, rRegsLocation - commaLocation - 3);
            usedRegs = atoi(numberOfRegistersString.c_str());
            if (usedRegs == 0) usedRegs = 1; //minimum number of used registers
            usedRegs *= 4; //the number given so far is for vector registers so we mul by 4
        }
    }
    
    std::stringstream ptxInfo;
    
    //todo we only now do registers
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

    for (int i = 0; i < bufferSize; i++) {
        bufferImage << buffer[i];
    }
    bufferImage.close();
    std::string convertCommand = "convert -depth " + std::to_string(depth) + " -size " + std::to_string(w) + "x" + std::to_string(h) 
                                 + " " + ss.str() + " " + ss.str() + ".jpg";
    system(convertCommand.c_str());
    system(std::string("rm " + ss.str()).c_str());
}

byte* renderData_t::setRenderBuffer(){
    gl_renderbuffer *rb = m_mesaCtx->DrawBuffer->_ColorDrawBuffers[0];
    
    GLenum bufferFormat = rb->InternalFormat;
    if(bufferFormat!=GL_RGBA){
       printf("Error: unsupported buffer format %x \n", bufferFormat);
       abort();
    }

    m_colorBufferByteSize = rb->Height * rb->Width * sizeof (byte)*VECTOR_SIZE; //todo: use the actual number of channels
    m_bufferWidthx4 = rb->Width * 4;
    m_bufferWidth = rb->Width;
    m_bufferHeight = rb->Height;
    //DPRINTF(MesaGpgpusim, "gpgpusim-graphics: fb height=%d width=%d\n",m_bufferHeight, m_bufferWidth);
    m_mesaColorBuffer = rb;
    m_colorBufferPutRow = rb->PutRow;
    GetRowFunc GetRow = rb->GetRow;
    byte(*tempBuffer)[4] = new byte [m_colorBufferByteSize / VECTOR_SIZE][4]; //RGBA
    for (int i = 0; i < getBufferHeight(); i++)
        GetRow(getMesaCtx(), rb, m_bufferWidth, 0, i, tempBuffer + ((m_bufferHeight - i - 1) * m_bufferWidth));

    //using BGRA order
    byte * tempBuffer2 = new byte [m_colorBufferByteSize];
    for (int i = 0; i < (m_colorBufferByteSize / 4); i++) {
        tempBuffer2[i * 4 + 2] = tempBuffer[i][0];
        tempBuffer2[i * 4 + 1] = tempBuffer[i][1];
        tempBuffer2[i * 4 + 0] = tempBuffer[i][2];
        tempBuffer2[i * 4 + 3] = tempBuffer[i][3];
    }

    delete [] tempBuffer;

    return tempBuffer2;
}

byte* renderData_t::setDepthBuffer(DepthSize activeDepthSize, DepthSize actualDepthSize){
    gl_renderbuffer *rb = getMesaCtx()->DrawBuffer->_DepthBuffer;

    uint32_t dbSize = (uint32_t) activeDepthSize; 
    uint32_t mesaDbSize = (uint32_t) actualDepthSize; 
    unsigned buffSize = rb->Height * rb->Width;
    m_depthBufferSize = buffSize * sizeof (byte)* dbSize;
    m_depthBufferWidth = rb->Width;
    m_depthBufferHeight = rb->Height;
    m_depthSize = activeDepthSize;
    m_mesaDepthSize = actualDepthSize;
    
    //DPRINTF(MesaGpgpusim, "gpgpusim-graphics: fb height=%d width=%d\n",m_bufferHeight, m_bufferWidth);
    m_mesaDepthBuffer = rb;
    m_depthBufferPutRow = rb->PutRow;
    GetRowFunc GetRow = rb->GetRow;
    uint32_t mesaDepthBufferSize = buffSize * sizeof (byte)* mesaDbSize;
    byte *tempBuffer  = new byte [mesaDepthBufferSize];
    for (int i = 0; i < m_depthBufferHeight; i++){
        unsigned xpos = ((m_depthBufferHeight - i - 1)* m_depthBufferWidth * mesaDbSize);
        GetRow(getMesaCtx(), rb, m_depthBufferWidth, 0, i, tempBuffer + xpos);
    }
    
    //convertng the buffer format from Z16 to Z32 in case the buffers are of different sizes
    //this case happes when in-shader depth is used with a 16 bit mesa depth buffer
    assert((actualDepthSize == activeDepthSize) or ((actualDepthSize == DepthSize::Z16) and (activeDepthSize == DepthSize::Z32)));
    if((actualDepthSize == DepthSize::Z16) and (activeDepthSize == DepthSize::Z32))
    {
       uint32_t * sbuf = new uint32_t[buffSize];
       uint16_t * mesaBuf = (uint16_t*) tempBuffer;
       for(unsigned i=0; i < buffSize; i++){
          sbuf[i] = mesaBuf[i] << 16;
       }
       delete [] tempBuffer;
       tempBuffer = (byte*) sbuf;
    }

    return tempBuffer;
}

void renderData_t::initializeCurrentDraw(gl_context * ctx) {

    if (ctx->Shader.ActiveProgram == NULL) {
       printf("no active shader \n");
    } else {
       printf("there is an active shader \n");
    }
    if (!GPGPUSimSimulationActive()) {
        std::cerr << "gpgpusim-graphics: Error, initializeCurrentDraw called when simulation is not active " << std::endl;
        exit(-1);
    }

    //remove or fix me
    if (!ctx->Shader.ActiveProgram) {
       //no active program skip
       return;
    }
   //

    DPRINTF(MesaGpgpusim, "initializing a draw call \n");
    m_mesaCtx = ctx;

    byte* currentBuffer = setRenderBuffer(); 

    m_useDefaultShaders = false;

    if (ctx->Shader.ActiveProgram == NULL) {
        if (ctx->VertexProgram._MaintainTnlProgram and ctx->FragmentProgram._MaintainTexEnvProgram)
            m_useDefaultShaders = true;
        else {
            std::cout << "GPGUSIM Graphics: no shader is provided and cannot use defualt shaders!\n" << std::endl;
            exit(-1);
        }
    }
    
    DepthSize activeDepthSize;
    DepthSize trueDepthSize;
    gl_renderbuffer *rb = getMesaCtx()->DrawBuffer->_DepthBuffer;
    
    if(isDepthTestEnabled()){
       if(rb->Format==MESA_FORMAT_Z32){
          activeDepthSize = trueDepthSize = DepthSize::Z32;
       } else if(rb->Format==MESA_FORMAT_Z16){
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

    //generate PTX code from ARB
    generateVertexCode();
    generateFragmentCode(activeDepthSize);
    
    std::string drawCallN = std::to_string(getDrawcallNum());
    std::string vertexARBFile = vARBPrfx + drawCallN;
    std::string vertexPTXFile = vPTXPrfx +drawCallN+".ptx";
    std::string fragmentARBFile = fARBPrfx + drawCallN;
    std::string fragmentPTXFile = fPTXPrfx +drawCallN+".ptx"; 
    
    //todo: unregister the shader when we are done. 
    //however currently our cuda implementation doesn't implement __cudaUnregisterFatBinary
    
    //todo: remove the header from ptx code generation, fix function names and add a header here 
    //get the fatbin
    void* cudaFatBin = getShaderFatBin(vertexPTXFile, fragmentPTXFile);
    std::string vertexPtxInfo = getShaderPTXInfo(vertexARBFile, getCurrentShaderName(VERTEX_PROGRAM));
    std::string fragmentPtxInfo = getShaderPTXInfo(fragmentARBFile, getCurrentShaderName(FRAGMENT_PROGRAM));
    
    std::string ptxInfoFileName = fPtxInfoPrfx+std::to_string(getDrawcallNum());
    std::ofstream ptxInfoFile(ptxInfoFileName.c_str());
    assert(ptxInfoFile.is_open());
    ptxInfoFile<< vertexPtxInfo + fragmentPtxInfo;
    ptxInfoFile.close();
    void ** fatCubinHandle = graphicsRegisterFatBinary(cudaFatBin, ptxInfoFileName.c_str(), &m_sShading_info.allocAddr);
    assert(m_sShading_info.allocAddr != NULL); //we always have some constants in the shaders
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
    
    
    assert(getDeviceData() == NULL);
    m_depthBuffer = NULL;
    if(isDepthTestEnabled()){
        m_depthBuffer = setDepthBuffer(activeDepthSize, trueDepthSize);
        graphicsMalloc((void**) &m_deviceData, m_colorBufferByteSize + m_depthBufferSize); //for color and depth

        /*printf("color buffer start=%llx,  end=%llx, depthBuffer start=%llx, end=%llx\n",
              m_deviceData, m_deviceData + m_colorBufferByteSize-1,
              m_deviceData + m_colorBufferByteSize, m_deviceData + m_colorBufferByteSize + m_depthBufferSize-1);*/

        graphicsMemcpy(m_deviceData + m_colorBufferByteSize,
                m_depthBuffer, m_depthBufferSize, graphicsMemcpyHostToSim);
    } else {
        graphicsMalloc((void**) &m_deviceData, m_colorBufferByteSize);
    }
    
    graphicsMemcpy(m_deviceData, currentBuffer, getColorBufferByteSize(), graphicsMemcpyHostToSim);

    
    writeDrawBuffer("pre", currentBuffer,  m_colorBufferByteSize, m_bufferWidth, m_bufferHeight, "bgra", 8);
    delete [] currentBuffer;

    if(m_depthBuffer!=NULL) {
       writeDrawBuffer("pre_depth", m_depthBuffer,  m_depthBufferSize, m_bufferWidth, m_bufferHeight, "a", 8*(int)activeDepthSize);
    }

    //vertexStageData = stage;
    //setAllTextures(lastFatCubinHandle);
    //copyStateData(lastFatCubinHandle);
}

void renderData_t::setAllTextures(void ** fatCubinHandle) {
    //initializing data that is used for calling gpgpusim
    for (int i = 0; i < MAX_COMBINED_TEXTURE_IMAGE_UNITS; i++) {
        gl_texture_object *texObject = getMesaCtx()->Texture.Unit[i]._Current;
        if (!texObject) continue;
        gl_texture_image* texImage = texObject->Image[0][texObject->BaseLevel];
        setTextureUnit(fatCubinHandle,i, texImage->TexFormat, texImage->Height, texImage->Width, (unsigned char*) texImage->Data,
                texObject->Sampler.WrapS, texObject->Sampler.WrapT, texObject->Sampler.MinFilter, texObject->Sampler.MagFilter, texObject->Target);
    }
}

void renderData_t::setTextureUnit(void** fatCubinHandle, int textureNumber, int textureType, int h, int w, byte * inData,
        int wrapS, int wrapT, int minFilter, int magFilter, GLenum target) {

    assert(textureNumber<MAX_COMBINED_TEXTURE_IMAGE_UNITS);
    assert(target==GL_TEXTURE_2D);//what we support so far
    int imgSize = h * w;
    int bytesPerTexel;
    byte* inTexData = inData;
    bool inTexDataNew = false;

    switch (textureType) {
        case MESA_FORMAT_RGB888:
           inTexDataNew = true;
           inTexData = Utils::RGB888_to_RGBA888(inTexData, imgSize);
           textureType = MESA_FORMAT_RGBA8888;
           bytesPerTexel = 4;
           break;
        case MESA_FORMAT_RGBA8888:
           bytesPerTexel = 4;
           break;
        case MESA_FORMAT_A8:
           bytesPerTexel = 1;
           break;
        default:
           std::cout << "GPGPUSIM: Unsupported texture format" << std::endl;
           abort();
    }
    assert(bytesPerTexel <= 4 and bytesPerTexel >= 1);

    //converting to the BGRA format
    int size = imgSize * sizeof (byte) * bytesPerTexel;
    byte * data = new byte[size];
    for (int i = 0; i < size; i += bytesPerTexel) {
       switch(bytesPerTexel){
          case 4: data[i + 3] = inTexData[i + bytesPerTexel - 4];
          case 3: data[i + 2] = inTexData[i + bytesPerTexel - 3];
          case 2: data[i + 1] = inTexData[i + bytesPerTexel - 2];
          case 1: data[i + 0] = inTexData[i + bytesPerTexel - 1];
       }
    }

    if(inTexDataNew)
       delete [] inTexData;

    cudaChannelFormatDesc channelDesc;
    std::string typeEx;
    switch (textureType) {
        case MESA_FORMAT_A8:
            channelDesc.x = 0;
            channelDesc.y = 0;
            channelDesc.z = 0;
            channelDesc.w = 8;
            channelDesc.f = cudaChannelFormatKindUnsigned;
            typeEx = "a";
            break;
        case MESA_FORMAT_RGBA8888:
            channelDesc.x = 8;
            channelDesc.y = 8;
            channelDesc.z = 8;
            channelDesc.w = 8;
            channelDesc.f = cudaChannelFormatKindUnsigned;
            typeEx = "rgba";
            break;
        default: std::cout << "GPGPUSIM: Unsupported number of channels ! " << std::endl;
            abort();
    }

    assert (textureArray[textureNumber] == NULL);
    graphicsMallocArray(&textureArray[textureNumber], &channelDesc, w, h, data, size);
    graphicsMemcpy(textureArray[textureNumber]->devPtr, data, size, graphicsMemcpyHostToSim);
    //textureArray[textureNumber]->devPtr32 = *((int*)(&textureArray[textureNumber]->devPtr));
    printf("texture %d ptr32=%x\n", textureNumber, textureArray[textureNumber]->devPtr32);
    printf("texture %d ptr=%x\n", textureNumber, textureArray[textureNumber]->devPtr);
    writeTexture(data, size, textureNumber, h , w, typeEx);

    delete [] data;
    data = NULL;
    
    textureReference * textureReferencePtr = new textureReference();
    textureRefs.push_back(textureReferencePtr);
    
    if (wrapS == GL_CLAMP_TO_EDGE) textureReferencePtr->addressMode[0] = cudaAddressModeClamp;
    else if (wrapS == GL_REPEAT) textureReferencePtr->addressMode[0] = cudaAddressModeWrap;
    else {
        std::cout << "GPGPUSIM: Unsupported WarpS mode" << std::endl;
        abort();
    }

    if (wrapT == GL_CLAMP_TO_EDGE) textureReferencePtr->addressMode[1] = cudaAddressModeClamp;
    else if (wrapT == GL_REPEAT) textureReferencePtr->addressMode[1] = cudaAddressModeWrap;
    else {
        std::cout << "GPGPUSIM: Unsupported WarpT mode" << std::endl;
        abort();
    }

    if(minFilter == GL_LINEAR_MIPMAP_LINEAR){
       minFilter = GL_LINEAR;
       warn_once("Ignoring GL_LINEAR_MIPMAP_LINEAR texture, using GL_LINEAR\n");
       //TODO: add mipmaps
    }

    assert((minFilter== GL_NEAREST or minFilter == GL_LINEAR) 
            and (magFilter== GL_NEAREST or magFilter == GL_LINEAR));

    if(minFilter != magFilter){
       warn("Using GL_LINEAR for min and mag filters!\n");
       //minFilter = magFilter = GL_LINEAR;
       minFilter = magFilter = GL_NEAREST;
    }

    if (minFilter == GL_LINEAR and magFilter == GL_LINEAR)
        textureReferencePtr->filterMode = cudaFilterModeLinear;
    else textureReferencePtr->filterMode = cudaFilterModePoint;
    textureReferencePtr->normalized = true;
    std::string textureName = "textureReference" + std::to_string((long long unsigned)textureNumber) + "_2D";
    //todo set the normalization and the dim flags properly
    //todo recheck how the next two calls interact in gpgpusim.
    graphicsRegisterTexture(fatCubinHandle, textureReferencePtr,
                (const void**) textureName.c_str(), textureName.c_str(), 2, 1, 0);
    graphicsBindTextureToArray(textureReferencePtr, textureArray[textureNumber], &channelDesc);
}

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
    
    TNLcontext *tnl = TNL_CONTEXT(getMesaCtx());
    vertex_buffer *VB = &tnl->vb;
    unsigned vertsCount = VB->Count;
    printf("starting vertex shading: vertices = %d\n", vertsCount);
    const unsigned verticesAttribsFloatsCount = vertsCount * VERT_ATTRIB_MAX * 4;
    float hostVertsAttribs[verticesAttribsFloatsCount];

    /* the vertex array case */
    for (unsigned i = 0; i < vertsCount; i++) {
        for (unsigned attr = 0; attr < VERT_ATTRIB_MAX; attr++) {
            if (getMesaCtx()->VertexProgram._Current->Base.InputsRead & (1 << attr)) {
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
                    //if(getMesaCtx()->VertexProgram._Current->Base.NumAttributes == 0) break;
                    for (int l = 0; l < getMesaCtx()->VertexProgram._Current->Base.Attributes->NumParameters; l++)
                        if (getMesaCtx()->VertexProgram._Current->Base.Attributes->Parameters[l].StateIndexes[0] == mesaNumber) {
                            mesaName = getMesaCtx()->VertexProgram._Current->Base.Attributes->Parameters[l].Name;
                            break;
                        }

                    int arbNumber = generalAttribPosition[mesaName];
                    place = &hostVertsAttribs[i * VERT_ATTRIB_MAX * 4 + (VERT_ATTRIB_GENERIC0 + arbNumber)*4];
                }
                COPY_CLEAN_4V(place, size, data);
            }
        }
    }

    /*DPRINTF(MesaGpgpusim, "vertex shader input attribs, verticesAttribsFloatsCount = %d \n", verticesAttribsFloatsCount);
    for (int vert = 0; vert < vertsCount; vert++) {
        for (int attrib = 0; attrib < VERT_ATTRIB_MAX; attrib++) {
            for (int i = 0; i < 4; i++) {
                int index = (vert * VERT_ATTRIB_MAX * 4) + (attrib * 4) + i;
                DPRINTF(MesaGpgpusim, "hostVertsAttribs[%d][%s][%d] = %f \n", vert, VERT_ATTRIB_NAMES[attrib], i, hostVertsAttribs[index]);
            }
        }
    }*/
    
    assert(m_sShading_info.deviceVertsAttribs == NULL);
    graphicsMalloc((void**) &m_sShading_info.deviceVertsAttribs, verticesAttribsFloatsCount * sizeof (float));
    graphicsMemcpy(m_sShading_info.deviceVertsAttribs, hostVertsAttribs, verticesAttribsFloatsCount * sizeof (float), graphicsMemcpyHostToSim);
    
    m_sShading_info.primCountMap = new unsigned [VB->PrimitiveCount];
    memset(m_sShading_info.primCountMap, 0, sizeof(unsigned)*VB->PrimitiveCount);
    
    
    m_sShading_info.render_init = false;
    assert(m_sShading_info.vertCodeAddr == NULL);

    unsigned threadsPerBlock = 256; //TODO: add it to options,
    unsigned numberOfBlocks = (vertsCount + threadsPerBlock -1) / threadsPerBlock;
    assert(graphicsConfigureCall(numberOfBlocks, threadsPerBlock, 0, 0) == cudaSuccess);
    assert(graphicsSetupArgument((void*)&m_sShading_info.deviceVertsAttribs, sizeof(float*), 0/*offset*/) == cudaSuccess);
    assert(graphicsLaunch(getCurrentShaderId(VERTEX_PROGRAM), &m_sShading_info.vertCodeAddr) == cudaSuccess); 
    
    m_sShading_info.launched_threads = numberOfBlocks * threadsPerBlock;
    printf("launched vertex threads = %d\n", m_sShading_info.launched_threads);
    m_sShading_info.completed_threads = 0;
    m_sShading_info.currentPass = stage_shading_info_t::GraphicsPass::Vertex;
   
    printf("do vertex inc ready sockets\n");fflush(stdout);
    SocketStream::incReadySockets(-1, true);
    //waiting the fragment shader to finish
    printf("first vertex lock\n");fflush(stdout);
    vertexFragmentLock.lock();
    printf("second vertex lock\n");fflush(stdout);
    vertexFragmentLock.lock(); //wait till end of fragment shader to unlock
    printf("unlock vertex lock\n");fflush(stdout);
    vertexFragmentLock.unlock();
    fflush(stdout);
    //int dum = 0;
    /*while(m_sShading_info.currentPass != stage_shading_info_t::GraphicsPass::NONE){
       dum++;
    }*/
    endDrawCall();
    return GL_TRUE;
}

unsigned int renderData_t::doFragmentShading() {
    if(!GPGPUSimSimulationActive()){
        std::cerr<<"Error: doFragmentShading called when simulation is not active "<<std::endl;
        exit(-1);
    }
   
    unsigned totalFragsCount = 0;
    for(int prim =0; prim < drawPrimitives.size(); prim++){
       totalFragsCount+= drawPrimitives[prim].size();
    }

    if (totalFragsCount == 0)
       return 0;
    m_sShading_info.completed_threads = 0;
    //uint64_t streamId = 0; //TODO: change
    assert(m_sShading_info.fragCodeAddr == NULL);
    for(int prim =0; prim < drawPrimitives.size(); prim++){
       unsigned fragmentsCount =  drawPrimitives[prim].size();
       if(fragmentsCount == 0) continue;
       unsigned threadsPerBlock = 512;
       unsigned numberOfBlocks = (fragmentsCount + threadsPerBlock -1) / threadsPerBlock;
       m_sShading_info.launched_threads += numberOfBlocks*threadsPerBlock;

       m_sShading_info.cudaStreams.push_back(cudaStream_t());
       graphicsStreamCreate(&m_sShading_info.cudaStreams.back()); 
      
       DPRINTF(MesaGpgpusim, "starting fragment shader, fragments = %d on stream %x\n", fragmentsCount,  m_sShading_info.cudaStreams.back());
       printf("starting fragment shader, fragments = %d, total=%d\n", fragmentsCount, m_sShading_info.launched_threads);

       mapTileStream_t map;
       map.primId =  prim;
       m_sShading_info.cudaStreamTiles[(uint64_t)(m_sShading_info.cudaStreams.back())] = map;

       assert(numberOfBlocks);
       //cudaStream_t cudaStream;
       //graphicsStreamCreate(&cudaStream);
       //addStreamCall((void*) streamId++, threadsPerBlock * numberOfBlocks); //TODO: change, related to above
       //todo
       byte* arg= getDeviceData() + getColorBufferByteSize();
       //assert(graphicsConfigureCall(numberOfBlocks, threadsPerBlock, 0, 0) == cudaSuccess);

       assert( graphicsConfigureCall(numberOfBlocks, threadsPerBlock, 0, m_sShading_info.cudaStreams.back())
          == cudaSuccess);
       assert(graphicsSetupArgument((void*) &arg, sizeof (byte*), 0/*offset*/) == cudaSuccess);
       assert(graphicsLaunch(getCurrentShaderId(FRAGMENT_PROGRAM), &m_sShading_info.fragCodeAddr) == cudaSuccess);
    }
    
    assert(m_sShading_info.fragCodeAddr != NULL);

    m_sShading_info.currentPass = stage_shading_info_t::GraphicsPass::Fragment;
    return m_sShading_info.launched_threads;
}

void renderData_t::putDataOnColorBuffer() {
    //copying the result render buffer to mesa
    byte * tempBuffer = new byte [getColorBufferByteSize()];
    graphicsMemcpy(tempBuffer, m_deviceData, getColorBufferByteSize(), graphicsMemcpySimToHost);

    byte(*tempBuffer2)[4] = new byte [getColorBufferByteSize() / VECTOR_SIZE][VECTOR_SIZE];

    for (int i = 0; i < (getColorBufferByteSize() / VECTOR_SIZE); i++) {
        tempBuffer2[i][0] = tempBuffer[i * VECTOR_SIZE + 2]; //bgra to rgba
        tempBuffer2[i][1] = tempBuffer[i * VECTOR_SIZE + 1];
        tempBuffer2[i][2] = tempBuffer[i * VECTOR_SIZE + 0];
        tempBuffer2[i][3] = tempBuffer[i * VECTOR_SIZE + 3];
    }

    writeDrawBuffer("post", (byte*)tempBuffer2, getColorBufferByteSize(), m_bufferWidth, m_bufferHeight, "rgba", 8);
    
    for (int i = 0; i < getBufferHeight(); i++)
        m_colorBufferPutRow(getMesaCtx(), getMesaBuffer(), getBufferWidth(), 0, getBufferHeight() - i - 1, tempBuffer2 + (i * getBufferWidth()), NULL);

    delete [] tempBuffer;
    delete [] tempBuffer2;
}

//copying the result depth buffer to mesa
void renderData_t::putDataOnDepthBuffer(){
    byte * tempBuffer = new byte [m_depthBufferSize];
    graphicsMemcpy(tempBuffer, m_deviceData + m_colorBufferByteSize,
          m_depthBufferSize, graphicsMemcpySimToHost);

    writeDrawBuffer("post_depth", tempBuffer,  m_depthBufferSize, m_bufferWidth, m_bufferHeight, "a", 8*(int)m_depthSize);

    assert((m_depthSize == m_mesaDepthSize) or ((m_mesaDepthSize == DepthSize::Z16) and (m_depthSize == DepthSize::Z32)));
    byte* readDepth = tempBuffer;
    if((m_mesaDepthSize == DepthSize::Z16) and (m_depthSize == DepthSize::Z32)){
       int pixelSize = getPixelBufferSize();
       uint16_t* depth16 = new uint16_t[pixelSize];
       readDepth = (byte*) depth16;
       uint32_t* depth32 = (uint32_t*) tempBuffer;
       for(int i=0; i<pixelSize; i++){
          depth16[i] = (uint16_t) (depth32[i] >> 16); //loose precision 
       } 
       delete [] tempBuffer;
    }
    
    gl_renderbuffer *rb = getMesaCtx()->DrawBuffer->_DepthBuffer;
    for (int i = 0; i < m_depthBufferHeight; i++){
        unsigned xpos = ((m_depthBufferHeight - i - 1)* m_depthBufferWidth * ((int)m_mesaDepthSize));
        m_depthBufferPutRow(getMesaCtx(), rb, m_depthBufferWidth, 0, i, readDepth + xpos, NULL);
    }

    delete [] readDepth;
}

gl_state_index renderData_t::getParamStateIndexes(gl_state_index index) {
    gl_program_parameter_list * paramList = getMesaCtx()->VertexProgram._Current->Base.Parameters;
    for (int i = 0; i < paramList->NumParameters; i++) {
        if (paramList->Parameters[i].Type == PROGRAM_STATE_VAR) {
            //DPRINTF(MesaGpgpusim, "state index %d = %d and the requested index is %d\n",i,paramList->Parameters[i].StateIndexes[0], index);
            if(paramList->Parameters[i].StateIndexes[0]==index)
                return paramList->Parameters[i].StateIndexes[0];
        }
    }
    
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
            
            GLfloat mvpMatrix [VECTOR_SIZE * VECTOR_SIZE];
            _mesa_fetch_state(getMesaCtx(), state, mvpMatrix);
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
            GLfloat modelViewMatrix [VECTOR_SIZE * VECTOR_SIZE];
            _mesa_fetch_state(getMesaCtx(), state, modelViewMatrix);
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
            GLfloat projectionMatrix [VECTOR_SIZE * VECTOR_SIZE];
            _mesa_fetch_state(getMesaCtx(), state, projectionMatrix);
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
            COPY_4V(vectorValue, getMesaCtx()->Light.Light[light].EyePosition);
            graphicsMemcpyToSymbol((char*)strHash(varName),
                    vectorValue,
                    sizeof (GLfloat)*VECTOR_SIZE,
                    sizeof (GLfloat)*(light * VECTOR_SIZE),
                    graphicsMemcpyHostToSim);
        }
    }

    //UNIFORM VARIABLES
    if (getMesaCtx()->Shader.ActiveProgram) {
        {
            std::string varName = "vertex_program_locals";
            graphicsRegisterVar(fatCubinHandle, (char*)strHash(varName), (char*) varName.c_str(),
            varName.c_str(), 0, MAX_UNIFORMS* float_4vectorSize, 1, 0);
            
            int fillIndex = 0;
            //we do this as parameters contains state parameters which we copy above in isolation
            for (int i = getMesaCtx()->Shader.ActiveProgram->Uniforms->NumUniforms - 1; i >= 0; i--) {
            //for (int i = 0; i< getMesaCtx()->Shader.ActiveProgram->Uniforms->NumUniforms; i++) {
                if (getMesaCtx()->Shader.ActiveProgram->Uniforms->Uniforms[i].VertPos >= 0) {
                    struct gl_program *prog = &getMesaCtx()->Shader.ActiveProgram->VertexProgram->Base;
                    int paramPos = getMesaCtx()->Shader.ActiveProgram->Uniforms->Uniforms[i].VertPos;

                    const struct gl_program_parameter *param = &prog->Parameters->Parameters[paramPos];
                    if (param->Type != PROGRAM_UNIFORM) continue; //we deal with different types in a different way (as texture of type "PROGRAM_SAMPLER")
                    int numberOfElements = param->Size / VECTOR_SIZE;
                    if (numberOfElements == 0) numberOfElements = 1; //at least 1 vector should be copied for smaller uniforms
                    
                    /*DPRINTF(MesaGpgpusim, "placing uniform:%d, size:%d at location:%d with values:\n", i, param->Size, fillIndex);
                    for (int r = 0; r < numberOfElements; r++){
                        DPRINTF(MesaGpgpusim, "%s: ", prog->Parameters->Parameters[paramPos + r].Name);
                        for(int e = 0; e < VECTOR_SIZE; e++){
                            DPRINTF(MesaGpgpusim, "%f ", prog->Parameters->ParameterValues[paramPos + r][e]);
                        }
                        DPRINTF(MesaGpgpusim, "\n");
                    }*/
                    
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
            for (int i = getMesaCtx()->Shader.ActiveProgram->Uniforms->NumUniforms - 1; i >= 0; i--) {
                if (getMesaCtx()->Shader.ActiveProgram->Uniforms->Uniforms[i].FragPos >= 0) {
                    struct gl_program *prog = &getMesaCtx()->Shader.ActiveProgram->FragmentProgram->Base;
                    int paramPos = getMesaCtx()->Shader.ActiveProgram->Uniforms->Uniforms[i].FragPos;

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
}

bool renderData_t::isDepthTestEnabled(){
    if (g_renderData.getMesaCtx()->Depth.Test != 0)
        return true;
    return false;
}

bool renderData_t::isBlendingEnabled() {
    static int lastState = 1;
    if (lastState != (int) _mesa_IsEnabled(GL_BLEND)) {
         //DPRINTF(MesaGpgpusim, "BLENDING STATE CHANGED NOW IS = %d\n", (int) _mesa_IsEnabled(GL_BLEND));
    }
    lastState = (int) _mesa_IsEnabled(GL_BLEND);
    
    if (g_renderData.getMesaCtx()->Color.BlendEnabled & 1) {
            return true;
    }
    return false;
}

void renderData_t::getBlendingMode(GLenum * src, GLenum * dst, GLenum* srcAlpha, GLenum * dstAlpha, GLenum* eqnRGB, GLenum* eqnAlpha, GLfloat * blendColor){
    *src = getMesaCtx()->Color.Blend[0].SrcRGB;
    *dst = getMesaCtx()->Color.Blend[0].DstRGB;
    *srcAlpha = getMesaCtx()->Color.Blend[0].SrcA;
    *dstAlpha = getMesaCtx()->Color.Blend[0].DstA;
    *eqnRGB = getMesaCtx()->Color.Blend[0].EquationRGB;
    *eqnAlpha = getMesaCtx()->Color.Blend[0].EquationA;
    memcpy(blendColor,&getMesaCtx()->Color.BlendColor,sizeof(GLfloat)*VECTOR_SIZE);
}

void renderData_t::writeVertexResult(unsigned threadID, unsigned resAttribID, unsigned attribIndex, float data){
    //DPRINTF(MesaGpgpusim, "writing vs result at thread=%d attrib=[%d][%d]=%f\n", threadID, resAttribID, attribIndex, data);
    vertexStageData->results[resAttribID].data[threadID][attribIndex] = data;
}

void renderData_t::endVertexShading(CudaGPU * cudaGPU){
    //DPRINTF(MesaGpgpusim, "gpgpusim-graphics: starting prim %d\n", 0);
    TNLcontext *tnl = TNL_CONTEXT(getMesaCtx());
    vertex_buffer *VB = &tnl->vb;
    copy_vp_results(m_mesaCtx, VB, vertexStageData, m_mesaCtx->VertexProgram._Current);

    for (GLuint primId = 0; primId < VB->PrimitiveCount; primId++)
        do_ndc_cliptest(m_mesaCtx, vertexStageData, primId);
   
    m_sShading_info.renderFunc = init_run_render(m_mesaCtx);

    //get fragments of the current batch of vertices 
    for (GLuint primId = 0; primId < VB->PrimitiveCount; primId++)
       run_render_prim(m_mesaCtx, m_sShading_info.renderFunc, VB, primId);
   
    //this will add current prim fragments
    finalize_run_render(m_mesaCtx);

    if(m_inShaderDepth or not(isDepthTestEnabled())){
       //now sorting them in raster order
       sortFragmentsInRasterOrder(getTileH(), getTileW(),getBlockH(), getBlockW(), HorizontalRaster); //BlockedHorizontal);

       if(0 == doFragmentShading()){
          //no fragment shader
          endFragmentShading();
       }
    } else {
       runEarlyZ(cudaGPU, getTileH(), getTileW(),getBlockH(), getBlockW(), HorizontalRaster); // BlockedHorizontal);
    }
}

void renderData_t::endFragmentShading() {
    printf("end fragment shading\n");
    SocketStream::decReadySockets(-1, true); 
    m_sShading_info.currentPass = stage_shading_info_t::GraphicsPass::NONE;
    printf("unlock frag lock\n"); fflush(stdout);
    vertexFragmentLock.unlock();
    //printf("set current pass to none\n");
    printf("waiting vertex all render socket ready\n");
    while(!SocketStream::allRenderSocketsReady())
    {
       std::this_thread::sleep_for (std::chrono::milliseconds(10));
      //printf("end frag shader waiting\n");
    }
}

void renderData_t::checkGraphicsThreadExit(void * kernelPtr, unsigned tid){
   printf("completed a thread pass =%d\n", m_sShading_info.currentPass);
   if(m_sShading_info.currentPass == stage_shading_info_t::GraphicsPass::NONE){
      //nothing to do
      return;
   } else if(m_sShading_info.currentPass == stage_shading_info_t::GraphicsPass::Vertex){
       m_sShading_info.completed_threads++;
       assert(m_sShading_info.completed_threads <= m_sShading_info.launched_threads);
       if(m_sShading_info.completed_threads == m_sShading_info.launched_threads){
          m_flagEndVertexShader = true;
       }
       printf("completed threads = %d out of %d\n", m_sShading_info.completed_threads,  m_sShading_info.launched_threads);
   } else  if(m_sShading_info.currentPass == stage_shading_info_t::GraphicsPass::Fragment){
      m_sShading_info.completed_threads++;
      assert(m_sShading_info.completed_threads <= m_sShading_info.launched_threads);

      printf("completed threads = %d out of %d\n", m_sShading_info.completed_threads,  m_sShading_info.launched_threads);
      if (m_sShading_info.completed_threads == m_sShading_info.launched_threads){
         if(m_inShaderDepth or !isDepthTestEnabled())
         {
            m_flagEndFragmentShader = true;
         } else {
            //only done if early-Z is also done
            m_flagEndFragmentShader = m_sShading_info.doneEarlyZ;
         }
      }
   }
}

void renderData_t::checkEndOfShader(CudaGPU * cudaGPU){
   if(m_flagEndVertexShader){ 
      m_sShading_info.launched_threads = 0; //reset
      endVertexShading(cudaGPU);
      m_flagEndVertexShader = false;
   }

   if(m_flagEndFragmentShader){
      endFragmentShading();
      m_flagEndFragmentShader = false;
   }

}

void renderData_t::doneEarlyZ(){
   m_sShading_info.doneEarlyZ = true;

   if(m_sShading_info.completed_threads == m_sShading_info.launched_threads){
      endFragmentShading();
      m_flagEndFragmentShader = false;
   } 
}



void renderData_t::launchFragmentTile(RasterTile * rasterTile, unsigned tileId){
   unsigned fragsCount = rasterTile->setActiveFragmentsIndices();

   printf( "Launching tile %d of fragments, active count=%d of of %d\n", tileId, fragsCount, rasterTile->size());

   if(fragsCount == 0){
      //not active fragments
      return;
   }

   DPRINTF(MesaGpgpusim, "Launching a tile of fragments, active count=%d of of %d\n", fragsCount, rasterTile->size());


   unsigned threadsPerBlock = 256; //TODO: add it to options, chunks used to distribute work
   unsigned numberOfBlocks = (rasterTile->size() + threadsPerBlock -1 ) / threadsPerBlock;

   m_sShading_info.cudaStreams.push_back(cudaStream_t());
   graphicsStreamCreate(&m_sShading_info.cudaStreams.back()); 


   byte* arg= getDeviceData() + getColorBufferByteSize();
   mapTileStream_t map;
   map.tileId = tileId;
   map.primId = rasterTile->primId;

   m_sShading_info.cudaStreamTiles[(uint64_t)(m_sShading_info.cudaStreams.back())] = map;

   uint64_t streamId = (uint64_t)m_sShading_info.cudaStreams.back();
   printf("running %d threads for  tile %d with %d fragments on stream %d\n", numberOfBlocks*threadsPerBlock, tileId, rasterTile->size(), streamId );
   //printf("running tile %d with %d fragments on stream %d\n", tileId, tcount, 0);
   assert( graphicsConfigureCall(numberOfBlocks, threadsPerBlock, 0, m_sShading_info.cudaStreams.back())  
   //assert( graphicsConfigureCall(numberOfBlocks, threadsPerBlock, 0, 0)  //0) cudaStream) 
      == cudaSuccess);
   assert(graphicsSetupArgument((void*) &arg, sizeof (byte*), 0/*offset*/) == cudaSuccess);
   assert(graphicsLaunch(getCurrentShaderId(FRAGMENT_PROGRAM), &m_sShading_info.fragCodeAddr) == cudaSuccess);
   assert(m_sShading_info.fragCodeAddr != NULL);

   //static uint64_t streamId = 0;
   m_sShading_info.launched_threads+= numberOfBlocks*threadsPerBlock;
   printf( "total launched threads = %d\n", m_sShading_info.launched_threads);
   m_sShading_info.earlyZTilesCounts.push_back(m_sShading_info.launched_threads);
   m_sShading_info.earlyZTilesIds.push_back(tileId);


   m_sShading_info.currentPass = stage_shading_info_t::GraphicsPass::Fragment;
   //fragmentShader << <numberOfBlocks, threadsPerBlock, 0, cudaStream >> >(getDeviceData() + getColorBufferByteSize());
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

