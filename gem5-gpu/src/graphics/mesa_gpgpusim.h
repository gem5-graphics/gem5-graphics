#ifndef __MESA_GPGPUSIM_H_
#define __MESA_GPGPUSIM_H_

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <mutex>
#include <map>
#include <GL/gl.h>
#include <unordered_set>

extern "C" {
#include "compiler/shader_enums.h"
#include "program/prog_statevars.h"
#include "mesa/main/config.h"
#include "main/mtypes.h"
#include "math/m_vector.h"
#include "pipe/p_state.h"
#include "tgsi/tgsi_exec.h"
#include "sp_tex_sample.h"
}

#define SKIP_API_GEM5
#include "api/cuda_syscalls.hh"
#include "graphics/gpgpusim_to_graphics_calls.h"


typedef unsigned char byte;
const int VECTOR_SIZE = 4;
const int QUAD_SIZE = 4;
const int CUDA_FLOAT_SIZE = 4;
class CudaGPU;

enum shaderType_t{
    NO_SHADER,
    VERTEX_PROGRAM,
    FRAGMENT_PROGRAM
};

//gpgpusim calls
extern "C" void gpgpusimLoadShader(int shaderType, std::string arbFile, std::string ptxFile);

//mesa calls we use
extern "C" {
  void _mesa_readpixels(struct gl_context *ctx,
                               GLint x, GLint y, GLsizei width, GLsizei height,
                               GLenum format, GLenum type,
                               const struct gl_pixelstore_attrib *packing,
                               GLvoid *pixels);

   GLboolean GLAPIENTRY _mesa_IsEnabled(GLenum cap);
   void finalize_softpipe_draw_vbo(struct softpipe_context *sp, const void* mapped_indices);


}

struct ch4_t {
  ch4_t(){
    channels[0] = channels[1] = channels[2] = channels [3] = 0.0;
  }

  /*union {
    float f;
    uint32_t u;
  };*/

  GLfloat channels[TGSI_NUM_CHANNELS];
  //union tgsi_exec_channel channels[TGSI_NUM_CHANNELS];

  GLfloat &operator[](int index){
    assert(index >=0 and index < TGSI_NUM_CHANNELS);
    return channels[index];
  }
};

struct fragmentData_t {
   fragmentData_t(): passedDepth(false) {}
   GLfloat attribs[PIPE_MAX_SHADER_INPUTS][4];
   std::vector<ch4_t> inputs;
   unsigned uintPos[3];
   unsigned quadIdx;
   //GLfloat floatPos[3];
   bool passedDepth;
   bool isLive;
};

class primitiveFragmentsData_t;

class RasterTile {
   public:
   struct rasterFragment_t {
      rasterFragment_t(): alive(false), frag(NULL),
      tile(NULL){}
      bool alive;
      fragmentData_t* frag;
      RasterTile* tile;
   };
   public:
      RasterTile(primitiveFragmentsData_t* const _prim,
            int _primId, int _tilePos, 
            unsigned _tileH, unsigned _tileW,
            unsigned _xCoord, unsigned _yCoord):
         primId(_primId), 
         tileH(_tileH), tileW(_tileW),
         xCoord(_xCoord), yCoord(_yCoord),
         m_tilePos(_tilePos), 
         lastPrimTile(false),
         m_fragmentsQuads(_tileH*_tileW/QUAD_SIZE, 
         std::vector<rasterFragment_t>(QUAD_SIZE)),
         m_prim(_prim), m_addedFragsCount(0),
         m_skipFineDepth(false)
      {}

      void addFragment(fragmentData_t* frag);

      unsigned size() const { return m_fragmentsQuads.size()*QUAD_SIZE;} 

      void setSkipFineDepth(){
         m_skipFineDepth = true;
      }

      bool skipFineDepth(){
         return m_skipFineDepth;
      }

      /*fragmentData_t& operator[] (const int index)
      {
         return *(m_fragments[index].frag);
      }*/


      fragmentData_t& getFragment (const int index)
      {
         //assert(m_fragments[index].alive);
         return *(m_fragmentsQuads[index/QUAD_SIZE][index%QUAD_SIZE].frag);
      }


      rasterFragment_t& getRasterFragment (const int quadId, unsigned fragId){
         return m_fragmentsQuads[quadId][fragId];
      }

      unsigned setActiveFragmentsIndices() {
         m_fragmentIndices.clear();
         unsigned activeCount = 0;
         for(unsigned i=0; i<m_fragmentsQuads.size(); i++){
            for(unsigned f=0; f<QUAD_SIZE; f++){
               if(m_fragmentsQuads[i][f].frag->isLive 
                     and m_fragmentsQuads[i][f].frag->passedDepth){
                  m_fragmentIndices.push_back(i*QUAD_SIZE+f);
                  activeCount++;
               }
            }
         }
         assert(m_fragmentIndices.size() == activeCount);
         return activeCount;
      }

      bool fullyCovered(){
         return (resetActiveCount()==size());
      }

      unsigned getFragmentIndex(unsigned id){
         return m_fragmentIndices[id];
      }

      unsigned resetActiveCount(){
         m_activeCount = 0;
         for(unsigned quadId=0; quadId < m_fragmentsQuads.size(); quadId++)
            for(unsigned fragId=0; fragId < m_fragmentsQuads[quadId].size(); fragId++){
               if(m_fragmentsQuads[quadId][fragId].alive)
                  m_activeCount++;
            }
         return m_activeCount;
      }

      unsigned getActiveCount(){
         return m_activeCount;
      }

      unsigned decActiveCount(){
         m_activeCount--;
         return m_activeCount;
      }

      uint64_t backDepth(){
         return m_backDepth;
      }

      uint64_t frontDepth(){
         return m_frontDepth;
      }

      const int primId;
      const unsigned tileH;
      const unsigned tileW;
      const unsigned xCoord;
      const unsigned yCoord;
      const unsigned m_tilePos;
      bool lastPrimTile;
   private:
      std::vector <std::vector <rasterFragment_t> > m_fragmentsQuads;
      std::vector<unsigned> m_fragmentIndices;
      std::vector<bool> m_validFragments;
      primitiveFragmentsData_t* const m_prim;
      unsigned m_activeCount;
      uint64_t m_frontDepth;
      uint64_t m_backDepth;
      unsigned m_addedFragsCount;
      bool m_skipFineDepth;
};


typedef std::vector<RasterTile* > RasterTiles;

enum class RasterDirection {
    HorizontalRaster,
    //VerticalRaster,
    //HilbertOrder,
    BlockedHorizontal
};

enum class DepthSize : uint32_t { Z16 = 2, Z32 = 4 };
typedef std::vector<RasterTile::rasterFragment_t*> tcTile_t;
typedef std::vector<RasterTile::rasterFragment_t*>* tcTilePtr_t;
struct mapTileStream_t{
   mapTileStream_t(): tcTilePtr(NULL){}
   tcTilePtr_t tcTilePtr;
   unsigned tileId;
   unsigned primId;
};


struct stage_shading_info_t {
    enum class GraphicsPass { NONE , Vertex, Fragment};
    GraphicsPass currentPass;
    unsigned sent_simt_prims;
    unsigned launched_threads;
    unsigned completed_threads;
    unsigned pending_kernels;
    bool doneEarlyZ;
    unsigned doneZTiles;
    RasterTiles * earlyZTiles;
    bool render_init;
    bool initStageKernelPtr;
    unsigned * primMap;
    unsigned * primCountMap;
    bool finishStageShaders;
    float * deviceVertsAttribs;
    std::vector<cudaStream_t> cudaStreams;
    void* allocAddr;
    void* vertCodeAddr;
    void* fragCodeAddr;
    //temporarly used with earlyZ util multiple streams are re-enabled
    uint32_t currentEarlyZTile;
    std::map<uint64_t, mapTileStream_t> cudaStreamTiles;
    std::map<uint64_t, primitiveFragmentsData_t* > cudaStreamPrims;

    stage_shading_info_t() {
        primMap = NULL;
        primCountMap = NULL;
        earlyZTiles = NULL;
        doneZTiles = 0;
        clear();
    }

    void clear() {
        currentPass = GraphicsPass::NONE;
        sent_simt_prims = 0;
        launched_threads = 0;
        completed_threads = 0;
        pending_kernels = 0;
        doneEarlyZ = true;
        doneZTiles = 0;
        render_init = true;
        allocAddr = NULL;
        vertCodeAddr = NULL;
        fragCodeAddr = NULL;
        deviceVertsAttribs = NULL;
        if(primMap!=NULL){ delete [] primMap;  primMap=NULL;}
        if(primCountMap!=NULL) { delete [] primCountMap; primCountMap= NULL;}
        if(earlyZTiles!=NULL) { assert(0); } //should be cleared when earlyZ is done
        //
        currentEarlyZTile = 0;
    
        for(auto it=cudaStreamTiles.begin(); it!=cudaStreamTiles.end(); ++it){
           if(it->second.tcTilePtr!=NULL){
              delete it->second.tcTilePtr;
           }
        }
        cudaStreamTiles.clear();
    }
};

class primitiveFragmentsData_t {
public:
    primitiveFragmentsData_t(int _primId): primId(_primId){
       maxDepth = (uint64_t) -1;
       minDepth = 0;
       m_validTiles = false;
    }
    ~primitiveFragmentsData_t(){
       for(unsigned t=0; t<m_rasterTiles.size(); t++){
          delete m_rasterTiles[t];
       }
    }
    shaderAttrib_t getFragmentData(unsigned utid, unsigned tid, unsigned attribID, unsigned attribIndex, 
          unsigned fileIdx, unsigned idx2D, void * stream, stage_shading_info_t* shadingData, bool z_unit_disabled);

    void addFragment(fragmentData_t fd);
    inline unsigned size() {
      return m_fragments.size();
    }
    void sortFragmentsInRasterOrder (unsigned frameHeight, unsigned frameWidth,
        const unsigned tileH, const unsigned tileW,
        const unsigned blockH, const unsigned blockW, const RasterDirection rasterDir);
    void sortFragmentsInTiles(unsigned frameHeight, unsigned frameWidth,
        const unsigned tileH, const unsigned tileW,
        const unsigned hTiles, const unsigned wTiles,
        const unsigned tilesCount,
        const unsigned blockH, const unsigned blockW, 
        const RasterDirection rasterDir,
        unsigned simtCount);

    //primitive max and min depth values, used for z-culling
    const int primId; //unique prim id for draw call
    uint64_t maxDepth;
    uint64_t minDepth;

    fragmentData_t& operator[] (const int index)
    {
       return m_fragments[index];
    }
    RasterTiles& getRasterTiles(){
       assert(m_validTiles);
       return m_rasterTiles;
    }

    RasterTiles& getSimtTiles(unsigned clusterId){
       assert(m_validTiles);
       return m_simtRasterTiles[clusterId];
    }

private:
    //the fragment shading data of this primitive
    std::vector<fragmentData_t> m_fragments; 
    RasterTiles m_rasterTiles;
    std::vector<RasterTiles> m_simtRasterTiles;
    bool m_validTiles;
};


class renderData_t {
public:
    renderData_t();
    ~renderData_t();
  
    //mesa calls
    bool GPGPUSimActiveFrame();
    bool GPGPUSimSimulationActive();
    bool GPGPUSimSkipCpFrames();
    void endOfFrame();
    void initializeCurrentDraw (struct tgsi_exec_machine* tmachine, void* sp, void* mapped_indices);
    void finalizeCurrentDraw();
    bool m_flagEndVertexShader;
    void endVertexShading(CudaGPU * cudaGPU);
    unsigned int doFragmentShading();
    unsigned int noDepthFragmentShading();
    bool m_flagEndFragmentShader;
    void endFragmentShading();
    void addFragmentsQuad(std::vector<fragmentData_t>& quad);

    //gpgpusim calls
    bool isDepthTestEnabled();
    bool isBlendingEnabled();
    void getBlendingMode(GLenum * src, GLenum * dst, GLenum* srcAlpha, GLenum * dstAlpha, GLenum* eqnRGB, GLenum* eqnAlpha, GLfloat * blendColor);
    void initParams(bool standalone_mode, unsigned int startFrame, unsigned int endFrame, int startDrawcall, unsigned int endDrawcall, unsigned int tile_H, unsigned int tile_W,
          unsigned int block_H, unsigned int block_W, unsigned blendingMode, unsigned depthMode, unsigned cptStartFrame, unsigned cptEndFrame, unsigned cptPeroid, bool skipCpFrames, char* outdir);
    GLuint getScreenWidth(){return m_bufferWidth;}
    GLuint getRBSize(){return m_bufferWidth*m_bufferHeight;}
    shaderAttrib_t getFragmentData(unsigned utid, unsigned tid, unsigned attribID, 
          unsigned attribIndex, unsigned fileIdx, unsigned idx2D, void * stream);
    uint32_t getVertexData(unsigned threadID, unsigned attribID, unsigned attribIndex, void * stream);
    void writeVertexResult(unsigned threadID, unsigned resAttribID, unsigned attribIndex, float data);
    void checkGraphicsThreadExit(void * kernelPtr, unsigned tid, void* stream);
    void setTcInfo(int pid, int tid){m_tcPid = pid; m_tcTid=tid;}

    //gem5 calls
    void checkEndOfShader(CudaGPU * cudaGPU);
    void doneEarlyZ(); 
    void launchFragmentTile(RasterTile * rasterTile, unsigned tileId);
    void launchTCTile(tcTilePtr_t tcTile, unsigned donePrims);
    void addPrimitive();
    void setVertShaderUsedRegs(int regs){
      m_usedVertShaderRegs = regs;
    }
    void setFragShaderUsedRegs(int regs){
    m_usedFragShaderRegs = regs;
    }

    void getFrameDrawcallNum(int* frameNum, int* drawcallNum){
      *frameNum = m_currentFrame;
      *drawcallNum = m_drawcall_num;
    }
    void setMesaCtx(struct gl_context * ctx){m_mesaCtx=ctx;}
    std::vector<uint64_t> fetchTexels(int modifier, int unit, int dim,
                                      float* coords, int num_coords,
                                      float* dst, int num_dst, unsigned tid,
                                      bool isTxf, bool isTxb);
    unsigned  getTexelSize(int samplingUnit);
    void addTexelFetch(int x, int y, int level);

    unsigned getFramebufferFormat();
    uint64_t getFramebufferFragmentAddr(uint64_t x, uint64_t y, uint64_t size);

    bool isBusy(){
      //if framebuffer is allocated then some rendering is going on
      return (m_deviceData != NULL);
    }

    void updateMachine(struct tgsi_exec_machine* tmachine){
      m_tmachine = tmachine;
    }
    const char* getShaderOutputDir(){
      return m_intFolder.c_str();
    }
    bool depthTest(uint64_t oldDepth, uint64_t newDepth);
    bool testHiz(RasterTile* tile);

private:
    bool useInShaderBlending() const;
    void sortFragmentsInRasterOrder(unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir);
    void runEarlyZ(CudaGPU * cudaGPU, unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir, unsigned clusterCount);
    void generateVertexCode();
    void generateFragmentCode(DepthSize);
    void addFragment(fragmentData_t fragmentData);
    void endDrawCall();
    void setAllTextures(void** fatCubinHandle);

    void putDataOnColorBuffer();
    void putDataOnDepthBuffer();
    void writeDrawBuffer(std::string time, byte * frame, int size, unsigned w, unsigned h, std::string extOrder, int depth);
    byte* setRenderBuffer();
    byte* setDepthBuffer(DepthSize activeDbSize, DepthSize actualDbSize);
    void writeTexture(byte* data, unsigned size, unsigned texNum, unsigned h, unsigned w, std::string typeEx);
    void copyStateData(void** fatCubinHandle);

    void incCurrentFrame();
    void incDrawcallNum(){
       m_drawcall_num++;
       checkExitCond();
    }

    void checkExitCond();
    void checkpoint();
    unsigned int getCurrentFrame(){return m_currentFrame;}
    long long unsigned getDrawcallNum(){return m_drawcall_num;}
    const char* getCurrentShaderId(int shaderType);
    std::string getCurrentShaderName(int shaderType){
        if(shaderType==VERTEX_PROGRAM)
            return ("vp" + std::to_string(m_currentFrame) + "_" + std::to_string(m_drawcall_num));
        if(shaderType==FRAGMENT_PROGRAM)
            return ("fp" + std::to_string(m_currentFrame) + "_" + std::to_string(m_drawcall_num));
        //only two types
        assert(0);
    }
    unsigned getTileH(){return m_tile_H;}
    unsigned getTileW(){return m_tile_W;}
    unsigned getBlockH(){return m_block_H;}
    unsigned getBlockW(){return m_block_W;}
    
    inline unsigned getColorBufferByteSize(){return m_colorBufferByteSize;}
    inline GLuint getBufferWidth(){return m_bufferWidth;}
    inline GLuint getBufferHeight(){return m_bufferHeight;}
    inline GLuint getPixelBufferSize(){ return m_bufferWidth*m_bufferHeight;}
    inline unsigned getFBPixelSize(){ return m_fbPixelSize; }
    struct gl_context * getMesaCtx(){return m_mesaCtx;}
    struct gl_renderbuffer * getMesaBuffer(){return m_mesaColorBuffer;}
    std::string getIntFolder(){return m_intFolder;}
    std::string getFbFolder(){return m_fbFolder;}
    //bool useDefaultShaders(){return m_useDefaultShaders;}
    byte* getDeviceData(){return m_deviceData;}
    byte** getpDeviceData(){return &m_deviceData;}
    //std::string getShaderPTXInfo(std::string arbFileName, std::string functionName);
    std::string getShaderPTXInfo(int usedRegs, std::string functionName);
    void* getShaderFatBin(std::string vertexShader, std::string fragmentShader);
    gl_state_index getParamStateIndexes(gl_state_index index);
    void setHizTiles(RasterDirection rasterDir);

private:
    std::string vPTXPrfx;
    std::string fPTXPrfx;
    std::string fPtxInfoPrfx;
    byte* m_deviceData;
    std::string m_intFolder;
    std::string m_fbFolder;
    bool m_standaloneMode;
    unsigned int m_startFrame;
    unsigned int m_endFrame;
    unsigned int m_cptStartFrame;
    unsigned int m_cptEndFrame;
    unsigned int m_cptPeroid;
    bool m_skipCpFrames;
    unsigned int m_cptNextFrame;
    unsigned int m_currentFrame;
    int m_startDrawcall;
    unsigned int m_endDrawcall;
    uint64_t m_colorBufferByteSize;
    uint64_t m_depthBufferSize;
    byte* m_depthBuffer;
    DepthSize m_depthSize;
    DepthSize m_mesaDepthSize;
    GLuint m_bufferWidth;
    GLuint m_bufferHeight;
    GLuint m_depthBufferWidth;
    GLuint m_depthBufferHeight;
    struct gl_context * m_mesaCtx;
    struct gl_renderbuffer * m_mesaColorBuffer;
    struct gl_renderbuffer * m_mesaDepthBuffer;
    unsigned int m_tile_H;
    unsigned int m_tile_W;
    unsigned int m_tilesCount;
    unsigned int m_wTiles;
    unsigned int m_hTiles;
    unsigned int m_block_H;
    unsigned int m_block_W;
    long long unsigned m_drawcall_num;
    bool currentFrameHasShader;
    bool m_inShaderBlending; //1 in shader, 0 in z-unit
    bool m_inShaderDepth; //1 in shader, 0 in z-unit
    std::vector<primitiveFragmentsData_t> drawPrimitives;
    stage_shading_info_t m_sShading_info;
    std::vector<textureReference*> textureRefs;
    void** lastFatCubinHandle;
    __cudaFatCudaBinary* lastFatCubin;
    int m_tcPid;
    int m_tcTid;
    std::string m_outdir;
    std::mutex vertexFragmentLock;
    int m_usedVertShaderRegs;
    int m_usedFragShaderRegs;
    struct tgsi_exec_machine* m_tmachine;
    void* m_sp;
    void* m_mapped_indices;
    struct texelInfo_t {
      texelInfo_t(uint64_t ba, const struct pipe_resource* t){
        baseAddr = ba;
        tex = t;
      }
      uint64_t baseAddr;
      const struct pipe_resource* tex;
    };
    std::vector<texelInfo_t> m_textureInfo;
    int m_currSamplingUnit;
    std::vector<uint64_t> m_texelFetches;
    std::vector< std::vector<ch4_t> > consts;
    unsigned m_fbPixelSize;

    struct hizBuffer_t {
       hizBuffer_t(renderData_t *rd){
          m_size = 0;
          m_renderData = rd;
       }
       void setSize(unsigned psize){
          m_size = psize;
          frontDepth.resize(psize);
          backDepth.resize(psize);
          depthValid.resize(psize, false);
       }

       void setDepth(unsigned tileIdx,
             unsigned xCoord, unsigned yCoord,
             uint64_t depth){
          assert(tileIdx < frontDepth.size() and tileIdx < backDepth.size());
          if(!depthValid[tileIdx]){
             depthValid[tileIdx] = true;
             frontDepth[tileIdx] = depth;
             backDepth [tileIdx] = depth;
             xCoords[tileIdx] = xCoord;
             yCoords[tileIdx] = yCoord;
          } else {
             assert(xCoords[tileIdx] == xCoord);
             assert(yCoords[tileIdx] == yCoord);
             if(m_renderData->depthTest(frontDepth[tileIdx], depth)){
                frontDepth[tileIdx] = depth;
             } else if(m_renderData->depthTest(depth, backDepth[tileIdx])){
                backDepth[tileIdx] = depth;
             }
          }
       }

       unsigned size() { return m_size;}
       std::vector<uint64_t> frontDepth;
       std::vector<uint64_t> backDepth;
       std::vector<unsigned> xCoords;
       std::vector<unsigned> yCoords;
       std::vector<bool> depthValid;
       private:
       unsigned m_size;
       renderData_t* m_renderData;
    };

    hizBuffer_t m_hizBuff;
};

extern renderData_t g_renderData;

class Utils {
   public:
      static byte* RGB888_to_RGBA888(byte* rgb, int size, byte alpha=255 /*fully opaque*/);
};

#endif //MESA_GPGPUSIM_H
