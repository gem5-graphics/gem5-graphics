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


class RasterTile {
   public:
      RasterTile(int _primId, int _tilePos): primId(_primId), m_tilePos(_tilePos) {}

      void push_back(fragmentData_t frag){ m_fragments.push_back(frag); }

      unsigned size() const { return m_fragments.size();} 

      fragmentData_t& operator[] (const int index)
      {
         return m_fragments[index];
      }

      unsigned setActiveFragmentsIndices() {
         m_fragmentIndices.clear();
         unsigned activeCount = 0;
         for(int i=0; i<m_fragments.size(); i++){
            if(m_fragments[i].passedDepth){
               m_fragmentIndices.push_back(i);
               activeCount++;
            }
         }
         assert(m_fragmentIndices.size() == activeCount);
         return activeCount;
      }

      unsigned getFragmentIndex(unsigned id){
         return m_fragmentIndices[id];
      }
      unsigned getTilePos(){
         return m_tilePos;
      }
      const int primId;
   private:
      std::vector<fragmentData_t> m_fragments;
      std::vector<unsigned> m_fragmentIndices;
      const unsigned m_tilePos;
};


typedef std::vector<RasterTile* > RasterTiles;

enum RasterDirection {
    VerticalRaster,
    HorizontalRaster,
    BlockedHorizontal,
    HilbertOrder
};

enum class DepthSize : uint32_t { Z16 = 2, Z32 = 4 };

struct mapTileStream_t{
   unsigned tileId;
   unsigned primId;
};

class primitiveFragmentsData_t;

struct stage_shading_info_t {
    enum class GraphicsPass { NONE , Vertex, Fragment};
    GraphicsPass currentPass;
    unsigned launched_threads;
    unsigned completed_threads;
    bool doneEarlyZ;
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
    std::vector<uint32_t> earlyZTilesCounts;
    std::vector<uint32_t> earlyZTilesIds;
    std::map<uint64_t, mapTileStream_t> cudaStreamTiles;
    std::map<uint64_t, primitiveFragmentsData_t* > cudaStreamPrims;

    stage_shading_info_t() {
        primMap = NULL;
        primCountMap = NULL;
        earlyZTiles = NULL;
        clear();
    }

    void clear() {
        currentPass = GraphicsPass::NONE;
        launched_threads = 0;
        completed_threads = 0;
        doneEarlyZ = true;
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
        earlyZTilesCounts.clear();
        earlyZTilesIds.clear();
        cudaStreamTiles.clear();
    }
};


class primitiveFragmentsData_t {
public:
    primitiveFragmentsData_t(int _primId): primId(_primId){
       maxDepth = (uint64_t) -1;
       minDepth = 0;
    }
    shaderAttrib_t getFragmentData(unsigned threadID, unsigned attribID, unsigned attribIndex, unsigned fileIdx, unsigned idx2D,
                          void * stream, stage_shading_info_t* shadingData, bool z_unit_disabled);

    void addFragment(fragmentData_t fd);
    inline unsigned size() {
      return m_fragments.size();
    }
    void sortFragmentsInRasterOrder (unsigned frameHeight, unsigned frameWidth,
        const unsigned tileH, const unsigned tileW,
        const unsigned blockH, const unsigned blockW, const RasterDirection rasterDir);
    RasterTiles* sortFragmentsInTiles(unsigned frameHeight, unsigned frameWidth,
        const unsigned tileH, const unsigned tileW,
        const unsigned blockH, const unsigned blockW, const RasterDirection rasterDir);
    void clear();

    //primitive max and min depth values, used for z-culling
    const int primId; //unique prim id for draw call
    uint64_t maxDepth;
    uint64_t minDepth;

    fragmentData_t& operator[] (const int index)
    {
       return m_fragments[index];
    }
private:
    std::vector<fragmentData_t> m_fragments; //the fragment shading data of this primitive
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
    bool m_flagEndFragmentShader;
    void endFragmentShading();
    void addFragmentsQuad(std::vector<fragmentData_t>& quad);

    //gpgpusim calls
    bool isDepthTestEnabled();
    bool isBlendingEnabled();
    void getBlendingMode(GLenum * src, GLenum * dst, GLenum* srcAlpha, GLenum * dstAlpha, GLenum* eqnRGB, GLenum* eqnAlpha, GLfloat * blendColor);
    void initParams(unsigned int startFrame, unsigned int endFrame, int startDrawcall, unsigned int endDrawcall, unsigned int tile_H, unsigned int tile_W,
          unsigned int block_H, unsigned int block_W, unsigned blendingMode, unsigned depthMode, unsigned cptStartFrame, unsigned cptEndFrame, unsigned cptPeroid, bool skipCpFrames, char* outdir);
    GLuint getScreenWidth(){return m_bufferWidth;}
    GLuint getRBSize(){return m_bufferWidth*m_bufferHeight;}
    shaderAttrib_t getFragmentData(unsigned threadID, unsigned attribID, unsigned attribIndex, unsigned fileIdx, unsigned idx2D, void * stream);
    uint32_t getVertexData(unsigned threadID, unsigned attribID, unsigned attribIndex, void * stream);
    void writeVertexResult(unsigned threadID, unsigned resAttribID, unsigned attribIndex, float data);
    void checkGraphicsThreadExit(void * kernelPtr, unsigned tid);
    void setTcInfo(int pid, int tid){m_tcPid = pid; m_tcTid=tid;}

    //gem5 calls
    void checkEndOfShader(CudaGPU * cudaGPU);
    void doneEarlyZ(); 
    void launchFragmentTile(RasterTile * rasterTile, unsigned tileId);
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

private:
    bool useInShaderBlending() const;
    void sortFragmentsInRasterOrder(unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir);
    void runEarlyZ(CudaGPU * cudaGPU, unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir);
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

private:
    std::string vPTXPrfx;
    std::string fPTXPrfx;
    std::string fPtxInfoPrfx;
    byte* m_deviceData;
    std::string m_intFolder;
    std::string m_fbFolder;
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
};

extern renderData_t g_renderData;

class Utils {
   public:
      static byte* RGB888_to_RGBA888(byte* rgb, int size, byte alpha=255 /*fully opaque*/);
};

#endif //MESA_GPGPUSIM_H
