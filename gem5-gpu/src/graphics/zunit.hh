/*
 * Copyright (c) 2016 Ayub A. Gubran and Tor M. Aamodt
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __GPGPU_ZUNIT_HH__
#define __GPGPU_ZUNIT_HH__

#include <unordered_map>
#include <GL/gl.h>
#include "base/callback.hh"
#include "mem/mem_object.hh"
#include "params/ZUnit.hh"
#include "stream_manager.h"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "graphics/mesa_gpgpusim.h"
//#include "graphics/graphics_structs.h"
      
extern renderData_t g_renderData;
class RasterTile;
typedef std::vector<RasterTile* > RasterTiles;

class ZUnit : public MemObject
{
   protected:
   typedef ZUnitParams Params;

   private:
      MasterID masterId;

   private:
      CudaGPU *cudaGPU;
      ShaderTLB * ztb;

      class ZCachePort : public MasterPort
   {
      friend class CudaGPU;

      private:
      // Pointer back to shader zunit for callbacks
      ZUnit * zunit;

      public:
      ZCachePort(const std::string &_name, ZUnit * _zunit)
         : MasterPort(_name, _zunit), zunit(_zunit) {}

      protected:
      virtual bool recvTimingResp(PacketPtr pkt);
      virtual void recvReqRetry();
      virtual Tick recvAtomic(PacketPtr pkt);
      virtual void recvFunctional(PacketPtr pkt);
   };

      //a port to the zcahe
      ZCachePort zcachePort;
      MasterID zcacheMasterId;
      const unsigned maxPendingReqs;
      const unsigned depthResponseQueueSize;
      const unsigned depthTestDelay;
      const unsigned zropWidth;
      const unsigned hizWidth;
      bool zPendingOnCache;

      void sendZcacheAccess(PacketPtr pkt);

      // Holds texture packets that need to be retried
      void handleZcacheRetry();
      bool stallOnCacheRetry;
      std::queue<PacketPtr> retryZPkts;
      EventWrapper<ZUnit, &ZUnit::handleZcacheRetry> zcacheRetryEvent;

      Addr depthAddrStart;
      Addr depthAddrEnd;
      DepthSize depthSize;
      RasterTiles* fragTiles;
      GLenum depthFunc;

      class DepthFragmentTile {
         public:
            DepthFragmentTile(): hizPassThresh((uint64_t)-1), doneFragments(0) {}

            class DepthFragment {
               public:
                  DepthFragment(unsigned _id, Addr _dAddr, uint32_t _dVal, DepthFragmentTile * _tile, fragmentData_t * _rasterFrag):
                     id(_id), depthVaddr(_dAddr), depthVal(_dVal), tile(_tile), rasterFrag(_rasterFrag){}
                  inline DepthFragmentTile* getTile() { return tile; }
                  inline uint64_t getDepthVal() { return depthVal;}
                  inline void setDepthVal(uint64_t dv) { depthVal = dv;}

                  inline Addr getDepthVaddr() { return depthVaddr;}
                  inline Addr getDepthPaddr() { return depthPaddr;}
                  inline void setDepthPaddr(Addr paddr) { depthPaddr=paddr;}

                  void setPassed(){ rasterFrag->passedDepth = true;}
                  void unsetPassed(){ rasterFrag->passedDepth = false;}
                  bool passed() { return rasterFrag->passedDepth;}
                  unsigned getId() { return id; }
                  unsigned getX() { return rasterFrag->intPos[0];}
                  unsigned getY() { return rasterFrag->intPos[1];}

               private:
                  unsigned id;
                  Addr depthVaddr; 
                  Addr depthPaddr;
                  uint64_t depthVal;
                  DepthFragmentTile * tile;
                  bool pass;
                  fragmentData_t * rasterFrag;
            }; 

            typedef std::vector<DepthFragment> DepthFragments;
            void incDoneFragments() {
               doneFragments++;
               assert(doneFragments <= depthFragments.size());
            }

            bool isEmpty() { return (0 == depthFragments.size()); }

            bool isDone(){ return (doneFragments == depthFragments.size()); }

            unsigned getId(){ return tileId; }

            void setId(unsigned _tileId){ tileId = _tileId; }

            DepthFragment * getFragment(unsigned idx){ return &depthFragments[idx]; } 

            void addFragment(DepthFragment df){ depthFragments.push_back(df); }

            unsigned size(){ return depthFragments.size(); }

            void setRasterTile(RasterTile * _tile) { rasterTile = _tile; }
            RasterTile* getRasterTile(){ return rasterTile; }

            void setHizDepth(uint64_t depth){ hizDepth = depth; }
            uint64_t hizDepth; 

            void setHizThresh(uint64_t thresh) { hizPassThresh = thresh;}
            uint64_t hizThresh() { return hizPassThresh;}

         private:
            uint64_t hizPassThresh;
            DepthFragments depthFragments;
            unsigned doneFragments;
            unsigned tileId;
            RasterTile * rasterTile;
      };

      typedef std::vector<DepthFragmentTile> DepthTiles;

      DepthTiles depthTiles;
      
      //std::map<Addr, bool> blockedLineAddrs;
      //std::map<Addr, std::queue<PacketPtr> > blockedAccesses;
      //unsigned blockedCount;
      std::unordered_map<Addr, DepthFragmentTile::DepthFragment* > ztable;

      std::vector<uint64_t> hizBuffer;
      void initHizBuffer(uint8_t* depthBuffer, unsigned frameWidth, unsigned frameHeight, 
               DepthSize dSize, const unsigned tileW, const unsigned tileH,
               const unsigned blockH, const unsigned blockW, const RasterDirection rasterDir);

      std::queue<DepthFragmentTile*> hizQ;
      
      std::queue<DepthFragmentTile::DepthFragment*> depthUpdateQ;
      void pushRequest();

      bool recvDepthResponse(PacketPtr pkt); 
      void processDepthResponse();
      std::queue<PacketPtr> depthResponseQ;
      EventWrapper<ZUnit, &ZUnit::processDepthResponse> depthResponseEvent;

      class TickEvent : public Event
   {
      friend class ZUnit;

      private:
      ZUnit * zunit;

      public:
      TickEvent(ZUnit *_zunit) : Event(CPU_Tick_Pri), zunit(_zunit) {}
      void process() { zunit->tick(); }
      virtual const char *description() const { return "ZUnit tick"; }
   };

      TickEvent tickEvent;
      void tick();
      void checkAndReleaseTickEvent();
      unsigned unitDelay;

      unsigned currTile;
      unsigned currFragment;
      unsigned totalFragments;
      unsigned doneFrags;
      unsigned doneTiles;
      bool doneFlag;
      void doneEarlyZ();
      bool doneEarlyZPending;

      //Initializes a z-cache fetch from gem5
      void sendZTransReq(DepthFragmentTile::DepthFragment * df);
      void sendZWrite(DepthFragmentTile::DepthFragment * df);
      void pushRequest(PacketPtr pkt);
      //void unblockZAccesses(Addr addr);
      unsigned pendingTranslations;
      bool depthTest(uint64_t oldDepthVal, uint64_t newDepthVal);

      void printStats();

   public:
      ZUnit(const Params *p);
      virtual BaseMasterPort& getMasterPort(const std::string &if_name, PortID idx = -1);

      //for z requests translation
      void finishTranslation(WholeTranslationState *state);

      //never squashed
      bool isSquashed() const { return false; }

      void regStats();

      //stats
      Stats::Scalar numZCacheRequests;
      Stats::Scalar numZCacheRetry;

      void startEarlyZ(uint64_t depthBuffStart, uint64_t depthBuffEnd, unsigned bufWidth, RasterTiles* tiles, DepthSize dSize, GLenum _depthFunc,
          uint8_t* depthBuf, unsigned frameWidth, unsigned frameHeight, unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection dir);
};

#endif
