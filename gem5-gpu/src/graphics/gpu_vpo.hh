/*
 * Copyright (c) 2019 Ayub A. Gubran and Tor M. Aamodt
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

#ifndef __GPU_VPO_HH__
#define __GPU_VPO_HH__

#include <queue>
#include <unordered_map>
#include "graphics/mesa_gpgpusim.h"
#include "mem/mem_object.hh"
#include "params/GPU_VPO.hh"
#include "stream_manager.h"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
      
extern renderData_t g_renderData;

class GPU_VPO : public MemObject
{
   protected:
      typedef GPU_VPOParams Params;

   private:
      MasterID masterId;
      CudaGPU *cudaGPU;

      class VPOMasterPort : public MasterPort
   {
      public:
      VPOMasterPort(const std::string &_name, GPU_VPO * _vpo)
         : MasterPort(_name, _vpo), vpo(_vpo) {}


      protected:
      virtual bool recvTimingResp(PacketPtr pkt);
      virtual void recvReqRetry();
      virtual Tick recvAtomic(PacketPtr pkt);
      virtual void recvFunctional(PacketPtr pkt);

      private:
      GPU_VPO * vpo;
   };

    class VPOSlavePort : public SlavePort
    {
      private:

        // a pointer to our specific cache implementation
        // Pointer back to vpo for callbacks
        GPU_VPO * vpo;

      protected:


        virtual bool recvTimingReq(PacketPtr pkt);
        virtual bool recvTimingSnoopResp(PacketPtr pkt);
        virtual Tick recvAtomic(PacketPtr pkt);
        virtual void recvFunctional(PacketPtr pkt);
        virtual void recvRespRetry();
        virtual AddrRangeList getAddrRanges() const;

      public:

        VPOSlavePort(const std::string &_name, GPU_VPO *_vpo):
           SlavePort(_name, _vpo), vpo(_vpo) {}

    };

      const unsigned vpoCount;
      const unsigned pvbSize;

      //ports
      std::vector<VPOMasterPort*> vpoMasterPorts;
      std::vector<VPOSlavePort*> vpoSlavePorts;

      //EventWrapper<GPU_VPO, &GPU_VPO::vpoTick> tickEvent;

      //EventWrapper<GPU_VPO, &GPU_VPO::processDepthResponse> depthResponseEvent;

      /*class TickEvent : public Event
        {
        friend class GPU_VPO;

        private:
        GPU_VPO * zunit;

        public:
        TickEvent(GPU_VPO *_zunit) : Event(CPU_Tick_Pri), zunit(_zunit) {}
        void process() { ->tick(); }
        virtual const char *description() const { return "GPU_VPO tick"; }
        };
        TickEvent tickEvent;*/

      void vpoTick(){
      }
      void printStats();

   public:
      GPU_VPO(const Params *p);
      ~GPU_VPO();
      BaseMasterPort& getMasterPort(const std::string& if_name,
            PortID idx = InvalidPortID) override;
      BaseSlavePort& getSlavePort(const std::string& if_name,
            PortID idx = InvalidPortID) override;

      void regStats();

      /*//stats
        Stats::Scalar numZCacheRequests;
        Stats::Scalar numZCacheRetry;*/
};

#endif
