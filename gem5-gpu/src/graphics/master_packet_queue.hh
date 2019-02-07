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

#ifndef __MASTER_PACKET_QUEUE_HH__
#define __MASTER_PACKET_QUEUE_HH__

#include <queue>
#include "mem/mem_object.hh"
#include "params/MasterPacketQueue.hh"
      

class MasterPacketQueue : public MemObject
{
   protected:
      typedef MasterPacketQueueParams Params;

   public:
      MasterPacketQueue(const Params *p);
      ~MasterPacketQueue();
      BaseMasterPort& getMasterPort(const std::string& if_name,
            PortID idx = InvalidPortID) override;
      BaseSlavePort& getSlavePort(const std::string& if_name,
            PortID idx = InvalidPortID) override;
      void regStats();

   private:
      class QueueSlavePort : public SlavePort
   {
      public:
         QueueSlavePort(const std::string &_name, MasterPacketQueue *_mpq):
            SlavePort(_name, _mpq), mpq(_mpq) {}

      protected:
         bool recvTimingReq(PacketPtr pkt){
            return mpq->recvTimingReq(pkt);
         }

         bool recvTimingSnoopResp(PacketPtr pkt){
            return mpq->recvTimingSnoopResp(pkt);
         }

         Tick recvAtomic(PacketPtr pkt){
            return mpq->recvAtomic(pkt);
         }

         void recvFunctional(PacketPtr pkt) {
            mpq->recvFunctional(pkt);
         }

         void recvRespRetry(){
            mpq->recvRespRetry();
         }

         AddrRangeList getAddrRanges() const {
            return mpq->getAddrRanges();
         }

      private:
         MasterPacketQueue* mpq;
   };

      class QueueMasterPort : public MasterPort
   {
      public:
         QueueMasterPort(const std::string &_name, MasterPacketQueue * _mpq)
            : MasterPort(_name, _mpq), mpq(_mpq) {}

      protected:
        void recvFunctionalSnoop(PacketPtr pkt)
        {
            mpq->recvFunctionalSnoop(pkt);
        }

        Tick recvAtomicSnoop(PacketPtr pkt)
        {
            return mpq->recvAtomicSnoop(pkt);
        }

        bool recvTimingResp(PacketPtr pkt)
        {
            return mpq->recvTimingResp(pkt);
        }

        void recvTimingSnoopReq(PacketPtr pkt)
        {
            mpq->recvTimingSnoopReq(pkt);
        }

        void recvRangeChange()
        {
            mpq->recvRangeChange();
        }

        bool isSnooping() const
        {
            return mpq->isSnooping();
        }

        void recvReqRetry()
        {
            mpq->recvReqRetry();
        }

        void recvRetrySnoopResp()
        {
            mpq->recvRetrySnoopResp();
        }

      private:
        MasterPacketQueue * mpq;
   };

      void drainPacketQueue(){
         if(packetQueue.size() > 0){
            if(masterPort.sendTimingReq(packetQueue.front())){
               packetQueue.pop();
               if(pendingMasterReq){
                  pendingMasterReq = false;
                  //slavePort.sendReqRetry();
               }
               if(!drainPacketQueueEvent.scheduled())
                  schedule(drainPacketQueueEvent, nextCycle());
            }
         }
      }

      const unsigned queueSize;
      QueueMasterPort masterPort;
      QueueSlavePort slavePort;
      std::queue<PacketPtr> packetQueue;
      EventWrapper<MasterPacketQueue, &MasterPacketQueue::drainPacketQueue> 
         drainPacketQueueEvent;
      bool pendingMasterReq;

      //slave interface
      bool recvTimingReq(PacketPtr pkt);
      bool recvTimingSnoopResp(PacketPtr pkt);
      Tick recvAtomic(PacketPtr pkt);
      void recvFunctional(PacketPtr pkt);
      void recvRespRetry();
      AddrRangeList getAddrRanges() const;

      //master interface
      void recvFunctionalSnoop(PacketPtr pkt);
      Tick recvAtomicSnoop(PacketPtr pkt);
      bool recvTimingResp(PacketPtr pkt);
      void recvTimingSnoopReq(PacketPtr pkt);
      void recvRangeChange();
      bool isSnooping() const;
      void recvReqRetry();
      void recvRetrySnoopResp();

      void printStats();
};

#endif
