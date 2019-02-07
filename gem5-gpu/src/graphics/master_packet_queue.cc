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

#include <iostream>

#include "debug/MasterPacketQueue.hh"
#include "params/MasterPacketQueue.hh"
#include "graphics/master_packet_queue.hh"
#include "sim/system.hh"


MasterPacketQueue::MasterPacketQueue(const Params *p):
   MemObject(p),  
   queueSize(p->size),
   masterPort(name() + ".master", this),
   slavePort(name() + ".master", this),
   drainPacketQueueEvent(this),
   pendingMasterReq(false)
{
   DPRINTF(MasterPacketQueue, "Created a MasterPacketQueue\n");
}

MasterPacketQueue::~MasterPacketQueue(){}

BaseMasterPort&
MasterPacketQueue::getMasterPort(const std::string &if_name, PortID idx)
{
   if(if_name == "master"){
      return masterPort;
   } else {
      return MemObject::getMasterPort(if_name, idx);
   }
}

BaseSlavePort&
MasterPacketQueue::getSlavePort(const std::string& if_name, PortID idx)
{
    if (if_name == "slave"){
        return slavePort;
    } else {
        return MemObject::getSlavePort(if_name, idx);
    }
}

//slave ports
bool MasterPacketQueue::recvTimingReq(PacketPtr pkt){
   if(packetQueue.size() < queueSize){
      packetQueue.push(pkt);
      schedule(drainPacketQueueEvent, nextCycle());
      return true;
   }
   pendingMasterReq = true;
   return false;
}

bool MasterPacketQueue::recvTimingSnoopResp(PacketPtr pkt){
   panic("not implemented");
   //return masterPort.sendTimingSnoopResp(pkt);
   return false;
}


Tick MasterPacketQueue::recvAtomic(PacketPtr pkt){
   panic("not implemented");
   //return masterPort.sendAtomic(pkt);
   return 0;
}

void MasterPacketQueue::recvFunctional(PacketPtr pkt){
   panic("not implemented");
   //masterPort.sendFunctional(pkt);
}

void MasterPacketQueue::recvRespRetry(){
    masterPort.sendRetryResp();
}

AddrRangeList MasterPacketQueue::getAddrRanges() const
{
   return masterPort.getAddrRanges();
}

//master functions
void MasterPacketQueue::recvFunctionalSnoop(PacketPtr pkt){
   panic("not implemented");
   //slavePort.sendFunctionalSnoop(pkt);
}

Tick MasterPacketQueue::recvAtomicSnoop(PacketPtr pkt){
   panic("not implemented");
   //return slavePort.sendAtomicSnoop(pkt);
   return 0;
}

bool MasterPacketQueue::recvTimingResp(PacketPtr pkt){
   panic("not implemented");
   //bool successful = slavePort.sendTimingResp(pkt);
   return false;
}

void MasterPacketQueue::recvTimingSnoopReq(PacketPtr pkt){
   panic("not implemented");
   //slavePort.sendTimingSnoopReq(pkt);
}

void MasterPacketQueue::recvRangeChange(){
   panic("not implemented");
   //slavePort.sendRangeChange();
}

bool MasterPacketQueue::isSnooping() const{
   panic("not implemented");
   //return slavePort.isSnooping();
   return false;
}

void MasterPacketQueue::recvReqRetry(){
   panic("not implemented");
   //slavePort.sendRetryReq();
}

void MasterPacketQueue::recvRetrySnoopResp(){
   panic("not implemented");
   //slavePort.sendRetrySnoopResp();
}

void MasterPacketQueue::regStats(){
   MemObject::regStats();

   /*numZCacheRequests
      .name(name() + ".z_cache_requests")
      .desc("Number of z-cache requests sent")
      ;
   numZCacheRetry
      .name(name() + ".z_cache_retries")
      .desc("Number of z-cache retries")
      ;*/
}

void MasterPacketQueue::printStats(){
   /*DPRINTF(MasterPacketQueue, "-------------------Z Stats-----------------------\n");
   DPRINTF(MasterPacketQueue, "retryZPkts.size()=%d\n", retryZPkts.size());
   //DPRINTF(MasterPacketQueue, "blockedCount=%d\n", blockedCount);
   DPRINTF(MasterPacketQueue, "pendingTranslations=%d\n", pendingTranslations);
   DPRINTF(MasterPacketQueue, "depthUpdateQ.size()=%d\n", depthUpdateQ.size());
   DPRINTF(MasterPacketQueue, "doneTiles=%d\n", doneTiles);
   DPRINTF(MasterPacketQueue, "-------------------------------------------------\n");*/
}

MasterPacketQueue *MasterPacketQueueParams::create() {
   return new MasterPacketQueue(this);
}
