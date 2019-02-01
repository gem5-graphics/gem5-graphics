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

#include "debug/GPU_VPO.hh"
#include "params/GPU_VPO.hh"
#include "graphics/gpu_vpo.hh"
#include "sim/system.hh"


GPU_VPO::GPU_VPO(const Params *p) :
   MemObject(p),  
   masterId(p->sys->getMasterId(name())),
   cudaGPU(p->gpu), 
   vpoMasterPort(name() + ".master_port", this),
   vpoSlavePort(name() + ".slave_port", this),
   vpoWidth(p->vpo_width),
   pvbSize(p->pvb_size)
   //tickEvent([this]{tick();}, name())
{
   DPRINTF(GPU_VPO, "Created a GPU_VPO\n");
}

BaseMasterPort&
GPU_VPO::getMasterPort(const std::string &if_name, PortID idx)
{
   if(if_name == "master_port"){
      return vpoMasterPort;
   } else {
      return MemObject::getMasterPort(if_name, idx);
   }
}

BaseSlavePort&
GPU_VPO::getSlavePort(const std::string& if_name, PortID idx)
{
    if (if_name == "slave_port") {
        return vpoSlavePort;
    } else {
        return MemObject::getSlavePort(if_name, idx);
    }
}

bool
GPU_VPO::VPOMasterPort::recvTimingResp(PacketPtr pkt)
{
   panic("Not implemented");
}

void
GPU_VPO::VPOMasterPort::recvReqRetry()
{
   panic("Not implemented");
}

Tick
GPU_VPO::VPOMasterPort::recvAtomic(PacketPtr pkt)
{
   panic("Not sure how to recvAtomic");
   return 0;
}

void
GPU_VPO::VPOMasterPort::recvFunctional(PacketPtr pkt)
{
   panic("Not sure how to recvFunctional");
}

bool GPU_VPO::VPOSlavePort::recvTimingReq(PacketPtr pkt){
   panic("Not implemented");
}

bool GPU_VPO::VPOSlavePort::recvTimingSnoopResp(PacketPtr pkt){
   panic("Not implemented");
}

Tick GPU_VPO::VPOSlavePort::recvAtomic(PacketPtr pkt){
   panic("Not implemented");
}

void GPU_VPO::VPOSlavePort::recvFunctional(PacketPtr pkt){
   panic("Not implemented");
}

void GPU_VPO::VPOSlavePort::recvRespRetry(){
   panic("Not implemented");
}

AddrRangeList GPU_VPO::VPOSlavePort::getAddrRanges() const{
   panic("Not implemented");
   return AddrRangeList();
}

void GPU_VPO::regStats(){
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

void GPU_VPO::printStats(){
   /*DPRINTF(GPU_VPO, "-------------------Z Stats-----------------------\n");
   DPRINTF(GPU_VPO, "retryZPkts.size()=%d\n", retryZPkts.size());
   //DPRINTF(GPU_VPO, "blockedCount=%d\n", blockedCount);
   DPRINTF(GPU_VPO, "pendingTranslations=%d\n", pendingTranslations);
   DPRINTF(GPU_VPO, "depthUpdateQ.size()=%d\n", depthUpdateQ.size());
   DPRINTF(GPU_VPO, "doneTiles=%d\n", doneTiles);
   DPRINTF(GPU_VPO, "-------------------------------------------------\n");*/
}

GPU_VPO *GPU_VPOParams::create() {
   return new GPU_VPO(this);
}
