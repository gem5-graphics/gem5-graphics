/*
 * Copyright (c) Ayub A. Gubran and Tor M. Aamodt 
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

#include "debug/ZUnit.hh"
#include "params/ZUnit.hh"
#include "gpu/gpgpu-sim/zunit.hh"
#include "mem/page_table.hh"
#include "sim/system.hh"
#include "arch/utility.hh"
#include "base/output.hh"


ZUnit::ZUnit(const Params *p) :
   MemObject(p),  masterId(p->sys->getMasterId(name())),
   cudaGPU(p->gpu), ztb(p->ztb),
   zcachePort(name() + ".z_port", this),
   zcacheMasterId(p->sys->getMasterId(name() + "z_unit")),
   maxPendingReqs(p->max_pending_reqs),
   depthResponseQueueSize(p->depth_response_queue_size),
   depthTestDelay(p->depth_test_delay),
   zcacheRetryEvent(this),
   depthResponseEvent(this), tickEvent(this)
{
   DPRINTF(ZUnit, "Created a ZUnit Interface\n");
   cudaGPU->registerZUnit(this);
   stallOnCacheRetry = false;
   zPendingOnCache = false;
   doneFrags = 0;
   doneTiles = 0;
   blockedCount = 0;
   pendingTranslations = 0;
   totalFragments = 0;
   doneEarlyZPending = false;
}

BaseMasterPort&
ZUnit::getMasterPort(const std::string &if_name, PortID idx)
{
   if(if_name == "z_port"){
      return zcachePort;
   } else {
      return MemObject::getMasterPort(if_name, idx);
   }
}

bool
ZUnit::ZCachePort::recvTimingResp(PacketPtr pkt)
{
   return zunit->recvDepthResponse(pkt);
}


void
ZUnit::ZCachePort::recvRetry()
{
   zunit->handleZcacheRetry();
}

Tick
ZUnit::ZCachePort::recvAtomic(PacketPtr pkt)
{
   panic("Not sure how to recvAtomic");
   return 0;
}

void
ZUnit::ZCachePort::recvFunctional(PacketPtr pkt)
{
   panic("Not sure how to recvFunctional");
}

bool
ZUnit::recvDepthResponse(PacketPtr pkt){
   assert(pkt->req->isZFetch());

   if(depthResponseQ.size() >= depthResponseQueueSize) 
      return false;

   depthResponseQ.push(pkt);
   if(!depthResponseEvent.scheduled())
      schedule(depthResponseEvent, nextCycle());
   return true; 
}

void
ZUnit::processDepthResponse(){
   assert(depthResponseQ.size() > 0);

   PacketPtr pkt = depthResponseQ.front();
   depthResponseQ.pop();

   if(!depthResponseQ.empty()){
      schedule(depthResponseEvent, nextCycle());
   }

   if(pkt->isWrite()){
      DPRINTF(ZUnit, "Finished updating z access on paddr 0x%x\n",
           pkt->req->getPaddr());
      return;
   }

   DPRINTF(ZUnit, "Fetched z access on vaddr 0x%x\n",
        pkt->req->getVaddr());

   DepthFragmentTile::DepthFragment * df = (DepthFragmentTile::DepthFragment*) pkt->req->getExtraData();

   doneFrags++;
   df->getTile()->incDoneFragments();
   DPRINTF(ZUnit, "doneFrags = %d\n", doneFrags);
   printStats();

   uint8_t * depthValue = new uint8_t[(int)depthSize];
   pkt->writeData(depthValue);

   uint64_t oldDepthVal;
   if(depthSize == DepthSize::Z16){
      oldDepthVal = *(uint16_t*) depthValue;
   } else if(depthSize == DepthSize::Z32) {
      oldDepthVal = *(uint32_t*) depthValue;
   } else {
      panic("Unsupported depth size\n");
   }

   uint64_t newDepthVal = df->getDepthVal();

   DPRINTF(ZUnit, "Tile: %d, Fragment: %d, oldDepth=%x, newDepth=%x\n", df->getTile()->getId(), df->getId(), oldDepthVal, newDepthVal);

   df->unsetPassed();
   switch(depthFunc){
      case GL_LESS: 
         if(newDepthVal < oldDepthVal)
            df->setPassed();
         break;
      case GL_LEQUAL:
         if(newDepthVal <= oldDepthVal)
            df->setPassed();
         break;
      case GL_GEQUAL: 
         if(newDepthVal >= oldDepthVal)
            df->setPassed();
         break;
      case GL_GREATER: 
         if(newDepthVal > oldDepthVal)
            df->setPassed();
         break;
      case GL_NOTEQUAL:
         if(newDepthVal != oldDepthVal)
            df->setPassed();
         break;
      case GL_EQUAL:
         if(newDepthVal == oldDepthVal)
            df->setPassed();
         break;
      default: 
         panic("Unsupported depth function %x\n", depthFunc);
   }

   if(df->passed()){
      df->setDepthPaddr(pkt->req->getPaddr());
      df->setDepthVal(newDepthVal);
      depthUpdateQ.push(df);
      if(!tickEvent.scheduled())
         schedule(tickEvent, nextCycle());
   } else {
      unblockZAccesses(pkt->req->getPaddr());
   }

   if(df->getTile()->isDone()){
      doneTiles++;
      assert(doneTiles <= depthTiles.size());
      DPRINTF(ZUnit, "Done tile %d\n", df->getTile()->getId());
      g_renderData.launchFragmentTile(df->getTile()->getRasterTile());
   }

   if(doneTiles == depthTiles.size()){
      doneEarlyZ();
   }
}

void ZUnit::unblockZAccesses(Addr addr){
   blockedLineAddrs.erase(addr);

   if(blockedAccesses.count(addr) > 0){ 
      PacketPtr nextReq = blockedAccesses[addr].front();
      blockedAccesses[addr].pop();
      blockedCount--;
      checkAndReleaseTickEvent();

      if (blockedAccesses[addr].empty()) {
         blockedAccesses.erase(addr);
      }

      if (!stallOnCacheRetry) {
         sendZcacheAccess(nextReq);
      } else {
         retryZPkts.push(nextReq);
         DPRINTF(ZUnit, "Z-port blocked, added vaddr: 0x%x to retry list: size @%d\n", nextReq->req->getVaddr(), retryZPkts.size());
         printStats();
      }
   }
}

void
ZUnit::handleZcacheRetry(){
   assert(stallOnCacheRetry);
   assert(retryZPkts.size());

   numZCacheRetry++;

   PacketPtr retry_pkt = retryZPkts.front();
   DPRINTF(ZUnit, "Received z-cache retry, pkt is write=%d, paddr: 0x%x\n",
         retry_pkt->cmd==MemCmd::WriteReq, retry_pkt->req->getPaddr());


   DPRINTF(ZUnit, "sendTimingReq to the z-cache, type write=%d @ cycle %d\n", retry_pkt->cmd == MemCmd::WriteReq, curCycle());
   if (zcachePort.sendTimingReq(retry_pkt)) {
      retryZPkts.pop();
      checkAndReleaseTickEvent();
      stallOnCacheRetry = (retryZPkts.size() > 0);
      if (stallOnCacheRetry) {
         //schedule the rest of waiting requests in the following cycle
         schedule(zcacheRetryEvent, nextCycle());
      } else {
         //no more retry pkts, check if early Z is done for the current batch
         if(doneEarlyZPending){
            doneEarlyZ();
         }
      }
   } else {
      //if the request fails then a retry is expected
      DPRINTF(ZUnit, "Z retry, paddr: 0x%x failed, waiting for another retry\n",
            retry_pkt->req->getPaddr());
   }
}

void ZUnit::sendZTransReq(DepthFragmentTile::DepthFragment * df){
   DPRINTF(ZUnit,
         "Sending a translation for fragment (%d,%d), vaddr: 0x%x, size: %d, addr: 0x%x\n",
         df->getX(), df->getY(), df->getDepthVaddr(), (int)depthSize, df->getDepthVaddr());

   RequestPtr req = new Request();
   Request::Flags flags;
   const int asid = 0;

   BaseTLB::Mode mode = BaseTLB::Read;
   req->setVirt(asid, df->getDepthVaddr(), (int)depthSize, flags, zcacheMasterId, 0);
   req->setFlags(Request::Z_FETCH);
   req->setExtraData((uint64_t)df);

   WholeTranslationState *state =
      new WholeTranslationState(req, NULL, NULL, mode);
   DataTranslation<ZUnit*> *translation
      = new DataTranslation<ZUnit*>(this, state);

   ztb->beginTranslateTiming(req, translation, mode, cudaGPU->getGraphicsTC());
   pendingTranslations++;
}


void ZUnit::finishTranslation(WholeTranslationState *state) { 
   if (state->getFault() != NoFault) { 
      panic("Z-cache translation encountered fault (%s) for address 0x%x", state->getFault()->name(), state->mainReq->getVaddr()); 
   } 

   pendingTranslations--;
   checkAndReleaseTickEvent();
   DepthFragmentTile::DepthFragment* df = (DepthFragmentTile::DepthFragment *) state->mainReq->getExtraData();

   DPRINTF(ZUnit, "Finished translation for fragment(%d,%d), vaddr=%llx ==> paddr=%llx\n", df->getX(), df->getY(), 
         state->mainReq->getVaddr(), state->mainReq->getPaddr());

   assert(state->mode == BaseTLB::Read);
   PacketPtr pkt = new Packet(state->mainReq, MemCmd::ReadReq);
   pkt->allocate();
   pushRequest(pkt);
   delete state;
}

void ZUnit::pushRequest(PacketPtr pkt){
   if (!stallOnCacheRetry) {
      sendZcacheAccess(pkt);
   } else {
      DPRINTF(ZUnit, "Z-cache blocked, paddr=%llx\n", pkt->req->getPaddr());
      retryZPkts.push(pkt);
   }
}

void ZUnit::sendZcacheAccess(PacketPtr pkt){
   assert(!stallOnCacheRetry);

   DPRINTF(ZUnit,
         "Sending z-fetch of %d bytes to paddr: 0x%x\n",
         pkt->getSize(), pkt->req->getPaddr());
   Addr addr = pkt->req->getPaddr();
   if(blockedLineAddrs[addr]){
      blockedAccesses[addr].push(pkt);
      blockedCount++;
   } else {
      DPRINTF(ZUnit, "sendTimingReq to the z-cache, type write=%d @ cycle %d\n", pkt->cmd == MemCmd::WriteReq, curCycle());
      if (!zcachePort.sendTimingReq(pkt)) {
         stallOnCacheRetry = true;
         if (pkt != retryZPkts.front()) {
            retryZPkts.push(pkt);
         }
         DPRINTF(ZUnit, "Send failed paddr: 0x%x. Waiting: %d\n",
               pkt->req->getPaddr(), retryZPkts.size());
      }
   }
   numZCacheRequests++;
}

void ZUnit::sendZWrite(DepthFragmentTile::DepthFragment * df){
   Request::Flags flags = Request::Z_FETCH;
   RequestPtr req = new Request(df->getDepthPaddr(), (int)depthSize, flags, zcacheMasterId);
   req->setExtraData((uint64_t)df);

   PacketPtr pkt = new Packet(req, MemCmd::WriteReq);
   uint64_t val = df->getDepthVal();
   if(depthSize == DepthSize::Z16){
      assert(val <= UINT16_MAX);
      uint16_t *depth = new uint16_t;
      *depth = val;
      pkt->dataDynamic(depth);
   } else if(depthSize == DepthSize::Z32){ 
      assert(val <= UINT32_MAX);
      uint32_t *depth = new uint32_t;
      *depth = val;
      pkt->dataDynamic(depth);
   } else {
      fatal("Unkown depth size \n");
   }

   DPRINTF(ZUnit, "Writing to location (%d,%d), depth value= %llx\n", df->getX(), df->getY(), val);

   pushRequest(pkt);
}

void ZUnit::regStats(){
   numZCacheRequests
      .name(name() + ".z_cache_requests")
      .desc("Number of z-cache requests sent")
      ;
   numZCacheRetry
      .name(name() + ".z_cache_retries")
      .desc("Number of z-cache retries")
      ;
}

void ZUnit::startEarlyZ(uint64_t depthBuffStart, uint64_t depthBuffEnd, unsigned bufWidth, RasterTiles* tiles, DepthSize dSize, GLenum _depthFunc){
   depthAddrStart = depthBuffStart;
   depthAddrEnd = depthBuffEnd;
   depthSize = dSize;
   depthFunc = _depthFunc;
   //if this case we kill all the fragments; TODO: check that the shader doesn't modify the depth
   if(depthFunc == GL_NEVER){
      return;
   }

   // this case needs a special handling
   assert(depthFunc != GL_ALWAYS);
   fragTiles = tiles;
   depthTiles.resize(tiles->size());
   currTile = 0;
   currFragment = 0;
   doneFlag = false;

   for(int i=0; i< tiles->size(); i++){
      depthTiles[i].setId(i);
      depthTiles[i].setRasterTile((*tiles)[i]);
      for(int j=0; j< (*tiles)[i]->size();  j++){
         unsigned xPos = (*(*tiles)[i])[j].intPos[0];
         unsigned yPos = (*(*tiles)[i])[j].intPos[1];
         unsigned zPos = (*(*tiles)[i])[j].intPos[3];
         Addr addr = depthAddrEnd - (yPos * bufWidth * (unsigned)depthSize) + (xPos * (unsigned)depthSize);
         assert((addr >= depthAddrStart) and (addr < depthAddrEnd));
         fragmentData_t * rasterFrag =  &((*(*tiles)[i])[j]);
         DepthFragmentTile::DepthFragment df(j, addr, zPos, &depthTiles[i], rasterFrag);
         depthTiles[i].addFragment(df);
         totalFragments++;
      }
   }

   DPRINTF(ZUnit, "Total tiles %d, fragments = %d\n", depthTiles.size(), totalFragments);

   schedule(tickEvent, nextCycle());
}

void ZUnit::checkAndReleaseTickEvent(){
   if(zPendingOnCache){
      zPendingOnCache = false;
      if(!tickEvent.scheduled())
         schedule(tickEvent, nextCycle());
   }
}

void ZUnit::tick(){
   if((pendingTranslations + blockedCount + retryZPkts.size()) < maxPendingReqs){

      //prioritize pending udpates
      if(depthUpdateQ.size() > 0) { 
         sendZWrite(depthUpdateQ.front());
         unblockZAccesses(depthUpdateQ.front()->getDepthPaddr());
         depthUpdateQ.pop();
         if(doneEarlyZPending and depthUpdateQ.empty()){
            doneEarlyZ();
         } else {
            schedule(tickEvent, nextCycle());
         }
      } else if(!doneFlag){
         //skipping empty tiles
         while(depthTiles[currTile].isEmpty() and (currTile < depthTiles.size())){
            currTile++;
            doneTiles++;
         }

         if(currTile == depthTiles.size()){
            doneFlag = true;
            return;
         }

         DepthFragmentTile::DepthFragment * df = depthTiles[currTile].getFragment(currFragment);
         currFragment++;
         if(currFragment == depthTiles[currTile].size()){
            currFragment = 0;
            currTile++;
         }

         if(currTile == depthTiles.size()){
            doneFlag = true;
         }

         sendZTransReq(df);
         schedule(tickEvent, nextCycle());
      }
   } else {
      zPendingOnCache = true;
   }
}

void ZUnit::doneEarlyZ(){
   if(!retryZPkts.empty() or !depthUpdateQ.empty()){
      //some requests are pending
      doneEarlyZPending = true;
      return;
   }

   DPRINTF(ZUnit, "Received early-Z done\n");
   printStats();
   doneEarlyZPending = false;
   assert(pendingTranslations == 0);
   assert(totalFragments == doneFrags);
   totalFragments = 0;
   stallOnCacheRetry = false;
   zPendingOnCache = false;
   doneTiles = 0;
   doneFrags = 0;
   depthAddrStart = 0;
   depthAddrEnd = 0;
   assert(blockedCount == 0);
   blockedAccesses.clear();
   blockedLineAddrs.clear();
   depthTiles.clear();
   g_renderData.doneEarlyZ();
}

void ZUnit::printStats(){
   DPRINTF(ZUnit, "-------------------Z Stats-----------------------\n");
   DPRINTF(ZUnit, "retryZPkts.size()=%d\n", retryZPkts.size());
   DPRINTF(ZUnit, "blockedCount=%d\n", blockedCount);
   DPRINTF(ZUnit, "pendingTranslations=%d\n", pendingTranslations);
   DPRINTF(ZUnit, "depthUpdateQ.size()=%d\n", depthUpdateQ.size());
   DPRINTF(ZUnit, "-------------------------------------------------\n");
}

ZUnit *ZUnitParams::create() {
   return new ZUnit(this);
}
