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
#include "graphics/zunit.hh"
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
   zropWidth(p->zrop_width),
   hizWidth(p->hiz_width),
   zcacheRetryEvent(this),
   depthResponseEvent(this),
   tickEvent(this)
{
   DPRINTF(ZUnit, "Created a ZUnit Interface\n");
   cudaGPU->registerZUnit(this);
   zPendingOnCache = false;
   doneFrags = 0;
   doneTiles = 0;
   //blockedCount = 0;
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
ZUnit::ZCachePort::recvReqRetry()
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

bool
ZUnit::depthTest(uint64_t oldDepthVal, uint64_t newDepthVal){
   bool returnVal = false;
   switch(depthFunc){
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
      default: 
         panic("Unsupported depth function %x\n", depthFunc);
   }

   return returnVal;
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

   //DepthFragmentTile::DepthFragment * df = (DepthFragmentTile::DepthFragment*) pkt->req->getExtraData();
   DepthFragmentTile::DepthFragment * df = ztable[pkt->req->getVaddr()];

   //remove from ztable
   ztable.erase(df->getDepthVaddr());

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
   bool pass = depthTest(oldDepthVal, newDepthVal);

   if(pass){
      df->setPassed();
      df->setDepthPaddr(pkt->req->getPaddr());
      df->setDepthVal(newDepthVal);
      depthUpdateQ.push(df);
      if(!tickEvent.scheduled())
         schedule(tickEvent, nextCycle());
   }

   /*else {
      unblockZAccesses(pkt->req->getPaddr());
   }*/

   if(df->getTile()->isDone()){
      doneTiles++;
      assert(doneTiles <= depthTiles.size());
      DPRINTF(ZUnit, "Done tile %d\n", df->getTile()->getId());
      g_renderData.launchFragmentTile(df->getTile()->getRasterTile(), df->getTile()->getId());
   }

   if(doneTiles == depthTiles.size()){
      doneEarlyZ();
   }
}

void
ZUnit::handleZcacheRetry(){
   assert(retryZPkts.size());

   numZCacheRetry++;

   PacketPtr retry_pkt = retryZPkts.front();
   DPRINTF(ZUnit, "Received z-cache retry, pkt is write=%d, paddr: 0x%x\n",
         retry_pkt->cmd==MemCmd::WriteReq, retry_pkt->req->getPaddr());


   DPRINTF(ZUnit, "sendTimingReq to the z-cache, type write=%d @ cycle %d\n", retry_pkt->cmd == MemCmd::WriteReq, curCycle());
   if (zcachePort.sendTimingReq(retry_pkt)) {
      retryZPkts.pop();
      checkAndReleaseTickEvent();
      if (retryZPkts.size() > 0) {
         //schedule the rest of waiting requests in the following cycle
         schedule(zcacheRetryEvent, nextCycle());
      } else {
         //no more retry pkts, check if early Z is done for the current batch
         doneEarlyZ();
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

#if TRACING_ON
   DepthFragmentTile::DepthFragment* df = (DepthFragmentTile::DepthFragment *) state->mainReq->getExtraData();
   DPRINTF(ZUnit, "Finished translation for fragment(%d,%d), vaddr=%llx ==> paddr=%llx\n", df->getX(), df->getY(), 
         state->mainReq->getVaddr(), state->mainReq->getPaddr());
#endif
   assert(state->mode == BaseTLB::Read);
   PacketPtr pkt = new Packet(state->mainReq, MemCmd::ReadReq);
   pkt->allocate();
   pushRequest(pkt);
   delete state;
}

void ZUnit::pushRequest(PacketPtr pkt){
      sendZcacheAccess(pkt);
}

void ZUnit::sendZcacheAccess(PacketPtr pkt){
   DPRINTF(ZUnit,
         "Sending z access of %d bytes to paddr: 0x%x\n",
         pkt->getSize(), pkt->req->getPaddr());
   DPRINTF(ZUnit, "sendTimingReq to the z-cache, type write=%d @ cycle %d\n", pkt->cmd == MemCmd::WriteReq, curCycle());
   if (!zcachePort.sendTimingReq(pkt)) {
      if (pkt != retryZPkts.front()) {
         retryZPkts.push(pkt);
      }
      DPRINTF(ZUnit, "Send failed paddr: 0x%x. Waiting: %d\n",
            pkt->req->getPaddr(), retryZPkts.size());
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

void ZUnit::startEarlyZ(uint64_t depthBuffStart, uint64_t depthBuffEnd, unsigned bufWidth, RasterTiles* tiles, DepthSize dSize, GLenum _depthFunc,
      uint8_t* depthBuffer, unsigned frameWidth, unsigned frameHeight, unsigned tileH, unsigned tileW, unsigned blockH, unsigned blockW, RasterDirection rasterDir){
   depthSize = dSize;
   depthAddrStart = depthBuffStart;
   depthAddrEnd = depthBuffEnd;
   depthFunc = _depthFunc;

   //if this case we kill all the fragments; TODO: check that the shader doesn't modify the depth
   if(depthFunc == GL_NEVER){
      return;
   }


   printf("number of tiles %zu\n", tiles->size());
   // this case needs a special handling
   assert(depthFunc != GL_ALWAYS);
   
   initHizBuffer(depthBuffer, frameWidth, frameHeight, dSize, tileW, tileH, blockH, blockW, rasterDir);
   fragTiles = tiles;
   depthTiles.resize(tiles->size());
   currTile = 0;
   currFragment = 0;
   doneFlag = false;

   for(int i=0; i< tiles->size(); i++){
      depthTiles[i].setId(i);
      depthTiles[i].setRasterTile((*tiles)[i]);
      depthTiles[i].hizDepth = (*(*tiles)[i])[0].intPos[2];
      for(int j=0; j< (*tiles)[i]->size();  j++){
         unsigned xPos = (*(*tiles)[i])[j].intPos[0];
         unsigned yPos = (*(*tiles)[i])[j].intPos[1];
         unsigned zPos = (*(*tiles)[i])[j].intPos[2];
         Addr addr = depthAddrEnd - ((yPos+1) * bufWidth * (unsigned)depthSize) + (xPos * (unsigned)depthSize);
         assert((addr >= depthAddrStart) and (addr < depthAddrEnd));
         fragmentData_t * rasterFrag =  &((*(*tiles)[i])[j]);
         DepthFragmentTile::DepthFragment df(j, addr, zPos, &depthTiles[i], rasterFrag);
         depthTiles[i].addFragment(df);
         if(depthTest(depthTiles[i].hizDepth, zPos)){
            depthTiles[i].hizDepth = zPos;
         }
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
   //prioritize pending udpates
   if(depthUpdateQ.size() > 0) { 
      sendZWrite(depthUpdateQ.front());
      //unblockZAccesses(depthUpdateQ.front()->getDepthPaddr());
      depthUpdateQ.pop();
      if(depthUpdateQ.empty()){
         doneEarlyZ();
      } else {
         schedule(tickEvent, nextCycle());
      }
   }


   bool bflag = false;
   for(int zw=0; zw < zropWidth; zw++){
      if(bflag) break; 
      if(hizQ.size() > 0){
         //only proceed if we have enough space in the ztable
         DepthFragmentTile * dt = hizQ.front();
         uint64_t hizThresh = dt->hizThresh();
         DepthFragmentTile::DepthFragment * df = dt->getFragment(currFragment);

         //first check if this fragment can even pass the hiZ value 
         uint64_t fragDepthVal = df->getDepthVal();
         if(!depthTest(hizThresh, fragDepthVal)){
            //fragment fail
            currFragment++;
            if(currFragment == dt->size()){
               currFragment = 0;
               hizQ.pop();
               bflag = true;
            }
         } else if((ztable.size() < maxPendingReqs) or (ztable.count(df->getDepthVaddr()) > 0)){
            currFragment++;
            if(currFragment == dt->size()){
               currFragment = 0;
               hizQ.pop();
               bflag = true;
            }

            //check if there is pending depth test to the same fragment position
            Addr dfVaddr = df->getDepthVaddr();
            if(ztable.count(dfVaddr) > 0){
               uint64_t newDepthVal = df->getDepthVal();
               uint64_t oldDepthVal = ztable[df->getDepthVaddr()]->getDepthVal();

               DepthFragmentTile::DepthFragment* done_df = NULL;
               if(depthTest(oldDepthVal, newDepthVal)){
                  DepthFragmentTile::DepthFragment* old_df = ztable[dfVaddr];
                  old_df->getTile()->incDoneFragments();
                  old_df->unsetPassed();
                  ztable[dfVaddr] = df;
                  done_df = old_df;
               } else{
                  df->getTile()->incDoneFragments();
                  df->unsetPassed();
                  done_df = df;
               }

               doneFrags++; //only one of the fragments will remain
               if(done_df->getTile()->isDone()){
                  doneTiles++;
                  assert(doneTiles <= depthTiles.size());
                  DPRINTF(ZUnit, "Done tile %d\n", done_df->getTile()->getId());
                  g_renderData.launchFragmentTile(done_df->getTile()->getRasterTile(), done_df->getTile()->getId());
               }
            } else {
               ztable[df->getDepthVaddr()] = df;
               sendZTransReq(df);
            }
            if(!tickEvent.scheduled())
               schedule(tickEvent, nextCycle());
         } else {
            zPendingOnCache = true;
         }
      } else {
         doneEarlyZ();
      }
   }


   //for now we actually handle one tile per cycle
   for(int hw=0; hw < hizWidth; hw++){
      if(!doneFlag){
         //skipping empty tiles
         while(depthTiles[currTile].isEmpty() and (currTile < depthTiles.size())){
            currTile++;
            doneTiles++;
         }

         if(currTile == depthTiles.size()){
            doneFlag = true;
            doneEarlyZ();
            return;
         }
         DepthFragmentTile * dt = &depthTiles[currTile];
         unsigned posId = dt->getRasterTile()->getTilePos();
         assert(posId < hizBuffer.size());
         if(depthFunc == GL_NOTEQUAL or depthFunc==GL_EQUAL){ 
            warn_once("Unsupported depth test (GL_NOTEQUAL or GL_EQUAL), skipping HiZ\n");
            //skip hiZ
            hizQ.push(dt);
         } else if(depthTest(hizBuffer[posId], dt->hizDepth)){
            dt->setHizThresh(hizBuffer[posId]);
            hizBuffer[posId] = dt->hizDepth;
            hizQ.push(dt);
         } else {
            //failed tile
            doneFrags+= dt->size();
         }
         currTile++;
         if(currTile == depthTiles.size()){
            doneFlag = true;
            doneEarlyZ();
            return;
         }
         if(!tickEvent.scheduled())
            schedule(tickEvent, nextCycle());
      }
   }
}

void ZUnit::doneEarlyZ(){
   if(!retryZPkts.empty() or !depthUpdateQ.empty() or !hizQ.empty()
         or (currTile != depthTiles.size())){
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
   zPendingOnCache = false;
   doneTiles = 0;
   doneFrags = 0;
   depthAddrStart = 0;
   depthAddrEnd = 0;
   depthTiles.clear();
   g_renderData.doneEarlyZ();
}

void ZUnit::initHizBuffer(uint8_t* depthBuffer, unsigned frameWidth, unsigned frameHeight, 
      DepthSize dSize, const unsigned tileW, const unsigned tileH,
      const unsigned blockH, const unsigned blockW, const RasterDirection rasterDir){
    
    //checking if a suitable block size is provided
    assert((blockH%tileH)==0);
    assert((blockW%tileW)==0);
   
    DPRINTF(ZUnit, "tileW = %d, and tileH = %d\n", tileW, tileH);
 
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
    }

    const unsigned frameDim = frameHeight * frameWidth;

    std::vector<uint64_t> depthValues(frameDim);
    std::vector<bool> touchedTiles(frameDim, false);

    if(dSize == DepthSize::Z16) {
       uint16_t* p = (uint16_t*) depthBuffer;
       for(unsigned i =0; i< frameDim; i++)
          depthValues[i] = p[i];
    } else if(dSize == DepthSize::Z32){
       uint32_t* p = (uint32_t*) depthBuffer;
       for(unsigned i =0; i< frameDim; i++)
          depthValues[i] = p[i];
    } else assert(0);
    
    const unsigned fragmentsPerTile = tileH * tileW;
    assert(0 == ((frameHeight* frameWidth) % fragmentsPerTile));
    unsigned tilesCount = frameDim / fragmentsPerTile;

    hizBuffer.resize(tilesCount);

    assert((frameWidth%tileW) == 0);
    assert((frameHeight%tileH) == 0);
            
    const unsigned tileRow = frameWidth / tileW;
    //const unsigned blockRow = frameWidth/blockW;
    //const unsigned hTilesPerBlock = blockW/tileW;
    //const unsigned vTilesPerBlock = blockH/tileH;

    for(unsigned i=0; i < depthValues.size(); i++){
       unsigned tileIdx = -1;
       unsigned xPos = i%frameWidth;
       unsigned yPos = i/frameWidth;
       unsigned tileXCoord = xPos/tileW;
       unsigned tileYCoord = yPos/tileH;
       if(rasterDir == HorizontalRaster){
          tileIdx = tileYCoord*tileRow + tileXCoord;
       } else if (rasterDir == BlockedHorizontal){
          assert(0); //TODO
       } else assert(0);

       assert(tileIdx < hizBuffer.size());
       if(touchedTiles[tileIdx]){
          if(depthTest(hizBuffer[tileIdx], depthValues[i])){
             hizBuffer[tileIdx] = depthValues[i];
          }
       } else {
          touchedTiles[tileIdx] = true;
          hizBuffer[tileIdx] = depthValues[i];
       }
    }
}

void ZUnit::printStats(){
   DPRINTF(ZUnit, "-------------------Z Stats-----------------------\n");
   DPRINTF(ZUnit, "retryZPkts.size()=%d\n", retryZPkts.size());
   //DPRINTF(ZUnit, "blockedCount=%d\n", blockedCount);
   DPRINTF(ZUnit, "pendingTranslations=%d\n", pendingTranslations);
   DPRINTF(ZUnit, "depthUpdateQ.size()=%d\n", depthUpdateQ.size());
   DPRINTF(ZUnit, "-------------------------------------------------\n");
}

ZUnit *ZUnitParams::create() {
   return new ZUnit(this);
}
