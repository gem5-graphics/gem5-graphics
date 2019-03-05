// Copyright (c) 2019, Ayub A. Gubran, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include "graphics_models.h"
#include "gpu-sim.h"
#include "gpu/gpgpu-sim/cuda_gpu.hh"

bool graphics_simt_pipeline::add_prim_batch(std::vector<unsigned> coverage_batch){
   //busy processing a batch
   if(m_curr_coverage_masks.size() > 0){
      return false;
   }
   //create a list of masks to be sent to the other shader cores
   const unsigned clust_count = m_cluster->m_config->n_simt_clusters;
   std::vector<std::vector<std::pair<unsigned, bool> > >
      coverage_masks(clust_count);

   for(unsigned primId=0; primId<coverage_batch.size(); primId++){
      std::set<unsigned> coverage = 
         g_renderData.getClustersCoveredByPrim(coverage_batch[primId]);
      for(unsigned c=0; c<clust_count; c++){
         coverage_masks[c].push_back(
               {primId, coverage.count(c)>0? true: false});
      }
   }

   for(unsigned cid=0; cid<coverage_masks.size(); cid++)
      m_curr_coverage_masks.push_back(c_mask_t(cid, coverage_masks[cid]));
}

void graphics_simt_pipeline::run_out_prim_batch(){
   if(m_curr_coverage_masks.empty())
      return;
   c_mask_t* fm = &m_curr_coverage_masks.front();
   //if this mask is for this cluster no need to send it 
   //over the network
   if(fm->clusterId == m_cluster_id){
      if(add_primitives(fm->prims)){
         m_curr_coverage_masks.pop_front();
      }
   } else if(m_cluster->get_gpu()->gem5CudaGPU->sendPrimMaskBatch(
            m_cluster_id,
            fm->clusterId, fm->prims)){
      m_curr_coverage_masks.pop_front();
   }
}

bool graphics_simt_pipeline::add_primitives(
      std::vector<std::pair<unsigned, bool> >primIds){
   //TODO: now we accept all mask packets
   //should make it configurable size so backpressure is created
   //e.g., 1-2 slots per source cluster
   for(auto val: primIds){
      m_curr_mapped_prims[val.first] = val.second;
   }
   return true;
}

void graphics_simt_pipeline::run_fetch_prim_attribs(){
   //if we cannot find the primitive status wait until
   //it's received
   if(m_curr_mapped_prims.find(m_curr_prim_counter)==
         m_curr_mapped_prims.end())
      return;
   //if the prim is not covered by this cluster then
   //skip to the next one
   if(!m_curr_mapped_prims[m_curr_prim_counter]){
      m_curr_mapped_prims.erase(m_curr_prim_counter);
      m_curr_prim_counter++;
   } else if(m_cluster->get_gpu()->gem5CudaGPU->fetchPrimAttribs(
               m_cluster_id, m_curr_prim_counter)){
         m_curr_mapped_prims.erase(m_curr_prim_counter);
         m_curr_prim_counter++;
   }
}
