// Copyright (c) 2018, Ayub A. Gubran, Tor M. Aamodt
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

#ifndef FIXED_GRAPHICS_PIPELINE_H
#define FIXED_GRAPHICS_PIPELINE_H

#include <map>
#include <set>
#include <vector>
#include <list>
#include <bitset>
#include <utility>
#include <algorithm>
#include <deque>
#include "graphics/mesa_gpgpusim.h"

#include "delayqueue.h"
#include "shader.h"


extern renderData_t g_renderData;

class graphics_simt_pipeline {
   private:
      struct primitive_data_t{
         primitive_data_t(primitiveFragmentsData_t* _prim):
            prim(_prim), c_start_tile(-1), c_end_tile(-1)
         {}
         primitiveFragmentsData_t* const prim;
         int c_start_tile;
         int c_end_tile;
      };
   public:
      graphics_simt_pipeline(unsigned simt_cluster_id,
            unsigned setup_delay, unsigned setup_q_len,
            unsigned c_tiles_per_cycle,
            unsigned f_tiles_per_cycle
            ): 
         m_cluster_id(simt_cluster_id),
         m_c_tiles_per_cycle(c_tiles_per_cycle),
         m_f_tiles_per_cycle(f_tiles_per_cycle)
   { 
      m_setup_pipe = new fifo_pipeline<primitive_data_t>("setup-stage", setup_delay, setup_q_len);
      m_c_raster_pipe = new fifo_pipeline<primitive_data_t>("coarse-raster-stage", 0, 2);
      m_hiz_pipe = new fifo_pipeline<primitive_data_t>("hiz-stage", 0, 5);
      m_f_raster_pipe = new fifo_pipeline<primitive_data_t>("fine-raster-stage", 0, 5);
      m_ta_pipe = new fifo_pipeline<primitive_data_t>("tile-assembly-stage", 0, 5);
      m_current_c_tile = 0;
      m_current_f_tile = 0;
   }

      ~graphics_simt_pipeline(){
         delete m_setup_pipe;
         delete m_c_raster_pipe;
      }

      void cycle(){
         printf("cycle gpipe\n");
         run_z_unit();
         run_tile_assembly();
         run_f_raster();
         run_hiz();
         run_c_raster();
         run_setup();
      }

      void run_setup(){
         primitive_data_t* prim = m_setup_pipe->top();
         if(prim){
            if(m_c_raster_pipe->full()) return;
            m_c_raster_pipe->push(prim);
            m_setup_pipe->pop();
         } else {
            m_setup_pipe->pop();
         }
      }

      void run_c_raster(){
         primitive_data_t* prim = m_c_raster_pipe->top();
         if(prim){
            if(m_hiz_pipe->full()) return;
            if((m_current_c_tile+m_c_tiles_per_cycle) >= 
                  prim->prim->getSimtTiles(m_cluster_id).size()){
               //last batch of c tiles
               prim->c_start_tile = m_current_c_tile;
               prim->c_end_tile = prim->prim->getSimtTiles(m_cluster_id).size() -1;
               m_hiz_pipe->push(prim);
               m_c_raster_pipe->pop();
               m_current_c_tile = 0;
            } else {
               primitive_data_t* new_prim = new primitive_data_t(*prim);
               new_prim->c_start_tile =  m_current_c_tile;
               new_prim->c_start_tile = m_current_c_tile + m_c_tiles_per_cycle -1;
               m_hiz_pipe->push(new_prim);
               m_current_c_tile+= m_c_tiles_per_cycle;
            }
         } else {
            m_c_raster_pipe->pop();
         }
      }

      void run_hiz(){
         primitive_data_t* prim = m_hiz_pipe->top();
         if(prim){
            if(m_f_raster_pipe->full()) return;
            m_f_raster_pipe->push(prim);
            m_hiz_pipe->pop();
         } else {
            m_c_raster_pipe->pop();
         }
      }
      void run_f_raster(){
         primitive_data_t* prim = m_f_raster_pipe->top();
         if(prim){
            if(m_ta_pipe->full()) return;
         } else {
            m_f_raster_pipe->pop();
         }
      }
      void run_tile_assembly(){

      }
      void run_z_unit(){}

      bool add_primitive(primitiveFragmentsData_t* prim, unsigned ctilesId){
         //this primitive doesn't touch this simt core
         if(prim->getSimtTiles(m_cluster_id).size() == 0)
            return true;
         if(m_setup_pipe->full())
            return false;
         primitive_data_t* prim_data = new primitive_data_t(prim);
         //prim_data->c_raster_delay = 
         m_setup_pipe->push(prim_data);
         return true;
      }

      //return if pipeline not empty
      unsigned get_not_completed(){
         bool empty = 
            m_setup_pipe->empty() and
            m_c_raster_pipe->empty();
         return !empty;
      }

   private:
      const unsigned m_cluster_id;
      fifo_pipeline<primitive_data_t>* m_setup_pipe;
      fifo_pipeline<primitive_data_t>* m_c_raster_pipe;
      fifo_pipeline<primitive_data_t>* m_hiz_pipe;
      fifo_pipeline<primitive_data_t>* m_f_raster_pipe;
      fifo_pipeline<primitive_data_t>* m_ta_pipe;
      unsigned m_current_c_tile;
      unsigned m_current_f_tile;

      //performance configs
      const unsigned m_c_tiles_per_cycle;
      const unsigned m_f_tiles_per_cycle;
};


#endif /* GRAPHICS_PIPELINE */
