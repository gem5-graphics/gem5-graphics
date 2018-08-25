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

class tc_engine_t {
   struct tc_fragment_t {
      tc_fragment_t(): covered(false), fragment(NULL)
      {}
      bool covered;
      RasterTile::rasterFragment_t* fragment;
   };

   public:
   tc_engine_t(unsigned tc_tile_h, unsigned tc_tile_w,
         unsigned r_tile_h, unsigned r_tile_w): 
      m_tc_tile_h(tc_tile_h), m_tc_tile_w(tc_tile_w),
      m_r_tile_h(r_tile_h), m_r_tile_w(r_tile_w)
   {
      //raster tiles should be made out of quads
      assert(m_r_tile_h%2 == 0 and m_r_tile_w%2 == 0);
      unsigned rtiles_count = m_tc_tile_h * m_tc_tile_w;
      unsigned per_tile = m_r_tile_h * m_r_tile_w;
      m_fragments.resize(rtiles_count*per_tile);
      m_covered_quads.resize(m_fragments.size()/4);
   }

   void set_curr_coord(unsigned x, unsigned y){
      //only possible to (re)assign empty tiles
      assert(m_tiles_bin.size() == 0);
      m_rtile_xstart = x;
      m_rtile_xend = x + m_tc_tile_w - 1;
      m_rtile_ystart = y;
      m_rtile_yend = y + m_tc_tile_h - 1;
   }

   //check if raster tile is mapped to this bin
   bool has_tile(unsigned x, unsigned y){
      if(x >= m_rtile_xstart 
            and x <= m_rtile_xend
            and y >= m_rtile_ystart
            and y <= m_rtile_yend)
         return true;
      return false;
   }

   private:
   //for x and y directions
   std::vector<tc_fragment_t> m_fragments;
   std::vector<bool> m_covered_quads;
   std::vector<RasterTile*> m_tiles_bin;
   //tc tile size in raster tiles
   const unsigned m_tc_tile_h;
   const unsigned m_tc_tile_w;
   //raster tile size in fragments
   const unsigned m_r_tile_h;
   const unsigned m_r_tile_w;
   //the current tile coord will range 
   //from (rx_coord, ry_coord) to (rx_coord + m_tc_tile_w, ry_coord + m_tc_tile_h)
   unsigned m_rtile_xstart; 
   unsigned m_rtile_xend;
   unsigned m_rtile_ystart;
   unsigned m_rtile_yend;
};

class tile_assembly_stage_t {
   public:
   tile_assembly_stage_t(unsigned _tc_bins, 
         unsigned tc_tile_h, unsigned tc_tile_w,
         unsigned r_tile_h, unsigned r_tile_w): 
      tc_engines(_tc_bins, tc_engine_t(tc_tile_h, tc_tile_w, r_tile_h, r_tile_w)), 
      tc_bins(_tc_bins)
   {}
   private:
   std::vector<tc_engine_t> tc_engines;
   const unsigned tc_bins;
};

class graphics_simt_pipeline {
   private:
      struct primitive_data_t{
         primitive_data_t(primitiveFragmentsData_t* _prim):
            prim(_prim)
         {}
         primitiveFragmentsData_t* const prim;
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
      m_hiz_pipe = new fifo_pipeline<RasterTile>("hiz-stage", 0, 5);
      m_f_raster_pipe = new fifo_pipeline<RasterTile>("fine-raster-stage", 0, 5);
      m_zunit_pipe = new fifo_pipeline<RasterTile>("zunit-stage", 0, 5);
      m_ta_pipe = new fifo_pipeline<RasterTile>("tile-assembly-stage", 0, 5);
      m_current_c_tile = 0;
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
         if(m_c_raster_pipe->empty()) return;
         unsigned processed_tiles = 0;    
         primitive_data_t* prim = m_c_raster_pipe->top();
         assert(prim);
         for(unsigned t=m_current_c_tile;
               t< prim->prim->getSimtTiles(m_cluster_id).size(); t++){
            if(m_hiz_pipe->full()) return;
            RasterTile* tile = prim->prim->getSimtTiles(m_cluster_id)[t];
            if(tile->size() > 0)
               m_hiz_pipe->push(tile);
            m_current_c_tile++;
            processed_tiles++;
            if(processed_tiles == m_c_tiles_per_cycle)
               break;
         }
         if(m_current_c_tile == prim->prim->getSimtTiles(m_cluster_id).size()){
            m_c_raster_pipe->pop();
            m_current_c_tile = 0;
         }
      }

      void run_hiz(){
         if(m_hiz_pipe->empty()) return;
         if(m_f_raster_pipe->full()) return;
         RasterTile* tile = m_hiz_pipe->top();
         assert(tile);
         assert(tile->size() > 0);
         m_f_raster_pipe->push(tile);
         m_hiz_pipe->pop();
      }

      void run_f_raster(){
         for(unsigned processed_tiles=0; processed_tiles < m_f_tiles_per_cycle;
               processed_tiles++){
            if(m_zunit_pipe->full()) return;
            if(m_f_raster_pipe->empty()) return;
            RasterTile* tile = m_f_raster_pipe->top();
            assert(tile);
            assert(tile->size() > 0);
            m_zunit_pipe->push(tile);
            m_f_raster_pipe->pop();
         }
      }

      void run_z_unit(){
         if(m_f_raster_pipe->empty()) return;
         if(m_ta_pipe->full()) return;
         RasterTile* tile = m_f_raster_pipe->top();
         assert(tile);
         assert(tile->size() > 0);
         m_ta_pipe->push(tile);
         m_f_raster_pipe->pop();
      }

      void run_tile_assembly(){
         if(m_ta_pipe->empty()) return;
      }


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
      fifo_pipeline<RasterTile>* m_hiz_pipe;
      fifo_pipeline<RasterTile>* m_f_raster_pipe;
      fifo_pipeline<RasterTile>* m_zunit_pipe;
      fifo_pipeline<RasterTile>* m_ta_pipe;
      unsigned m_current_c_tile;

      //performance configs
      const unsigned m_c_tiles_per_cycle;
      const unsigned m_f_tiles_per_cycle;
};


#endif /* GRAPHICS_PIPELINE */
