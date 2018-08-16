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

#include "delayqueue.h"


class graphics_simt_pipeline {
   private:
      struct primitive_data_t{
      };
   public:
      graphics_simt_pipeline( unsigned setup_delay, unsigned setup_q_len
            ){ 
         m_setup_pipe = new fifo_pipeline<primitive_data_t>("setup-stage", setup_delay, setup_q_len);
      }

      void cycle(){}
      void run_setup(){}
      void run_c_raster(){}
      void run_hiz(){}
      void run_f_raster(){}
      void run_tile_assembly(){}
      void run_z_unit(){}

      bool add_primitive(){
         assert(0);
         return false;
      }

   private:
      fifo_pipeline<primitive_data_t>* m_setup_pipe;
};


#endif /* GRAPHICS_PIPELINE */
