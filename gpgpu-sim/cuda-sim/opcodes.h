// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda
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

#ifndef opcodes_h_included
#define opcodes_h_included

enum opcode_t {
#define OP_DEF(OP,FUNC,STR,DST,CLASSIFICATION) OP,
#include "opcodes.def"
   NUM_OPCODES
#undef OP_DEF
};

enum special_regs {
   CLOCK_REG,
   HALFCLOCK_ID,
   CLOCK64_REG,
   CTAID_REG,
   ENVREG_REG,
   GRIDID_REG,
   LANEID_REG,
   LANEMASK_EQ_REG,
   LANEMASK_LE_REG,
   LANEMASK_LT_REG,
   LANEMASK_GE_REG,
   LANEMASK_GT_REG,
   NCTAID_REG,
   NTID_REG,
   NSMID_REG,
   NWARPID_REG,
   PM_REG,
   SMID_REG,
   TID_REG,
   WARPID_REG,
   WARPSZ_REG,
   //////// START OF SPECIAL REGISTERS USED IN GRAPHICS ////////
   UTID_REG,

   //PART1: Vertex Result Registers: Used to write the vertex shading stage results
   VERTEX_POSITION,
   VERTEX_COLOR0,
   VERTEX_COLOR1,
   VERTEX_TEXCOORD0,
   VERTEX_TEXCOORD1,
   VERTEX_TEXCOORD2,
   VERTEX_TEXCOORD3,
   VERTEX_TEXCOORD4,
   VERTEX_TEXCOORD5,
   VERTEX_TEXCOORD6,
   VERTEX_TEXCOORD7,
   VERTEX_ACTIVE,
           
   //PART2: Fragment Input Registers: Used to read the fragment input attribs
   SHADER_IN0,
   SHADER_IN1,
   SHADER_IN2,
   SHADER_IN3,
   SHADER_COLOR0,
   SHADER_CONST00,
   SHADER_CONST01,
   SHADER_CONST02,
   SHADER_CONST03,
   FRAGMENT_ACTIVE,
   FQUAD_ACTIVE,
   SKIP_DEPTH_TEST,
   
   //PART3: other
   RB_WIDTH, //the width of the render buffer used to calculate the written to address
   RB_SIZE   //render buffer size in bytes
};

#endif
