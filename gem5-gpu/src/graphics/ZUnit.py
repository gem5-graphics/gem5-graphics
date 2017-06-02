# Copyright (c) 2016 Ayub A. Gubran and Tor M. Aamodt
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from MemObject import MemObject
from ShaderTLB import ShaderTLB
from m5.defines import buildEnv
from m5.params import *
from m5.proxy import *

class ZUnit(MemObject):
    type = 'ZUnit'
    cxx_class = 'ZUnit'
    cxx_header = "graphics/zunit.hh"

    sys = Param.System(Parent.any, "system will run on")
    gpu = Param.CudaGPU(Parent.any, "The GPU")
    z_port = MasterPort("The z-cache port")
    ztb = Param.ShaderTLB(ShaderTLB(access_host_pagetable = True), "Zcache TLB");
    max_pending_reqs = Param.Int(1024, "Maximum pending cache requests") 
    depth_response_queue_size = Param.Int(1024, "Size of the depth response queue") 
    zrop_width = Param.Int(64, "ZROP throughput in samples/cycle");
    hiz_width = Param.Int(1, "HiZ throughput in tiles/cycle")
    id = Param.Int(-1, "ID of the CE")
    depth_test_delay = Param.Int(1, "Depth test delay") #TODO: to use in the z-unit for depth ops delay 
