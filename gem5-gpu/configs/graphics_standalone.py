# Copyright (c) 2018 University of British Columbia
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
# Author: Ayub A. Gubran

import optparse
import m5
from m5.objects import *
from m5.defines import buildEnv
from m5.util import addToPath
import os, optparse, sys

addToPath('../')
addToPath('../../gem5/configs/common')
addToPath('../../gem5/configs/ruby')
addToPath('../../gem5/configs/topologies')
addToPath('../../gem5/configs')

import Options
import GPUConfig
import GPUMemConfig
import Simulation
import MemConfig

# Get paths we might need.  It's expected this file is in m5/configs/example.
#config_path = os.path.dirname(os.path.abspath(__file__))

parser = optparse.OptionParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)
GPUConfig.addGPUOptions(parser)
GPUMemConfig.addMemCtrlOptions(parser)
#Options.addNoISAOptions(parser)

parser.add_option("--sim-cycles", type="int", default=1000,
                  help="Number of simulation cycles")
parser.add_option("--gtrace", type="string", default="",
                   help="apitrace trace file")

(options, args) = parser.parse_args()

if options.gtrace == "":
   print "Error: no trace (--gtrace) specified"
   exit(1)

options.gpgpusim_config  = "gpu_config2"
options.config_icnt  = "config_soc2.icnt"
options.clusters = 6
options.cores_per_cluster = 1
options.ctas_per_shader = 8
options.gpu_warp_size = 32
options.gpu_threads_per_core = 2048
options.gpu_core_clock = "1000MHz"
options.gpu_core_config = "Maxwell"
#options.sys_clock = "1200MHz" #from the file above
options.gpu_dram_clock = "800Mhz"
options.mem_type = "LPDDR3_1600_1x32"
options.mem_channels = 4

options.g_standalone_mode = True
options.mem_size = "1GB"
options.g_raster_tw = 4 #1024;
options.g_raster_th = 4 #768;
options.g_raster_bw = 4 #1024;
options.g_raster_bh = 4 #768;

options.g_setup_delay = 8
options.g_setup_q = 1000
options.g_coarse_tiles = 1
options.g_fine_tiles = 1
options.g_hiz_tiles = 1
options.g_tc_engines = 2
options.g_tc_bins = 4
options.g_tc_h = 2
options.g_tc_w = 2
#options.g_tc_block_dim = 2
options.g_tc_thresh = 20
options.g_wg_size = 64
#options.gpgpusim_stats = True
options.drawcall_stats = True

#should be set by the gpgpusim config file anyway

#icache
options.sc_il1_size = "32kB"
options.sc_il1_assoc = 4
#color (dl1)
options.sc_l1_size = "32kB"
options.sc_l1_assoc = 8
options.sc_l1_buf_depth = 24
#texture
options.sc_tl1_size = "48kB"
options.sc_tl1_assoc = 24
options.sc_tl1_buf_depth = 32
#zcache
options.sc_zl1_size = "32kB"
options.sc_zl1_assoc= 8
options.gpu_zl1_buf_depth = 96
#l2 cache
options.sc_l2_size = "2048kB"
options.sc_l2_assoc = 32

#options.flush_kernel_end = True
options.shMemDelay = 1
options.cacheline_size = 128

#for fs mode
options.gpu_l1_pagewalkers = 1
options.gpu_tlb_entries = 8
options.gpu_tlb_assoc = 8
options.gpu_ttlb_entries = 8
options.gpu_ttlb_assoc = 8
options.pwc_size = "128kB"
options.pwc_assoc = 4



if args:
     print "Error: script doesn't take any positional arguments"
     sys.exit(1)

gpu_tracers = [ GraphicsStandalone(trace_path=options.gtrace) \
         for i in xrange(1) ]

# create the desired simulated system
system = System(cpu = gpu_tracers, mem_ranges = [AddrRange(options.mem_size)])
system.cache_line_size = options.cacheline_size


# Create a top-level voltage domain and clock domain
system.voltage_domain = VoltageDomain(voltage = options.sys_voltage)

system.clk_domain = SrcClockDomain(clock = options.sys_clock,
                                   voltage_domain = system.voltage_domain)

'''Ruby.create_system(options, False, system)

# Create a seperate clock domain for Ruby
system.ruby.clk_domain = SrcClockDomain(clock = options.ruby_clock,
                                        voltage_domain = system.voltage_domain)

i = 0
for ruby_port in system.ruby._cpu_ports:
     #
     # Tie the cpu test ports to the ruby cpu port
     #
     cpus[i].test = ruby_port.slave
     i += 1
'''

MemClass = Simulation.setMemClass(options)
system.membus = SystemXBar()
system.system_port = system.membus.slave
#CacheConfig.config_cache(options, system)
MemConfig.config_mem(options, system)

 
options.access_host_pagetable = True
gpu_mem_range = AddrRange(options.mem_size)
system.gpu = GPUConfig.createGPU(options, gpu_mem_range)
GPUConfig.connectGPUPorts_classic(system, system.gpu, options)

system.gpu.l2cache.write_buffers = 128
system.gpu.l2cache.mshrs = 128
system.gpu.l2cache.tgts_per_mshr = 20
system.membus.width = 128
system.gpu.l2NetToL2.width = 128
system.membus.clk_domain = SrcClockDomain(clock = "2GHz",
                                   voltage_domain = system.voltage_domain)
system.gpu.l2NetToL2.clk_domain = SrcClockDomain(clock = "2GHz",
                                   voltage_domain = system.voltage_domain)

# -----------------------
# run simulation
# -----------------------

root = Root(full_system = False, system = system)
root.system.mem_mode = 'timing'

# Not much point in this being higher than the L1 latency
m5.ticks.setGlobalFrequency('1ns')

# instantiate configuration
m5.instantiate()

# simulate until program terminates
exit_event = m5.simulate(options.abs_max_tick)

print 'Exiting @ tick', m5.curTick(), 'because', exit_event.getCause()
