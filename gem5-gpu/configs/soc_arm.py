#Author: Ayub Gubran
#University of British Columbia

# Copyright (c) 2009-2012 Advanced Micro Devices, Inc.
# Copyright (c) 2012-2013 Mark D. Hill and David A. Wood
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
# Authors: Jason Power, Joel Hestness

import optparse
import os
import sys
from os.path import join as joinpath

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.util import addToPath, fatal

addToPath('../../gem5/configs/common')
addToPath('../../gem5/configs/ruby')
addToPath('../../gem5/configs/topologies')
addToPath('gpu_protocol')

import GPUConfig
import GPUMemConfig
import Options
import Ruby
import Simulation

from FSConfig import *
from SysPaths import *
from Benchmarks import *

parser = optparse.OptionParser()
Options.addCommonOptions(parser)
Options.addFSOptions(parser)
GPUConfig.addGPUOptions(parser)
GPUMemConfig.addMemCtrlOptions(parser)

#
# Add the ruby specific and protocol specific options
#
Ruby.define_options(parser)

(options, args) = parser.parse_args()

#system options
options.ruby = True
#options.cpu_type = "detailed" #for an OOO core, timing for an inorder core
options.num_cpus = 4
options.cpu_clock = '1.2GHz'


options.l1d_size = "32kB"
options.l1d_assoc = 4
options.l1i_size = "32kB"
options.l1i_assoc = 4
options.l2_size = "1MB"

###########################################################################
#GPU configs
options.gpgpusim_config  = "gpu_soc.config"
#We pin graphics memory so graphics accesses should not page fault. 
options.access_host_pagetable = True
options.kernel_stats = True

#should be set by the gpgpusim config file anyway
options.clusters = 6
options.cores_per_cluster = 1
options.gpu_core_clock = "425MHz"
options.ctas_per_shader = 8
options.gpu_warp_size = 32
options.gpu_threads_per_core = 1536

options.sc_l1_size = "16kB"
options.sc_l1_assoc = 4
options.sc_l1_buf_depth = 24

options.sc_tl1_size = "16kB"
options.sc_tl1_assoc = 4
options.sc_tl1_buf_depth = 24

options.gpu_l1_pagewalkers = 12
options.gpu_tlb_entries = 8
options.gpu_tlb_assoc = 8 
options.gpu_ttlb_entries = 8
options.gpu_ttlb_assoc = 8 

options.gpu_num_l2caches = 1
options.sc_l2_size = "256kB"
options.sc_l2_assoc = 8
#options.gpu_l2_resource_stalls = ? default: False

options.pwc_size = "1kB"
options.pwc_assoc = 4
options.pwc_policy = "LRU"
options.flush_kernel_end = True

#gpu memory
options.shMemDelay = 1
#gpu_mem_* options are used in split mode ignoring here

#System memory conifg tbd
options.mem_type = "RubyLPDDR3_1600_x32"
options.total_mem_size = "2112MB"
options.num_dev_dirs = 0
options.num_dirs = 2

if args:
    print "Error: script doesn't take any positional arguments"
    sys.exit(1)

if buildEnv['TARGET_ISA'] != "arm":
    fatal("gem5-gpu : this config works with an arm system!")
#
# CPU type configuration
#
if options.cpu_type != "timing" and options.cpu_type != "detailed":
    print "Warning: gem5-gpu only works with timing and detailed CPUs. Defaulting to timing"
    options.cpu_type = "timing"

print "Running Ruby with %s CPU model" % options.cpu_type

(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(options)

#
# Memory space configuration
#
(cpu_mem_range, gpu_mem_range) = GPUConfig.configureMemorySpaces(options)

#
# Setup benchmark to be run
#
if options.benchmark:
    try:
        bm = Benchmarks[options.benchmark]
    except KeyError:
        print "Error benchmark %s has not been defined." % options.benchmark
        print "Valid benchmarks are: %s" % DefinedBenchmarks
        sys.exit(1)
else:
    bm = [SysConfig(disk=options.disk_image)]
    bm[0].memsize = '%dB' % cpu_mem_range.size()

# Hard code the cache block width to 128B for now
# TODO: Remove this if/when block size can be different than 128B
if options.cacheline_size != 128:
    print "Warning: Only block size currently supported is 128B. Defaulting to 128."
    options.cacheline_size = 128

#allow host page table accesses to allow GPU to access the timing TLB.
#We pin graphics memory so graphics accesses should not page fault. 
options.access_host_pagetable = True

#
# Instantiate system
#
system = makeArmSystem(test_mem_mode, options.machine_type, bm[0],
                                 options.dtb_filename,
                                 bare_metal=False, ruby=True)

if options.enable_context_switch_stats_dump:
    test_sys.enable_context_switch_stats_dump = True

# Set the cache line size for the entire system
system.cache_line_size = options.cacheline_size

# Create a top-level voltage domain
system.voltage_domain = VoltageDomain(voltage = options.sys_voltage)

# Create a source clock for the system and set the clock period
system.clk_domain = SrcClockDomain(clock = options.sys_clock,
                               voltage_domain = system.voltage_domain)

# Create a CPU volatage domain
system.cpu_voltage_domain = VoltageDomain()

# Create a source clock for the CPUs and set the clock peroid
system.cpu_clk_domain = SrcClockDomain(clock = options.cpu_clock,
                                voltage_domain = system.cpu_voltage_domain)

if options.kernel is not None:
    system.kernel = binary(options.kernel)

if options.script is not None:
    system.readfile = options.script

if options.lpae:
    system.have_lpae = True

# Assign all the CPUs to the same clock domain
system.cpu = [CPUClass(cpu_id = i, clk_domain = system.cpu_clk_domain)
              for i in xrange(options.num_cpus)]

Simulation.setWorkCountOptions(system, options)

#
# Create the GPU
#
system.gpu = GPUConfig.createGPU(options, gpu_mem_range)

# Create the appropriate memory controllers and connect them to the
# PIO bus
system.mem_ctrls = [SimpleMemory(range = r) for r in system.mem_ranges]
for i in xrange(len(system.mem_ctrls)):
    system.mem_ctrls[i].port = system.iobus.master

#
# Setup Ruby
#
system.ruby_clk_domain = SrcClockDomain(clock = options.ruby_clock,
                                        voltage_domain = system.voltage_domain)
Ruby.create_system(options, system, system.iobus, system._dma_ports)

system.gpu.ruby = system.ruby
system.ruby.clk_domain = system.ruby_clk_domain

#
# Connect CPU ports
#
for (i, cpu) in enumerate(system.cpu):
    ruby_port = system.ruby._cpu_ports[i]

    cpu.clk_domain = system.cpu_clk_domain
    cpu.createThreads()
    cpu.createInterruptController()
    #
    # Tie the cpu ports to the correct ruby system ports
    #
    cpu.icache_port = system.ruby._cpu_ports[i].slave
    cpu.dcache_port = system.ruby._cpu_ports[i].slave
    if buildEnv['TARGET_ISA'] == "arm":
        cpu.itb.walker.port = system.ruby._cpu_ports[i].slave
        cpu.dtb.walker.port = system.ruby._cpu_ports[i].slave
    else:
        fatal("Not sure how to connect TLB walker ports in non-x86 system!")

    system.ruby._cpu_ports[i].access_phys_mem = True

#
# Connect GPU ports
#
GPUConfig.connectGPUPorts(system.gpu, system.ruby, options)

GPUMemConfig.setDRAMMemoryControlOptions(system, options)

#
# Finalize setup and run
#

root = Root(full_system = True, system = system)

if options.timesync:
    root.time_sync_enable = True

if options.frame_capture:
    VncServer.frame_capture = True

#m5.disableAllListeners()

Simulation.run(options, root, system, FutureClass)
