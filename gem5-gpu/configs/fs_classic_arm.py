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
#from os.path import join as joinpath

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.util import addToPath, fatal

addToPath('../../gem5/configs')

from common.FSConfig import *
from common.SysPaths import *
from common.Benchmarks import *
from common import Simulation
from common import CacheConfig
from common import MemConfig
from common import CpuConfig
from common.Caches import *
from common import Options
from common import O3_ARM_v7a

import GPUConfig
import GPUMemConfig

# from Caches import *

have_kvm_support = 'BaseKvmCPU' in globals()
def is_kvm_cpu(cpu_class):
    return have_kvm_support and cpu_class != None and \
        issubclass(cpu_class, BaseKvmCPU)

def cmd_line_template():
    if options.command_line and options.command_line_file:
        print "Error: --command-line and --command-line-file are " \
              "mutually exclusive"
        sys.exit(1)
    if options.command_line:
        return options.command_line
    if options.command_line_file:
        return open(options.command_line_file).read().strip()
    return None

parser = optparse.OptionParser()
GPUConfig.addGPUOptions(parser)
GPUMemConfig.addMemCtrlOptions(parser)
Options.addCommonOptions(parser)
Options.addFSOptions(parser)
(options, args) = parser.parse_args()


#system options
#options.gpgpusim_config  = "gpu_soc2.config"
#options.icnt_config  = "config_soc.icnt"
#options.cpu_type = "timing"
#options.cpu_type = "detailed" #for an OOO core, timing for an inorder core
#options.num_cpus = 1
options.cpu_clock = '20GHz'


options.l1d_size = "32kB"
options.l1d_assoc = 4
options.l1i_size = "32kB"
options.l1i_assoc = 4
options.l2_size = "1MB"

###########################################################################
#GPU configs
#options.gpgpusim_config  = "gpu_soc2.config"
#We pin graphics memory so graphics accesses should not page fault. 
#options.access_host_pagetable = True
#options.kernel_stats = True

options.g_raster_th = 16;
options.g_raster_tw = 16;
options.g_raster_bh = 16;
options.g_raster_bw = 16;

#should be set by the gpgpusim config file anyway
options.clusters = 2
options.cores_per_cluster = 2
#options.gpu_core_clock = "950MHz"
options.gpu_core_clock = "1GHz"
options.ctas_per_shader = 8
options.gpu_warp_size = 32
options.gpu_threads_per_core = 2048

#options.sc_l1_size = "16kB"
options.sc_l1_assoc = 4
options.sc_l1_buf_depth = 24

#options.sc_tl1_size = "16kB"
options.sc_tl1_assoc = 4
options.sc_tl1_buf_depth = 24

options.gpu_l1_pagewalkers = 1
options.gpu_tlb_entries = 8
options.gpu_tlb_assoc = 8
options.gpu_ttlb_entries = 8
options.gpu_ttlb_assoc = 8

options.gpu_num_l2caches = 1
#options.sc_l2_size = "256kB"
options.sc_l2_assoc = 8
#options.gpu_l2_resource_stalls = ? default: False

options.pwc_size = "1kB"
options.pwc_assoc = 4
#options.pwc_policy = "LRU"
options.flush_kernel_end = True

#gpu memory
options.shMemDelay = 1
#gpu_mem_* options are used in split mode ignoring here

#System memory conifg tbd
#options.mem_type = "RubyLPDDR3_1600_x32"
options.mem_type = "SimpleMemory"
#options.total_mem_size = "2112MB"
#options.total_mem_size = "3136MB"
options.num_dev_dirs = 0
#options.num_dirs = 1
options.mem_channels = options.num_dirs;
options.cacheline_size = 128

assert(options.gpu_num_l2caches == 1)

if args:
    print "Error: script doesn't take any positional arguments"
    sys.exit(1)

if buildEnv['TARGET_ISA'] != "arm":
    fatal("gem5-gpu : this config works with an arm system!")


if buildEnv['TARGET_ISA'] != "arm": 
   fatal("This is an ARM config script, please modify to use with other architectures!")

#if options.cpu_type != "timing" and options.cpu_type != "TimingSimpleCPU" \
#    and options.cpu_type != "detailed" and options.cpu_type != "DerivO3CPU":
#    print "Warning: gem5-gpu only known to work with timing and detailed CPUs: Proceed at your own risk!"
(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(options)

# Match the memories with the CPUs, based on the options for the test system
MemClass = Simulation.setMemClass(options)
print "Using %s memory model" % options.mem_type

#if(options.???.lower().count('emm')): #bm?
(cpu_mem_range, gpu_mem_range, total_mem_range) = GPUConfig.configureMemorySpaces(options)

gpu_mem_range = AddrRange('10GB', size='1GB')

if options.benchmark:
    try:
        bm = Benchmarks[options.benchmark]
    except KeyError:
        print "Error benchmark %s has not been defined." % options.benchmark
        print "Valid benchmarks are: %s" % DefinedBenchmarks
        sys.exit(1)
else:
    if options.dual:
        bm = [SysConfig(disk=options.disk_image, rootdev=options.root_device,
                        mem=options.mem_size, os_type=options.os_type),
              SysConfig(disk=options.disk_image, rootdev=options.root_device,
                        mem=options.mem_size, os_type=options.os_type)]
    else:
        bm = [SysConfig(disk=options.disk_image, rootdev=options.root_device,
                        mem=options.mem_size, os_type=options.os_type)]

np = options.num_cpus


# Hard code the cache block width to 128B for now
# TODO: Remove this if/when block size can be different than 128B
#if options.cacheline_size != 128:
#    print "Warning: Only block size currently supported is 128B. Defaulting to 128."
#    options.cacheline_size = 128

#allow host page table accesses to allow GPU to access the timing TLB.
#We pin graphics memory so graphics accesses should not page fault. 
options.access_host_pagetable = True

#
# Instantiate system
#
#system = makeArmSystem(test_mem_mode, options.machine_type, options.num_cpus, bm[0],
#                                 options.dtb_filename, bare_metal=False, cmdline=cmd_line_template(),
#                                 ruby=True)

system = makeArmSystem(test_mem_mode, options.machine_type,
                          options.num_cpus, bm[0], options.dtb_filename,
                          bare_metal=options.bare_metal,
                          cmdline=cmd_line_template(),
                          external_memory=options.external_memory_system,
                          ruby=options.ruby)
if options.enable_context_switch_stats_dump:
    test_sys.enable_context_switch_stats_dump = True


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

if options.virtualisation:
    system.have_virtualization = True

system.init_param = options.init_param

# Assign all the CPUs to the same clock domain
system.cpu = [CPUClass(cpu_id = i, clk_domain = system.cpu_clk_domain)
              for i in xrange(np)]


if is_kvm_cpu(CPUClass) or is_kvm_cpu(FutureClass):
    system.kvm_vm = KvmVM()

Simulation.setWorkCountOptions(system, options)


#
# Create the GPU
#
system.gpu = GPUConfig.createGPU(options, gpu_mem_range)


if options.caches or options.l2cache:
    # By default the IOCache runs at the system clock
    system.iocache = IOCache(addr_ranges = system.mem_ranges)
    system.iocache.cpu_side = system.iobus.master
    system.iocache.mem_side = system.membus.slave
elif not options.external_memory_system:
    system.iobridge = Bridge(delay='50ns', ranges = system.mem_ranges)
    system.iobridge.slave = system.iobus.master
    system.iobridge.master = system.membus.slave

# Sanity check
if options.fastmem:
    if CPUClass != AtomicSimpleCPU:
        fatal("Fastmem can only be used with atomic CPU!")
    if (options.caches or options.l2cache):
        fatal("You cannot use fastmem in combination with caches!")

if options.simpoint_profile:
    if not options.fastmem:
        # Atomic CPU checked with fastmem option already
        fatal("SimPoint generation should be done with atomic cpu and fastmem")
    if np > 1:
        fatal("SimPoint generation not supported with more than one CPUs")

for i in xrange(np):
    if options.fastmem:
        system.cpu[i].fastmem = True
    if options.simpoint_profile:
        system.cpu[i].addSimPointProbe(options.simpoint_interval)
    if options.checker:
        system.cpu[i].addCheckerCpu()
    system.cpu[i].createThreads()

# If elastic tracing is enabled when not restoring from checkpoint and
# when not fast forwarding using the atomic cpu, then check that the
# CPUClass is DerivO3CPU or inherits from DerivO3CPU. If the check
# passes then attach the elastic trace probe.
# If restoring from checkpoint or fast forwarding, the code that does this for
# FutureCPUClass is in the Simulation module. If the check passes then the
# elastic trace probe is attached to the switch CPUs.
if options.elastic_trace_en and options.checkpoint_restore == None and \
    not options.fast_forward:
    CpuConfig.config_etrace(CPUClass, system.cpu, options)

CacheConfig.config_cache(options, system)

system.mem_ranges.append(gpu_mem_range)
MemConfig.config_mem(options, system)


print "ranges:"
for r in  system.mem_ranges:
   print r
print "total mem ranges: ", total_mem_range 



#system.gpu.ruby = system.ruby

#system.ruby.clk_domain = system.ruby_clk_domain

#
# Connect GPU ports
#
GPUConfig.connectGPUPorts_classic(system, system.gpu, options)

#if options.mem_type == "RubyMemoryControl":
#   GPUMemConfig.setMemoryControlOptions(system, options)

 
#
# Finalize setup and run
#

if len(bm) == 2:
    drive_sys = build_drive_system(np)
    root = makeDualRoot(True, system, drive_sys, options.etherdump)
elif len(bm) == 1 and options.dist:
    # This system is part of a dist-gem5 simulation
    root = makeDistRoot(system,
                        options.dist_rank,
                        options.dist_size,
                        options.dist_server_name,
                        options.dist_server_port,
                        options.dist_sync_repeat,
                        options.dist_sync_start,
                        options.ethernet_linkspeed,
                        options.ethernet_linkdelay,
                        options.etherdump);
elif len(bm) == 1:
    root = Root(full_system=True, system=system)
else:
    print "Error I don't know how to create more than 2 systems."
    sys.exit(1)

system.load_addr_mask = 0x7ffffff
#system.load_addr_mask = 0xfffffff

if options.timesync:
    root.time_sync_enable = True

if options.frame_capture:
    VncServer.frame_capture = True

Simulation.setWorkCountOptions(system, options)
Simulation.run(options, root, system, FutureClass)
