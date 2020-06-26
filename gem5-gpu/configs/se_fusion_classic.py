import optparse
import os
import sys
from os.path import join as joinpath

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.util import addToPath, fatal

addToPath('../../gem5/configs')
'''addToPath('../../gem5/configs/common')
addToPath('../../gem5/configs/ruby')
addToPath('../../gem5/configs/topologies')
addToPath('gpu_protocol')'''

from common import Options
from common import Simulation
from common import CacheConfig
from common import CpuConfig
from common import MemConfig
from common.Caches import *
from common import Simulation
import GPUConfig
import GPUMemConfig

parser = optparse.OptionParser()
GPUConfig.addGPUOptions(parser)
GPUMemConfig.addMemCtrlOptions(parser)
Options.addCommonOptions(parser)
Options.addSEOptions(parser)

parser.add_option("--sim-cycles", type="int", default=1000,
                  help="Number of simulation cycles")
parser.add_option("--gtrace", type="string", default="",
                   help="apitrace trace file")
#
# Add the ruby specific and protocol specific options
#
#Ruby.define_options(parser)

(options, args) = parser.parse_args()

# GPU configs
options.gpgpusim_config  = "example_gpu_config"
options.config_icnt  = "example_config.icnt"
options.clusters = 1
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

options.g_raster_tw = 8 #1024;
options.g_raster_th = 8 #768;
options.g_raster_bw = 8 #1024;
options.g_raster_bh = 8 #768;

options.g_setup_delay = 8
options.g_setup_q = 100000
options.g_coarse_tiles = 1

options.g_fine_tiles = 1
options.g_hiz_tiles = 1
options.g_tc_engines = 2
options.g_tc_bins = 4
options.g_tc_h = 2
options.g_tc_w = 2
options.g_tc_block_dim = 2
options.g_tc_thresh = 20
options.g_vert_wg_size = 64
options.g_frag_wg_size = 256
options.g_pvb_size = 16384
options.g_core_prim_pipe_size = 2
options.g_core_prim_delay = 4
#options.g_core_prim_warps = options.gpu_threads_per_core/options.gpu_warp_size
options.g_core_prim_warps = 5000

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
xbarFreq = "2GHz"


if args:
    print "Error: script doesn't take any positional arguments"
    sys.exit(1)

if buildEnv['TARGET_ISA'] not in ["x86", "arm"]:
    fatal("gem5-gpu SE doesn't currently work with non-x86 or non-ARM system!")

#
# CPU type configuration
#
if options.cpu_type != "timing" and options.cpu_type != "TimingSimpleCPU" \
    and options.cpu_type != "detailed" and options.cpu_type != "DerivO3CPU":
    print "Warning: gem5-gpu only known to work with timing and detailed CPUs: Proceed at your own risk!"
(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(options)

# If fast-forwarding, set the fast-forward CPU and mem mode for
# timing rather than atomic
if options.fast_forward:
    assert(CPUClass == AtomicSimpleCPU)
    assert(test_mem_mode == "atomic")
    CPUClass, test_mem_mode = Simulation.getCPUClass("TimingSimpleCPU")

#
# Memory space configuration
#
(cpu_mem_range, gpu_mem_range, total_mem_range) = GPUConfig.configureMemorySpaces(options)

#
# Setup benchmark to be run
#
process = Process()
process.executable = options.cmd
process.cmd = [options.cmd] + options.options.split()

if options.input != "":
    process.input = options.input
if options.output != "":
    process.output = options.output
if options.errout != "":
    process.errout = options.errout

# Hard code the cache block width to 128B for now
# TODO: Remove this if/when block size can be different than 128B
if options.cacheline_size != 128:
    print "Warning: Only block size currently supported is 128B. Defaulting to 128."
    options.cacheline_size = 128

#
# Instantiate system
#
system = System(cpu = [CPUClass(cpu_id = i,
                                workload = process)
                       for i in xrange(options.num_cpus)],
                mem_mode = test_mem_mode,
                mem_ranges = [cpu_mem_range],
                cache_line_size = options.cacheline_size)

# Create a top-level voltage domain
system.voltage_domain = VoltageDomain(voltage = options.sys_voltage)

# Create a source clock for the system and set the clock period
system.clk_domain = SrcClockDomain(clock = options.sys_clock,
                                   voltage_domain = system.voltage_domain)

# Create a CPU voltage domain
system.cpu_voltage_domain = VoltageDomain()

# Create a separate clock domain for the CPUs
system.cpu_clk_domain = SrcClockDomain(clock = options.cpu_clock,
                                       voltage_domain =
                                       system.cpu_voltage_domain)

Simulation.setWorkCountOptions(system, options)

#
# Create the GPU
#
vpo_mem_start = 0x80000000 #options.mem_size
system.gpu = GPUConfig.createGPU(options, gpu_mem_range, vpo_mem_start)

'''if options.caches or options.l2cache:
    # By default the IOCache runs at the system clock
    system.iocache = IOCache(addr_ranges = system.mem_ranges)
    system.iocache.cpu_side = system.iobus.master
    system.iocache.mem_side = system.membus.slave
elif not options.external_memory_system:
    system.iobridge = Bridge(delay='50ns', ranges = system.mem_ranges)
    system.iobridge.slave = system.iobus.master
    system.iobridge.master = system.membus.slave'''

'''
#
# Setup Ruby
#
system.ruby_clk_domain = SrcClockDomain(clock = options.ruby_clock,
                                        voltage_domain = system.voltage_domain)
Ruby.create_system(options, False, system)

system.gpu.ruby = system.ruby
system.ruby.clk_domain = system.ruby_clk_domain

if options.split:
    if options.access_backing_store:
        #
        # Reset Ruby's phys_mem to add the device memory range
        #
        system.ruby.phys_mem = SimpleMemory(range=total_mem_range,
                                            in_addr_map=False)

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
    cpu.itb.walker.port = system.ruby._cpu_ports[i].slave
    cpu.dtb.walker.port = system.ruby._cpu_ports[i].slave
    if buildEnv['TARGET_ISA'] == "x86":
        cpu.interrupts.pio = ruby_port.master
        cpu.interrupts.int_master = ruby_port.slave
        cpu.interrupts.int_slave = ruby_port.master'''


MemClass = Simulation.setMemClass(options)
system.membus = SystemXBar()
system.system_port = system.membus.slave
CacheConfig.config_cache(options, system)
MemConfig.config_mem(options, system)

#
# Connect GPU ports
#
GPUConfig.connectGPUPorts_classic(system, system.gpu, options)

#
# Finalize setup and run
#

root = Root(full_system = False, system = system)

m5.disableAllListeners()

Simulation.run(options, root, system, FutureClass)
