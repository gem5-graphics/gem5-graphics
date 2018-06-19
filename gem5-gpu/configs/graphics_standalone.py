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

options.g_standalone_mode = True
options.mem_size = "1GB"
options.cacheline_size = 128

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
