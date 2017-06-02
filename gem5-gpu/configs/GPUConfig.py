# Copyright (c) 2012 Mark D. Hill and David A. Wood
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

import m5
import os
import re
from m5.objects import *
from m5.util.convert import *
from m5.util import fatal

gpu_core_configs = ['Fermi', 'Maxwell', 'Tegra']

def addGPUOptions(parser):
    parser.add_option("--gpgpusim-config", type="string", default=None, help="Path to the gpgpusim.config to use. This overrides the gpgpusim.config template")
    parser.add_option("--access-host-pagetable", action="store_true", default=False)
    parser.add_option("--split", default=False, action="store_true", help="Use split CPU and GPU cache hierarchies instead of fusion")
    parser.add_option("--kernel_stats", default=False, action="store_true", help="Dump statistics on GPU kernel boundaries")
    parser.add_option("--gpgpusim_stats", default=False, action="store_true", help="Dump statistics of GPGPU-Sim on GPU kernel boundaries")
    
   #gpu cores, note: these 3 configs will be loaded from the gpgpusim config file when specified
    parser.add_option("--clusters", default=16, help="Number of shader core clusters in the gpu that GPGPU-sim is simulating", type="int")
    parser.add_option("--cores_per_cluster", default=1, help="Number of shader cores per cluster in the gpu that GPGPU-sim is simulating", type="int")
    parser.add_option("--gpu-core-clock", default='700MHz', help="The frequency of GPU clusters (note: shaders operate at double this frequency when modeling Fermi)")
    parser.add_option("--ctas_per_shader", default=8, help="Number of simultaneous CTAs that can be scheduled to a single shader", type="int")
    parser.add_option("--gpu_warp_size", type="int", default=32, help="Number of threads per warp, also functional units per shader core/SM")
    parser.add_option("--gpu_threads_per_core", type="int", default=1536, help="Maximum number of threads per GPU core (SM)")
    #caches
    parser.add_option("--sc_l1_size", default="64kB", help="size of l1 cache hooked up to each sc")
    parser.add_option("--sc_l1_assoc", default=4, help="associativity of l1 cache hooked up to each sc", type="int")
    parser.add_option("--gpu_l1_buf_depth", type="int", default=96, help="Number of buffered L1 requests per shader")

    parser.add_option("--sc_tl1_size", default="64kB", help="size of l1 texture cache hooked up to each sc")
    parser.add_option("--sc_tl1_assoc", default=4, help="associativity of l1 texture cache hooked up to each sc", type="int")
    parser.add_option("--gpu_tl1_buf_depth", type="int", default=96, help="Number of buffered L1 requests per shader")

    parser.add_option("--gpu_l1_pagewalkers", type="int", default=32, help="Number of GPU L1 pagewalkers")

    parser.add_option("--sc_zl1_size", default="32kB", help="size of l1 z cache hooked up to each sc")
    parser.add_option("--sc_zl1_assoc", default=4, help="associativity of l1 z cache", type="int")
    parser.add_option("--gpu_zl1_buf_depth", type="int", default=96, help="Number of buffered Z-cache requests")

    parser.add_option("--gpu_tlb_entries", type="int", default=0, help="Number of entries in GPU Data TLB. 0 implies infinite")
    parser.add_option("--gpu_tlb_assoc", type="int", default=0, help="Associativity of the Data L1 TLB. 0 implies infinite")

    parser.add_option("--gpu_ttlb_entries", type="int", default=0, help="Number of entries in GPU Tex TLB. 0 implies infinite")
    parser.add_option("--gpu_ttlb_assoc", type="int", default=0, help="Associativity of the Tex L1 TLB. 0 implies infinite")

    parser.add_option("--gpu_num_l2caches", default=1, help="num of l2 GPU caches")
    parser.add_option("--sc_l2_size", default="1MB", help="size of L2 cache divided by num L2 caches")
    parser.add_option("--sc_l2_assoc", default=16, help="associativity of L2 cache backing SC L1's", type="int")
    parser.add_option("--gpu-l2-resource-stalls", action="store_true", default=False)
    
    parser.add_option("--pwc_size", default="8kB", help="Capacity of the page walk cache")
    parser.add_option("--pwc_assoc", default=16, help="Assoc of the page walk cache")
    parser.add_option("--pwc_policy", default= LRUReplacementPolicy(), help="Replacement policy of the page walk cache")
    parser.add_option("--flush_kernel_end", default=False, action="store_true", help="Flush the L1s at the end of each kernel. (Only VI_hammer)")
    #gpu memory
    parser.add_option("--gpu_core_config", type="choice", choices=gpu_core_configs, default='Fermi', help="configure the GPU cores like %s" % gpu_core_configs)
    parser.add_option("--gpu-mem-size", default='1GB', help="In split hierarchies, amount of GPU memory")
    parser.add_option("--gpu_mem_ctl_latency", type="int", default=-1, help="GPU memory controller latency in cycles")
    parser.add_option("--gpu_mem_freq", type="string", default=None, help="GPU memory controller frequency")
    parser.add_option("--gpu_membus_busy_cycles", type="int", default=-1, help="GPU memory bus busy cycles per data transfer")
    parser.add_option("--gpu_membank_busy_time", type="string", default=None, help="GPU memory bank busy time in ns (CL+tRP+tRCD+CAS)")
    #system memory
    parser.add_option("--total-mem-size", default='2GB', help="Total size of memory in system")
    parser.add_option("--dev-numa-high-bit", type="int", default=0, help="High order address bit to use for device NUMA mapping.")
    parser.add_option("--num-dev-dirs", default=1, help="In split hierarchies, number of device directories", type="int")
    #graphics options
    parser.add_option("--g_depth_shader", type = "int", default=0, help="depth test in shader")
    parser.add_option("--g_blend_shader", type = "int", default=1, help="Blend in shader")
    parser.add_option("--g_start_frame", type = "int", default=-1, help="Simulation start frame")
    parser.add_option("--g_end_frame", type = "int", default=-1, help="Simulation end frame")
    parser.add_option("--g_start_call", type = "int", default=0, help="Simulation start draw call")
    parser.add_option("--g_end_call", type = "int", default=-1, help="Simulation end draw call")
    parser.add_option("--g_raster_th", type = "int", default=32, help="Graphics raster tile height")
    parser.add_option("--g_raster_tw", type = "int", default=32, help="Graphics raster tile width")
    parser.add_option("--g_raster_bh", type = "int", default=128, help="Graphics raster block height")
    parser.add_option("--g_raster_bw", type = "int", default=128, help="Graphics raster block width")
    parser.add_option("--g_cp_start", type = "int", default=-1, help="Graphics checkpoint start frame")
    parser.add_option("--g_cp_end", type = "int", default=-1, help="Graphics checkpoint end frame")
    parser.add_option("--g_cp_period", type = "int", default=5, help="Graphics checkpoint period")
    parser.add_option("--g_skip_cp_frames", type = "int", default=0,  help="Graphics skip rendering checkpoint loading frames")
    parser.add_option("--ce_buffering", type="int", default=128, help="Maximum cache lines buffered in the GPU CE. 0 implies infinite")

def configureMemorySpaces(options):
    total_mem_range = AddrRange(options.total_mem_size)
    cpu_mem_range = total_mem_range
    gpu_mem_range = total_mem_range
    if options.split:
        buildEnv['PROTOCOL'] +=  '_split'
        total_mem_size = total_mem_range.size()
        gpu_mem_range = AddrRange(options.gpu_mem_size)
        if gpu_mem_range.size() >= total_mem_size:
            fatal("GPU memory size (%s) won't fit within total memory size (%s)!" % (options.gpu_mem_size, options.total_mem_size))
        gpu_segment_base_addr = Addr(total_mem_size - gpu_mem_range.size())
        gpu_mem_range = AddrRange(gpu_segment_base_addr, size = options.gpu_mem_size)
        options.total_mem_size = long(gpu_segment_base_addr)
        cpu_mem_range = AddrRange(options.total_mem_size)
    else:
        buildEnv['PROTOCOL'] +=  '_fusion'
    return (cpu_mem_range, gpu_mem_range, total_mem_range)

def parseGpgpusimConfig(options):
    # parse gpgpu config file
    # First check the cwd, and if there is not a gpgpusim.config file there
    # Use the template found in gem5-fusion/configs/gpu_config and fill in
    # the missing information with command line options.
    if options.gpgpusim_config:
        usingTemplate = False
        gpgpusimconfig = os.path.join(os.path.dirname(__file__),'gpu_config/'+options.gpgpusim_config)
        icntconfig = os.path.join(os.path.dirname(__file__),'gpu_config/'+options.icnt_config)
    else:
       usingTemplate = True
       if options.gpu_core_config == 'Fermi':
         gpgpusimconfig = os.path.join(os.path.dirname(__file__), 'gpu_config/gpgpusim.fermi.config.template')
       elif options.gpu_core_config == 'Maxwell':
         gpgpusimconfig = os.path.join(os.path.dirname(__file__), 'gpu_config/gpgpusim.maxwell.config.template')
       elif options.gpu_core_config == 'Tegra':
         gpgpusimconfig = os.path.join(os.path.dirname(__file__), 'gpu_config/gpgpusim.tegra.template')
       else:
         gpgpusimconfig = os.path.join(os.path.dirname(__file__), 'gpu_config/gpgpusim.config.template')

    if not os.path.isfile(gpgpusimconfig):
        fatal("Unable to find gpgpusim config (%s)" % gpgpusimconfig)
    f = open(gpgpusimconfig, 'r')
    config = f.read()
    f.close()

#    fDumpPath    = os.path.join(options.outdir, 'gpgpusimFrameDumps')
#    fShadersPath = os.path.join(options.outdir, 'gpgpusimShaders')
#    if not os.path.isdir(fDumpPath)
#       os.makedirs(fDumpPath)
#    if not os.path.isdir(fShadersPath)
#       os.makedirs.isdir(fShadersPath)

    config = config.replace("%outdir%", m5.options.outdir)
    config = config.replace("%gDepthShader%",   str(options.g_depth_shader) +"\n")
    config = config.replace("%gBlendShader%",   str(options.g_blend_shader) +"\n")
    config = config.replace("%mem_ctrls%",     str(options.mem_channels) +"\n")
    config = config.replace("%gStartFrame%",   str(options.g_start_frame) +"\n")
    config = config.replace("%gEndFrame%",     str(options.g_end_frame) +"\n")
    config = config.replace("%gStartCall%",    str(options.g_start_call) +"\n")
    config = config.replace("%gEndCall%",      str(options.g_end_call) +"\n")
    config = config.replace("%gRasterTH%",      str(options.g_raster_th) +"\n")
    config = config.replace("%gRasterTW%",      str(options.g_raster_tw) +"\n")
    config = config.replace("%gRasterBH%",      str(options.g_raster_bh) +"\n")
    config = config.replace("%gRasterBW%",      str(options.g_raster_bw) +"\n")
    config = config.replace("%gCpStart%",      str(options.g_cp_start) +"\n")
    config = config.replace("%gCpEnd%",        str(options.g_cp_end) +"\n")
    config = config.replace("%gCpPeriod%",     str(options.g_cp_period) +"\n")
    config = config.replace("%gSkipCpFrames%", str(options.g_skip_cp_frames) +"\n")

    if usingTemplate:
        print "Using template and command line options for gpgpusim.config"

        # Modify the GPGPU-Sim configuration template
        config = config.replace("%clusters%", str(options.clusters))
        config = config.replace("%cores_per_cluster%", str(options.cores_per_cluster))
        config = config.replace("%ctas_per_shader%", str(options.ctas_per_shader))
        icnt_outfile = os.path.join(m5.options.outdir, 'config_fermi_islip.icnt')
        config = config.replace("%icnt_file%", icnt_outfile)
        config = config.replace("%warp_size%", str(options.gpu_warp_size))
        # GPGPU-Sim config expects freq in MHz
        config = config.replace("%freq%", str(toFrequency(options.gpu_core_clock) / 1.0e6))
        config = config.replace("%threads_per_sm%", str(options.gpu_threads_per_core))
        options.num_sc = options.clusters*options.cores_per_cluster

        # Write out the configuration file to the output directory
        f = open(m5.options.outdir + '/gpgpusim.config', 'w')
        f.write(config)
        f.close()
        gpgpusimconfig = m5.options.outdir + '/gpgpusim.config'

        # Read in and modify the interconnect config template
        icnt_template = os.path.join(os.path.dirname(__file__), 'gpu_config/config_fermi_islip.template.icnt')
        f = open(icnt_template)
        icnt_config = f.read()
        f.close()

        # The number of nodes in the GPU network is the number of core clusters,
        # plus the number of GPU memory partitions, plus one extra (it is not
        # clear in GPGPU-Sim what this extra is for). Note: Aiming to remove
        # GPGPU-Sim interconnect completely as it only models parameter memory
        # handling currently (i.e. tiny fraction of accesses). Only model one
        # memory partition currently by default.
        num_icnt_nodes = str(options.clusters + 1 + 1)
        icnt_config = icnt_config.replace("%num_nodes%", num_icnt_nodes)

        # Write out the interconnect config file to the output directory
        f = open(icnt_outfile, 'w')
        f.write(icnt_config)
        f.close()
    else:
        print "Using gpgpusim.config for clusters, cores_per_cluster, Frequency, warp size"
        config = re.sub(re.compile("#.*?\n"), "", config)
        start = config.find("-gpgpu_n_clusters ") + len("-gpgpu_n_clusters ")
        end = config.find('-', start)
        gpgpu_n_clusters = int(config[start:end])
        start = config.find("-gpgpu_n_cores_per_cluster ") + len("-gpgpu_n_cores_per_cluster ")
        end = config.find('-', start)
        gpgpu_n_cores_per_cluster = int(config[start:end])
        num_sc = gpgpu_n_clusters * gpgpu_n_cores_per_cluster
        options.num_sc = num_sc
        start = config.find("-gpgpu_clock_domains ") + len("-gpgpu_clock_domains ")
        end = config.find(':', start)
        options.gpu_core_clock = config[start:end] + "MHz"
        start = config.find('-gpgpu_shader_core_pipeline ') + len('-gpgpu_shader_core_pipeline ')
        start = config.find(':', start) + 1
        end = config.find('\n', start)
        options.gpu_warp_size = int(config[start:end])
        icnt_outfile =   os.path.join(m5.options.outdir, 'config_network.icnt')
        gpgpusimconfig = os.path.join(m5.options.outdir, 'gpgpusim.config')
        config = config.replace("%icnt_file%", icnt_outfile)
        print icnt_outfile
        print gpgpusimconfig
        f = open(gpgpusimconfig, 'w')
        f.write(config)
        f.close()

        # Read in and modify the interconnect config template
        if(options.icnt_config):
           icnt_template = os.path.join(os.path.dirname(__file__), 'gpu_config/'+ options.icnt_config)
        else: 
           icnt_template = os.path.join(os.path.dirname(__file__), 'gpu_config/template_icnt.icnt')
        f = open(icnt_template)
        icnt_config = f.read()
        f.close()
        num_icnt_nodes = str(options.clusters + options.mem_channels)
        icnt_config = icnt_config.replace("%num_nodes%", num_icnt_nodes)
        f = open(icnt_outfile, 'w')
        f.write(icnt_config)
        f.close()

    if options.pwc_size == "0":
        # Bypass the shared L1 cache
        options.gpu_tlb_bypass_l1 = True
    else:
        # Do not bypass the page walk cache
        options.gpu_tlb_bypass_l1 = False

    # DEPRECATED: Get the GPU DRAM clock from the config file to be passed to
    # the DRAM component wrapper. This should be removed at a later date!
    config = re.sub(re.compile("#.*?\n"), "", config)
    start = config.find("-gpgpu_clock_domains ")
    end = config.find('\n', start)
    clk_domains = config[start:end].split(':')
    options.gpu_dram_clock = clk_domains[3] + "MHz"

    return gpgpusimconfig

def createGPU(options, gpu_mem_range):
    # DEPRECATED: Set a default GPU DRAM clock to be passed to the wrapper.
    # This must be eliminated when the wrapper can be removed.
    options.gpu_dram_clock = None

    gpgpusimOptions = parseGpgpusimConfig(options)

    # The GPU's clock domain is a source for all of the components within the
    # GPU. By making it a SrcClkDomain, it can be directly referenced to change
    # the GPU clock frequency dynamically.
    gpu = CudaGPU(warp_size = options.gpu_warp_size,
                  manage_gpu_memory = options.split,
                  clk_domain = SrcClockDomain(clock = options.gpu_core_clock,
                                              voltage_domain = VoltageDomain()),
                  gpu_memory_range = gpu_mem_range,
                  system_cacheline_size = options.cacheline_size)

    gpu.cores_wrapper = GPGPUSimComponentWrapper(clk_domain = gpu.clk_domain)

    gpu.icnt_wrapper = GPGPUSimComponentWrapper(clk_domain = DerivedClockDomain(
                                                    clk_domain = gpu.clk_domain,
                                                    clk_divider = 2))

    gpu.l2_wrapper = GPGPUSimComponentWrapper(clk_domain = gpu.clk_domain)
    gpu.dram_wrapper = GPGPUSimComponentWrapper(
                            clk_domain = SrcClockDomain(
                                clock = options.gpu_dram_clock,
                                voltage_domain = gpu.clk_domain.voltage_domain))

    warps_per_core = options.gpu_threads_per_core / options.gpu_warp_size
    gpu.shader_cores = [CudaCore(id = i, warp_contexts = warps_per_core)
                            for i in xrange(options.num_sc)]
    gpu.ce = GPUCopyEngine(driver_delay = 5000000,
                           buffering = options.ce_buffering)
    gpu.zunit = ZUnit()

    for sc in gpu.shader_cores:
        sc.lsq = ShaderLSQ()
        sc.tex_lq = ShaderLSQ()

        sc.lsq.data_tlb.entries = options.gpu_tlb_entries
        sc.tex_lq.data_tlb.entries = options.gpu_ttlb_entries

        sc.lsq.forward_flush = (buildEnv['PROTOCOL'] == 'VI_hammer_fusion' and options.flush_kernel_end)
        sc.tex_lq.forward_flush = (buildEnv['PROTOCOL'] == 'VI_hammer_fusion' and options.flush_kernel_end)

        sc.lsq.warp_size = options.gpu_warp_size
        sc.tex_lq.warp_size = options.gpu_warp_size

        sc.lsq.cache_line_size = options.cacheline_size
        sc.tex_lq.cache_line_size = options.cacheline_size
#sc.lsq.request_buffer_depth = options.gpu_l1_buf_depth
#sc.tex_lq.request_buffer_depth = options.gpu_tl1_buf_depth
        if options.gpu_threads_per_core % options.gpu_warp_size:
            fatal("gpu_warp_size must divide gpu_threads_per_core evenly.")
        sc.lsq.warp_contexts = warps_per_core
        sc.tex_lq.warp_contexts = warps_per_core
        if options.gpu_core_config == 'Fermi':
            # Fermi latency for zero-load independent memory instructions is
            # roughly 19 total cycles with ~4 cycles for tag access
            sc.lsq.l1_tag_cycles = 4
            sc.lsq.latency = 14
        elif options.gpu_core_config == 'Maxwell':
            # Maxwell latency for zero-load independent memory instructions is
            # 8-10 cycles quicker than Fermi, and tag access appears shorter
            sc.lsq.l1_tag_cycles = 1
            sc.lsq.latency = 6
        elif options.gpu_core_config == 'Tegra':
            #for now copy fermi configs
            #FIXME
            sc.lsq.l1_tag_cycles = 1
            sc.lsq.latency = 6

    # This is a stop-gap solution until we implement a better way to register device memory
    if options.access_host_pagetable:
        gpu.access_host_pagetable = True
        for sc in gpu.shader_cores:
            sc.itb.access_host_pagetable = True
            sc.ttb.access_host_pagetable = True
            sc.lsq.data_tlb.access_host_pagetable = True
            sc.tex_lq.data_tlb.access_host_pagetable = True
        gpu.ce.device_dtb.access_host_pagetable = True
        gpu.ce.host_dtb.access_host_pagetable = True
        gpu.zunit.ztb.access_host_pagetable = True

    gpu.config_path = gpgpusimOptions
    gpu.dump_kernel_stats = options.kernel_stats
    gpu.dump_gpgpusim_stats = options.gpgpusim_stats

    return gpu

def connectGPUPorts(system, gpu, ruby, options):

    # for now only VI_fusion has tex and z caches added
    mp = 1
    if(buildEnv['PROTOCOL'].lower().count("vi") and (not options.split)):
      mp = 2
      idx = options.num_cpus+len(gpu.shader_cores)*mp+2
      print "connecting zunit to ", idx
      gpu.zunit.z_port = ruby._cpu_ports[idx].slave
    else:
      #if not VI assert the g_depth_shader option is used
      assert(options.g_depth_shader==1), "No z-cache, g_depth_shader has to be enabled"

    for i,sc in enumerate(gpu.shader_cores):
        sc.inst_port = ruby._cpu_ports[options.num_cpus+i*mp].slave
        sc.tex_port  = ruby._cpu_ports[options.num_cpus+i*mp].slave

        for j in xrange(options.gpu_warp_size):
            sc.lsq_port[j] = sc.lsq.lane_port[j]
            sc.tex_lq_port[j] = sc.tex_lq.lane_port[j]
        sc.lsq.cache_port = ruby._cpu_ports[options.num_cpus+i*mp].slave
        sc.tex_lq.cache_port = ruby._cpu_ports[options.num_cpus+(i*mp)+mp-1].slave
        sc.lsq_ctrl_port = sc.lsq.control_port
        sc.tex_ctrl_port = sc.tex_lq.control_port

    # The total number of sequencers is equal to the number of CPU cores * 2, plus
    # the number of GPU cores plus any pagewalk caches, copy engine and z
    # caches. Currently, for unified address space architectures, there is one
    # pagewalk cache, one copy engine cache and one z-cache (3 total), and the pagewalk cache
    # is indexed first, then the CE then the Z. 
    #For split address space architectures, there are 2 copy
    # engine caches, and the host-side cache is indexed before the device-side.
    try:
      datapathsCount = len(system.datapaths)
    except:
      datapathsCount = 0
    assert(len(ruby._cpu_ports) == options.num_cpus + options.num_sc*mp + mp +1 + datapathsCount)

    # Initialize the MMU, connecting it to either the pagewalk cache port for
    # unified address space, or the copy engine's host-side sequencer port for
    # split address space architectures.
    gpu.shader_mmu.setUpPagewalkers(options.gpu_l1_pagewalkers,
                    ruby._cpu_ports[options.num_cpus+options.num_sc*mp].slave,
                    options.gpu_tlb_bypass_l1)

    if options.split:
        # NOTE: In split address space architectures, the MMU only provides the
        # copy engine host-side TLB access to a page walker. This should
        # probably be changed so that the copy engine doesn't manage
        # translations, but only the data handling

        # If inappropriately used, crash to inform MMU config problems to user:
        if options.access_host_pagetable:
            fatal('Cannot access host pagetable from the GPU or the copy ' \
                  'engine\'s GPU-side port\n in split address space. Use ' \
                  'only one of --split or --access-host-pagetable')

        # Tie copy engine ports to appropriate sequencers
        gpu.ce.host_port = \
            ruby._cpu_ports[options.num_cpus+options.num_sc*mp].slave
        gpu.ce.device_port = \
            ruby._cpu_ports[options.num_cpus+options.num_sc*mp+1].slave
        gpu.ce.device_dtb.access_host_pagetable = False
    else:
        # With a unified address space, tie both copy engine ports to the same
        # copy engine controller. NOTE: The copy engine is often unused in the
        # unified address space
        gpu.ce.host_port = \
            ruby._cpu_ports[options.num_cpus+options.num_sc*mp+1].slave
        gpu.ce.device_port = \
            ruby._cpu_ports[options.num_cpus+options.num_sc*mp+1].slave
