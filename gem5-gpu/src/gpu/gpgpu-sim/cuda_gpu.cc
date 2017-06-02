/*
 * Copyright (c) 2011 Mark D. Hill and David A. Wood
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "api/gpu_syscall_helper.hh"
#include "arch/x86/regs/misc.hh"
#include "arch/utility.hh"
#include "arch/vtophys.hh"
#include "base/chunk_generator.hh"
#include "base/statistics.hh"
#include "cpu/thread_context.hh"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx-stats.h"
#include "debug/CudaGPU.hh"
#include "debug/CudaGPUAccess.hh"
#include "debug/CudaGPUPageTable.hh"
#include "debug/CudaGPUTick.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "mem/ruby/system/System.hh"
#include "params/GPGPUSimComponentWrapper.hh"
#include "params/CudaGPU.hh"
#include "sim/full_system.hh"
#include "gpgpusim_entrypoint.h"
#include "graphics/mesa_gpgpusim.h"

using namespace std;

extern unsigned g_active_device;
extern renderData_t g_renderData;
vector<CudaGPU*> CudaGPU::gpuArray;

const size_t CudaGPU::GMemory::XLGBLOCK_SIZE;
const size_t CudaGPU::GMemory::XLGBLOCK_COUNT;
const size_t CudaGPU::GMemory::LGBLOCK_SIZE;
const size_t CudaGPU::GMemory::SGBLOCK_SIZE;

// From GPU syscalls
void registerFatBinaryTop(GPUSyscallHelper *helper, Addr sim_fatCubin, size_t sim_binSize);
unsigned int registerFatBinaryBottom(GPUSyscallHelper *helper, Addr sim_alloc_ptr);
void register_var(Addr sim_deviceAddress, const char* deviceName, int sim_size, int sim_constant, int sim_global, int sim_ext, Addr sim_hostVar);

CudaGPU::CudaGPU(const Params *p) :
    ClockedObject(p), _params(p), streamTickEvent(this),
    clkDomain((SrcClockDomain*)p->clk_domain),
    coresWrapper(*p->cores_wrapper), icntWrapper(*p->icnt_wrapper),
    l2Wrapper(*p->l2_wrapper), dramWrapper(*p->dram_wrapper),
    system(p->sys), warpSize(p->warp_size), 
    gpgpusimConfigPath(p->config_path), unblockNeeded(false), ruby(p->ruby),
    system_cacheline_size(p->system_cacheline_size),
    runningTC(NULL), runningStream(NULL), runningTID(-1), clearTick(0),
    dumpKernelStats(p->dump_kernel_stats),
    dumpGpgpusimStats(p->dump_gpgpusim_stats), pageTable(),
    manageGPUMemory(p->manage_gpu_memory),
    accessHostPageTable(p->access_host_pagetable),
    gpuMemoryRange(p->gpu_memory_range), shaderMMU(p->shader_mmu),
    _currentBlockedStream(NULL)
{
    // Register this device as a CUDA-enabled GPU
    cudaDeviceID = registerCudaDevice(this);
    if (cudaDeviceID >= 1) {
        // TODO: Remove this when multiple GPUs can exist in system
        panic("GPGPU-Sim is not currently able to simulate more than 1 CUDA-enabled GPU\n");
    }

    streamDelay = 1;

    running = false;

    restoring = false;

    if(p->kernel_delay_enabled){
       launchDelay = p->kernel_launch_delay * SimClock::Frequency;
       returnDelay = p->kernel_return_delay * SimClock::Frequency;
    } else {
       launchDelay = returnDelay = 0;
    }

    // GPU memory handling
    instBaseVaddr = 0;
    localBaseVaddr = 0;
    // Reserve the 0 virtual page for NULL pointers
    virtualGPUBrkAddr = TheISA::PageBytes;
    physicalGPUBrkAddr = gpuMemoryRange.start();

    // Initialize GPGPU-Sim
    theGPU = gem5_ptx_sim_init_perf(&streamManager, this, getConfigPath());
    theGPU->init();

    // Set up the component wrappers in order to cycle the GPGPU-Sim
    // shader cores, interconnect, L2 cache and DRAM
    // TODO: Eventually, we want to remove the need for the GPGPU-Sim L2 cache
    // and DRAM. Currently, these are necessary to handle parameter memory
    // accesses.
    coresWrapper.setGPU(theGPU);
    coresWrapper.setStartCycleFunction(&gpgpu_sim::core_cycle_start);
    coresWrapper.setEndCycleFunction(&gpgpu_sim::core_cycle_end);
    icntWrapper.setGPU(theGPU);
    icntWrapper.setStartCycleFunction(&gpgpu_sim::icnt_cycle_start);
    icntWrapper.setEndCycleFunction(&gpgpu_sim::icnt_cycle_end);
    l2Wrapper.setGPU(theGPU);
    l2Wrapper.setStartCycleFunction(&gpgpu_sim::l2_cycle);
    dramWrapper.setGPU(theGPU);
    dramWrapper.setStartCycleFunction(&gpgpu_sim::dram_cycle);

    // Setup the device properties for this GPU
    snprintf(deviceProperties.name, 256, "GPGPU-Sim_v%s", g_gpgpusim_version_string);
    deviceProperties.major = 2;
    deviceProperties.minor = 0;
    deviceProperties.totalGlobalMem = gpuMemoryRange.size();
    deviceProperties.memPitch = 0;
    deviceProperties.maxThreadsPerBlock = 1024;
    deviceProperties.maxThreadsDim[0] = 1024;
    deviceProperties.maxThreadsDim[1] = 1024;
    deviceProperties.maxThreadsDim[2] = 64;
    deviceProperties.maxGridSize[0] = 0x40000000;
    deviceProperties.maxGridSize[1] = 0x40000000;
    deviceProperties.maxGridSize[2] = 0x40000000;
    deviceProperties.totalConstMem = gpuMemoryRange.size();
    deviceProperties.textureAlignment = 0;
    deviceProperties.multiProcessorCount = cudaCores.size();
    deviceProperties.sharedMemPerBlock = theGPU->shared_mem_size();
    deviceProperties.regsPerBlock = theGPU->num_registers_per_core();
    deviceProperties.warpSize = theGPU->wrp_size();
    deviceProperties.clockRate = theGPU->shader_clock();
#if (CUDART_VERSION >= 2010)
    deviceProperties.multiProcessorCount = theGPU->get_config().num_shader();
#endif

    // Print gpu configuration and stats at exit
    GPUExitCallback* gpuExitCB = new GPUExitCallback(this, p->stats_filename);
    registerExitCallback(gpuExitCB);

}

void CudaGPU::serialize(CheckpointOut &cp) const
{
    DPRINTF(CudaGPU, "Serializing\n");
    if (running) {
        panic("Checkpointing during GPU execution not supported\n");
    }

    SERIALIZE_SCALAR(m_last_fat_cubin_handle);
    SERIALIZE_SCALAR(instBaseVaddr);
    SERIALIZE_SCALAR(localBaseVaddr);

    SERIALIZE_SCALAR(runningTID);

    int numBinaries = fatBinaries.size();
    SERIALIZE_SCALAR(numBinaries);
    for (int i=0; i<numBinaries; i++) {
        stringstream ss;
        ss << i;
        string num = ss.str();
        paramOut(cp, num+"fatBinaries.tid", fatBinaries[i].tid);
        paramOut(cp, num+"fatBinaries.handle", fatBinaries[i].handle);
        paramOut(cp, num+"fatBinaries.sim_fatCubin", fatBinaries[i].sim_fatCubin);
        paramOut(cp, num+"fatBinaries.sim_binSize", fatBinaries[i].sim_binSize);
        paramOut(cp, num+"fatBinaries.sim_alloc_ptr", fatBinaries[i].sim_alloc_ptr);

        paramOut(cp, num+"fatBinaries.funcMap.size", fatBinaries[i].funcMap.size());
        std::map<const void*,string>::const_iterator it = fatBinaries[i].funcMap.begin();
        int j = 0;
        for (; it != fatBinaries[i].funcMap.end(); it++) {
            paramOut(cp, csprintf("%dfatBinaries.funcMap[%d].first", i, j), (uint64_t)it->first);
            paramOut(cp, csprintf("%dfatBinaries.funcMap[%d].second", i, j), it->second);
            j++;
        }
    }

    int numVars = cudaVars.size();
    SERIALIZE_SCALAR(numVars);
    for (int i=0; i<numVars; i++) {
        _CudaVar var = cudaVars[i];
        paramOut(cp, csprintf("cudaVars[%d].sim_deviceAddress", i), var.sim_deviceAddress);
        paramOut(cp, csprintf("cudaVars[%d].deviceName", i), var.deviceName);
        paramOut(cp, csprintf("cudaVars[%d].sim_size", i), var.sim_size);
        paramOut(cp, csprintf("cudaVars[%d].sim_constant", i), var.sim_constant);
        paramOut(cp, csprintf("cudaVars[%d].sim_global", i), var.sim_global);
        paramOut(cp, csprintf("cudaVars[%d].sim_ext", i), var.sim_ext);
        paramOut(cp, csprintf("cudaVars[%d].sim_hostVar", i), var.sim_hostVar);
    }

    pageTable.serialize(cp);
}

void CudaGPU::unserialize(CheckpointIn &cp)
{
    DPRINTF(CudaGPU, "UNserializing\n");

    restoring = true;

    UNSERIALIZE_SCALAR(m_last_fat_cubin_handle);
    UNSERIALIZE_SCALAR(instBaseVaddr);
    UNSERIALIZE_SCALAR(localBaseVaddr);

    UNSERIALIZE_SCALAR(runningTID);

    DPRINTF(CudaGPU, "UNSerializing %d, %d\n", m_last_fat_cubin_handle, instBaseVaddr);

    int numBinaries;
    UNSERIALIZE_SCALAR(numBinaries);
    DPRINTF(CudaGPU, "UNserializing %d binaries\n", numBinaries);
    fatBinaries.resize(numBinaries);
    for (int i=0; i<numBinaries; i++) {
        stringstream ss;
        ss << i;
        string num = ss.str();
        paramIn(cp, num+"fatBinaries.tid", fatBinaries[i].tid);
        paramIn(cp, num+"fatBinaries.handle", fatBinaries[i].handle);
        paramIn(cp, num+"fatBinaries.sim_fatCubin", fatBinaries[i].sim_fatCubin);
        paramIn(cp, num+"fatBinaries.sim_binSize", fatBinaries[i].sim_binSize);
        paramIn(cp, num+"fatBinaries.sim_alloc_ptr", fatBinaries[i].sim_alloc_ptr);
        DPRINTF(CudaGPU, "Got %d %d %d %d\n", fatBinaries[i].handle, fatBinaries[i].sim_fatCubin, fatBinaries[i].sim_binSize, fatBinaries[i].sim_alloc_ptr);

        int funcMapSize;
        paramIn(cp, num+"fatBinaries.funcMap.size", funcMapSize);
        for (int j=0; j<funcMapSize; j++) {
            uint64_t first;
            string second;
            paramIn(cp, csprintf("%dfatBinaries.funcMap[%d].first", i, j), first);
            paramIn(cp, csprintf("%dfatBinaries.funcMap[%d].second", i, j), second);
            fatBinaries[i].funcMap[(const void*)first] = second;
        }
    }

    int numVars;
    UNSERIALIZE_SCALAR(numVars);
    cudaVars.resize(numVars);
    for (int i=0; i<numVars; i++) {
        paramIn(cp, csprintf("cudaVars[%d].sim_deviceAddress", i), cudaVars[i].sim_deviceAddress);
        paramIn(cp, csprintf("cudaVars[%d].deviceName", i), cudaVars[i].deviceName);
        paramIn(cp, csprintf("cudaVars[%d].sim_size", i), cudaVars[i].sim_size);
        paramIn(cp, csprintf("cudaVars[%d].sim_constant", i), cudaVars[i].sim_constant);
        paramIn(cp, csprintf("cudaVars[%d].sim_global", i), cudaVars[i].sim_global);
        paramIn(cp, csprintf("cudaVars[%d].sim_ext", i), cudaVars[i].sim_ext);
        paramIn(cp, csprintf("cudaVars[%d].sim_hostVar", i), cudaVars[i].sim_hostVar);
    }

    pageTable.unserialize(cp);
}

void CudaGPU::startup()
{
    // Initialize CUDA cores
    vector<CudaCore*>::iterator iter;
    for (iter = cudaCores.begin(); iter != cudaCores.end(); ++iter) {
        (*iter)->initialize();
    }

    if (!restoring) {
        return;
    }

    if (runningTID >= 0) {
        runningTC = system->getThreadContext(runningTID);
        assert(runningTC);
    }

    // Setting everything up again!
    std::vector<_FatBinary>::iterator binaries;
    for (binaries = fatBinaries.begin(); binaries != fatBinaries.end(); ++binaries) {
        _FatBinary bin = *binaries;
        GPUSyscallHelper helper(system->getThreadContext(bin.tid));
        assert(helper.getThreadContext());
        registerFatBinaryTop(&helper, bin.sim_fatCubin, bin.sim_binSize);
        registerFatBinaryBottom(&helper, bin.sim_alloc_ptr);

        std::map<const void*, string>::iterator functions;
        for (functions = bin.funcMap.begin(); functions != bin.funcMap.end(); ++functions) {
            const char *host_fun = (const char*)functions->first;
            const char *device_fun = functions->second.c_str();
            register_function(bin.handle, host_fun, device_fun);
        }
    }

    std::vector<_CudaVar>::iterator variables;
    for (variables = cudaVars.begin(); variables != cudaVars.end(); ++variables) {
        _CudaVar var = *variables;
        register_var(var.sim_deviceAddress, var.deviceName.c_str(), var.sim_size, var.sim_constant, var.sim_global, var.sim_ext, var.sim_hostVar);
    }
}

void CudaGPU::clearStats()
{
    ruby->resetStats();
    clearTick = curTick();
}

void CudaGPU::registerCudaCore(CudaCore *sc)
{
    cudaCores.push_back(sc);

    // Update the multiprocessor count
    deviceProperties.multiProcessorCount = cudaCores.size();
}

void CudaGPU::registerZUnit(ZUnit * zu){
   zunit = zu;
}

void CudaGPU::registerCopyEngine(GPUCopyEngine *ce)
{
    copyEngine = ce;
}

void CudaGPU::streamTick() {
    DPRINTF(CudaGPUTick, "Stream Tick\n");

    // launch operation on device if one is pending and can be run
    stream_operation op = streamManager->front();
    op.do_operation(theGPU);

    //op.print(stdout);

    if (op.is_done() and streamManager->ready()) {
        schedule(streamTickEvent, curTick() + streamDelay);
    }
}

void CudaGPU::scheduleStreamEvent() {
    if (streamTickEvent.scheduled()) {
        DPRINTF(CudaGPUTick, "Already scheduled a tick, ignoring\n");
        return;
    }

    schedule(streamTickEvent, nextCycle());
}

// FIXME: So, this code doesn't allow another kernel to be scheduled once a previous
// one already has been scheduled. If already scheduled, each wrapper will be scheduled on its
// own. So, if we just don't schedule anything, the kernel should run anyway. However, we lose
// the timing with the launchDelay. If we want to be more accurate, I think the thing to do would
// be to delay the launch manually and setup the tick events to match whatever the next GPU clock 
// cycle should be as well as when the kernel should be scheduled.
//
// In the meantime, just check if it's already scheduled, if so, do nothing and let it go on its own
void CudaGPU::beginRunning(Tick stream_queued_time, struct CUstream_st *_stream)
{
    beginStreamOperation(_stream);

    DPRINTF(CudaGPU, "Beginning kernel execution at %llu\n", curTick());
    kernelTimes.push_back(curTick());
    if (dumpKernelStats) {
        Stats::dump();
        Stats::reset();
    }

    if(dumpGpgpusimStats){
       theGPU->print_stats();
       theGPU->update_stats();
    }
    numKernelsStarted++;
        
    Tick delay = clockPeriod();
    if ((stream_queued_time + launchDelay) > curTick()) {
        // Delay launch to the end of the launch delay
        delay = (stream_queued_time + launchDelay) - curTick();
    }

    if( running ) {
        //panic("Should not already be running if we are starting\n");
        DPRINTF(CudaGPU, "Something is already running on the GPU! Let's do some more work!\n");
        if(!coresWrapper.isScheduled()) coresWrapper.scheduleEvent(delay);
        if(!icntWrapper.isScheduled()) icntWrapper.scheduleEvent(delay);
        if(!l2Wrapper.isScheduled()) l2Wrapper.scheduleEvent(delay);
        if(!dramWrapper.isScheduled()) dramWrapper.scheduleEvent(delay);
    } else {
        running = true;
        coresWrapper.scheduleEvent(delay);
        icntWrapper.scheduleEvent(delay);
        l2Wrapper.scheduleEvent(delay);
        dramWrapper.scheduleEvent(delay);
    }

#if 0
        if( !coresWrapper.isScheduled() ) coresWrapper.scheduleEvent(delay); 
        if( !icntWrapper.isScheduled() ) icntWrapper.scheduleEvent(delay);
        if( !l2Wrapper.isScheduled() ) l2Wrapper.scheduleEvent(delay);
        if( !dramWrapper.isScheduled() ) dramWrapper.scheduleEvent(delay);
#endif

#if 0    
    if (running) {
        //panic("Should not already be running if we are starting\n");
        DPRINTF(CudaGPU, "Something is already running on the GPU! Let's do some more work!\n");
    }
    running = true;

    Tick delay = clockPeriod();
    if ((stream_queued_time + launchDelay) > curTick()) {
        // Delay launch to the end of the launch delay
        delay = (stream_queued_time + launchDelay) - curTick();
    }

    coresWrapper.scheduleEvent(delay);
    icntWrapper.scheduleEvent(delay);
    l2Wrapper.scheduleEvent(delay);
    dramWrapper.scheduleEvent(delay);
#endif
}

void CudaGPU::finishKernel(int grid_id)
{
    numKernelsCompleted++;
    FinishKernelEvent *e = new FinishKernelEvent(this, grid_id);
    schedule(e, curTick() + returnDelay);
}

void CudaGPU::processFinishKernelEvent(int grid_id)
{
    DPRINTF(CudaGPU, "GPU finished a kernel id %d\n", grid_id);

    CUstream_st* stream = streamManager->get_kernel_stream(grid_id);
    streamManager->register_finished_kernel(grid_id);

    kernelTimes.push_back(curTick());
    if (dumpKernelStats) {
        Stats::dump();
        Stats::reset();
    }

    if (unblockNeeded && streamManager->empty()) {
        DPRINTF(CudaGPU, "Stream manager is empty, unblocking\n");
        unblockThread(runningTC);
    }

    scheduleStreamEvent();

    endStreamOperation(stream);

    g_renderData.checkEndOfShader(this);
}

CudaCore *CudaGPU::getCudaCore(int coreId)
{
    assert(coreId < cudaCores.size());
    return cudaCores[coreId];
}

CudaGPU *CudaGPUParams::create() {
    return new CudaGPU(this);
}

void CudaGPU::gpuPrintStats(std::ostream& out) {
    if(dumpGpgpusimStats){
       theGPU->print_stats();
       theGPU->update_stats();
    }
    // Print kernel statistics
    Tick total_kernel_ticks = 0;
    Tick last_kernel_time = 0;
    bool kernel_active = false;
    vector<Tick>::iterator it;

    out << "spa frequency: " << frequency()/1000000000.0 << " GHz\n";
    out << "spa period: " << clockPeriod() << " ticks\n";
    out << "kernel times (ticks):\n";
    out << "start, end, start, end, ..., exit\n";
    for (it = kernelTimes.begin(); it < kernelTimes.end(); it++) {
        out << *it << ", ";
        if (kernel_active) {
            total_kernel_ticks += (*it - last_kernel_time);
            kernel_active = false;
        } else {
            last_kernel_time = *it;
            kernel_active = true;
        }
    }
    out << curTick() << "\n";

    // Print Shader CTA statistics
    out << "\nshader CTA times (ticks):\n";
    out << "shader, CTA ID, start, end, start, end, ..., exit\n";
    std::vector<CudaCore*>::iterator cores;
    for (cores = cudaCores.begin(); cores != cudaCores.end(); cores++) {
        (*cores)->printCTAStats(out);
    }
    out << "\ntotal kernel time (ticks) = " << total_kernel_ticks << "\n";

    if (clearTick) {
        out << "Stats cleared at tick " << clearTick << "\n";
    }
}

extern char *ptx_line_stats_filename;

void CudaGPU::printPTXFileLineStats() {
    char *temp_ptx_line_stats_filename = ptx_line_stats_filename;
    std::string outfile = simout.directory() + ptx_line_stats_filename;
    ptx_line_stats_filename = (char*)outfile.c_str();
    ptx_file_line_stats_write_file();
    ptx_line_stats_filename = temp_ptx_line_stats_filename;
}

bool CudaGPU::memcpy(void *src, void *dst, size_t count, struct CUstream_st *_stream, stream_operation_type type) {
    if( copyEngine->Ready() ) {
        beginStreamOperation(_stream);
        copyEngine->setCurrentStream(_stream);
        copyEngine->memcpy((Addr)src, (Addr)dst, count, type);
        return true;
    }
    return false;
}

bool CudaGPU::memcpy_to_symbol(const char *hostVar, const void *src, size_t count, size_t offset, struct CUstream_st *_stream) {
    if( copyEngine->Ready() ) {
        // First, initialize the stream operation
        beginStreamOperation(_stream);

        // Lookup destination address for transfer:
        std::string sym_name = gpgpu_ptx_sim_hostvar_to_sym_name(hostVar);
        std::map<std::string,symbol_table*>::iterator st = g_sym_name_to_symbol_table.find(sym_name.c_str());
        assert(st != g_sym_name_to_symbol_table.end());
        symbol_table *symtab = st->second;

        symbol *sym = symtab->lookup(sym_name.c_str());
        assert(sym);
        unsigned dst = sym->get_address() + offset;
        printf("GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %zu bytes to symbol %s+%zu @0x%x ...\n",
               count, sym_name.c_str(), offset, dst);

        copyEngine->setCurrentStream(_stream);
        copyEngine->memcpy((Addr)src, (Addr)dst, count, stream_memcpy_host_to_device);
        return true;
    }
    return false;
}

bool CudaGPU::memcpy_from_symbol(void *dst, const char *hostVar, size_t count, size_t offset, struct CUstream_st *_stream) {
    if( copyEngine->Ready() ) {
        // First, initialize the stream operation
        beginStreamOperation(_stream);

        // Lookup destination address for transfer:
        std::string sym_name = gpgpu_ptx_sim_hostvar_to_sym_name(hostVar);
        std::map<std::string,symbol_table*>::iterator st = g_sym_name_to_symbol_table.find(sym_name.c_str());
        assert(st != g_sym_name_to_symbol_table.end());
        symbol_table *symtab = st->second;

        symbol *sym = symtab->lookup(sym_name.c_str());
        assert(sym);
        unsigned src = sym->get_address() + offset;
        printf("GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %zu bytes from symbol %s+%zu @0x%x ...\n",
               count, sym_name.c_str(), offset, src);

        copyEngine->setCurrentStream(_stream);
        copyEngine->memcpy((Addr)src, (Addr)dst, count, stream_memcpy_device_to_host);
        return true;
    }
    return false;
}

bool CudaGPU::memset(Addr dst, int value, size_t count, struct CUstream_st *_stream) {
    if( copyEngine->Ready() ) {
        beginStreamOperation(_stream);
        copyEngine->setCurrentStream(_stream);
        copyEngine->memset(dst, value, count);
        return true;
    } 
    return false;
}

Addr CudaGPU::getSymbolSimAddr(const char *hostVar, size_t offset){
    std::string sym_name = gpgpu_ptx_sim_hostvar_to_sym_name(hostVar);
    std::map<std::string,symbol_table*>::iterator st = g_sym_name_to_symbol_table.find(sym_name.c_str());
    assert( st != g_sym_name_to_symbol_table.end() );
    symbol_table *symtab = st->second;

    symbol *sym = symtab->lookup(sym_name.c_str());
    assert(sym);
    Addr dst = sym->get_address() + offset;
    return dst;
}

//FIXME:
void CudaGPU::finishCopyOperation()
{
    runningStream->record_next_done();
    scheduleStreamEvent();
    unblockThread(runningTC);
    //endStreamOperation();
}

void CudaGPU::finishStreamCopyOperation(CUstream_st* stream)
{
    stream->record_next_done();
    scheduleStreamEvent();
    unblockThread(runningTC);
    endStreamOperation(stream);
}

// TODO: When we move the stream manager into libcuda, this will need to be
// eliminated, and libcuda will have to decide when to block the calling thread
bool CudaGPU::needsToBlock(CUstream_st* stream)
{
    if( !streamManager->streamEmpty(stream) ) {
        DPRINTF(CudaGPU, "Suspend request for stream %x: Need to activate CPU later\n", stream);
        unblockNeeded = true;
        streamManager->print(stdout);
        return true;
    } else {
        DPRINTF(CudaGPU, "Suspend request: Already done.\n");
        return false;
    }
}

void CudaGPU::blockThread(ThreadContext *tc, Addr signal_ptr, CUstream_st* stream)
{
    if( streamManager->streamEmpty(stream) ) { // NULL stream looks at stream zero
        // It is common in small memcpys for the stream operation to be complete
        // by the time cudaMemcpy calls blockThread. In this case, just signal
        DPRINTF(CudaGPU, "No stream operations to block thread %p. Continuing...\n", tc);
        signalThread(tc, signal_ptr);
        blockedThreads.erase(tc);
        unblockNeeded = false;
    } else {
        if (!shaderMMU->isFaultInFlight(tc)) {
            DPRINTF(CudaGPU, "Blocking thread %p for GPU syscall\n", tc);
            blockedThreads[tc] = signal_ptr;
            tc->suspend();
            _currentBlockedStream = stream; // Register the stream that we're blocking on
        } else {
            DPRINTF(CudaGPU, "Pending GPU fault must be handled: Not blocking thread\n");
        }
    }
}

void CudaGPU::signalThread(ThreadContext *tc, Addr signal_ptr)
{
    GPUSyscallHelper helper(tc);
    bool signal_val = true;

    // Read signal value and ensure that it is currently false
    // (i.e. thread should be currently blocked)
    helper.readBlob(signal_ptr, (uint8_t*)&signal_val, sizeof(bool));
    if (signal_val) {
        panic("Thread doesn't appear to be blocked!\n");
    }

    signal_val = true;
    helper.writeBlob(signal_ptr, (uint8_t*)&signal_val, sizeof(bool));
}

void CudaGPU::unblockThread(ThreadContext *tc)
{
    if (!tc) tc = runningTC;
    if (tc->status() != ThreadContext::Suspended) return;
    assert(unblockNeeded);
    if( !streamManager->streamEmpty(_currentBlockedStream) ) {
        // There must be more in the queue of work to complete. Need to
        // continue blocking
        if( _currentBlockedStream ) 
            DPRINTF(CudaGPU, "Still something in stream %x, continuing block\n", _currentBlockedStream);
        else
            DPRINTF(CudaGPU, "Still something in stream zero, continuing block\n");
        return;
    }

    DPRINTF(CudaGPU, "Unblocking thread %p for GPU syscall\n", tc);
    std::map<ThreadContext*, Addr>::iterator tc_iter = blockedThreads.find(tc);
    if (tc_iter == blockedThreads.end()) {
        panic("Cannot find blocked thread!\n");
    }

    Addr signal_ptr = blockedThreads[tc];
    signalThread(tc, signal_ptr);

    blockedThreads.erase(tc);
    unblockNeeded = false;
    _currentBlockedStream = NULL;
    tc->activate();
}

void CudaGPU::add_binary( symbol_table *symtab, unsigned fat_cubin_handle )
{
    m_code[fat_cubin_handle] = symtab;
    m_last_fat_cubin_handle = fat_cubin_handle;
}

void CudaGPU::add_ptxinfo( const char *deviceFun, const struct gpgpu_ptx_sim_kernel_info info )
{
    symbol *s = m_code[m_last_fat_cubin_handle]->lookup(deviceFun);
    assert( s != NULL );
    function_info *f = s->get_pc();
    assert( f != NULL );
    f->set_kernel_info(info);
}

void CudaGPU::register_function( unsigned fat_cubin_handle, const char *hostFun, const char *deviceFun )
{
    if( m_code.find(fat_cubin_handle) != m_code.end() ) {
        symbol *s = m_code[fat_cubin_handle]->lookup(deviceFun);
        assert( s != NULL );
        function_info *f = s->get_pc();
        assert( f != NULL );
        m_kernel_lookup[hostFun] = f;
    } else {
        m_kernel_lookup[hostFun] = NULL;
    }
}

function_info *CudaGPU::get_kernel(const char *hostFun)
{
    std::map<const void*,function_info*>::iterator i = m_kernel_lookup.find(hostFun);
    if (i != m_kernel_lookup.end()) {
        return i->second;
    }
    return NULL;
}

void CudaGPU::setInstBaseVaddr(uint64_t addr)
{
   instBaseVaddr = addr;
}

uint64_t CudaGPU::getInstBaseVaddr()
{
    return instBaseVaddr;
}

void CudaGPU::setLocalBaseVaddr(Addr addr)
{
    assert(!localBaseVaddr);
    localBaseVaddr = addr;
}

uint64_t CudaGPU::getLocalBaseVaddr()
{
    if (!localBaseVaddr) {
        panic("Local base virtual address is unset!"
              " Make sure bench was compiled with latest libcuda.\n");
    }
    return localBaseVaddr;
}

Addr CudaGPU::GPUPageTable::addrToPage(Addr addr)
{
    Addr offset = addr % TheISA::PageBytes;
    return addr - offset;
}

void CudaGPU::GPUPageTable::serialize(CheckpointOut &cp) const
{
    unsigned int num_ptes = pageMap.size();
    unsigned int index = 0;
    Addr* pagetable_vaddrs = new Addr[num_ptes];
    Addr* pagetable_paddrs = new Addr[num_ptes];
    std::map<Addr, Addr>::const_iterator it = pageMap.begin();
    for (; it != pageMap.end(); ++it) {
        pagetable_vaddrs[index] = (*it).first;
        pagetable_paddrs[index++] = (*it).second;
    }
    SERIALIZE_SCALAR(num_ptes);
    SERIALIZE_ARRAY(pagetable_vaddrs, num_ptes);
    SERIALIZE_ARRAY(pagetable_paddrs, num_ptes);
    delete[] pagetable_vaddrs;
    delete[] pagetable_paddrs;
}

void CudaGPU::GPUPageTable::unserialize(CheckpointIn &cp)
{
    unsigned int num_ptes = 0;
    UNSERIALIZE_SCALAR(num_ptes);
    Addr* pagetable_vaddrs = new Addr[num_ptes];
    Addr* pagetable_paddrs = new Addr[num_ptes];
    UNSERIALIZE_ARRAY(pagetable_vaddrs, num_ptes);
    UNSERIALIZE_ARRAY(pagetable_paddrs, num_ptes);
    for (unsigned int i = 0; i < num_ptes; ++i) {
        pageMap[pagetable_vaddrs[i]] = pagetable_paddrs[i];
    }
    delete[] pagetable_vaddrs;
    delete[] pagetable_paddrs;
}

void CudaGPU::registerDeviceMemory(ThreadContext *tc, Addr vaddr, size_t size)
{
    if (manageGPUMemory || accessHostPageTable) return;
    DPRINTF(CudaGPUPageTable, "Registering device memory vaddr: %x, size: %d\n", vaddr, size);
    // Get the physical address of full memory allocation (i.e. all pages)
    Addr page_vaddr, page_paddr;
    for (ChunkGenerator gen(vaddr, size, TheISA::PageBytes); !gen.done(); gen.next()) {
        page_vaddr = pageTable.addrToPage(gen.addr());
        if (FullSystem) {
            page_paddr = TheISA::vtophys(tc, page_vaddr);
        } else {
            tc->getProcessPtr()->pTable->translate(page_vaddr, page_paddr);
        }
        pageTable.insert(page_vaddr, page_paddr);
    }
}

void CudaGPU::registerDeviceInstText(ThreadContext *tc, Addr vaddr, size_t size)
{
    if (manageGPUMemory) {
        // Allocate virtual and physical memory for the device text
        Addr gpu_vaddr = allocateGPUMemory(size);
        setInstBaseVaddr(gpu_vaddr);
    } else {
        setInstBaseVaddr(vaddr);
        registerDeviceMemory(tc, vaddr, size);
    }
}

Addr CudaGPU::allocateGPUMemory(size_t size)
{
    assert(manageGPUMemory);
    DPRINTF(CudaGPUPageTable, "GPU allocating %d bytes\n", size);

    if (size == 0) return 0;

    // TODO: When we need to reclaim memory, this will need to be modified
    // heavily to actually track allocated and free physical and virtual memory

    // Cache block align the allocation size
    size_t block_part = size % ruby->getBlockSizeBytes();
    size_t aligned_size = size + (block_part ? (ruby->getBlockSizeBytes() - block_part) : 0);

    Addr base_vaddr = virtualGPUBrkAddr;
    virtualGPUBrkAddr += aligned_size;
    Addr base_paddr = physicalGPUBrkAddr;
    physicalGPUBrkAddr += aligned_size;

    if (virtualGPUBrkAddr > gpuMemoryRange.size()) {
        panic("Ran out of GPU memory!");
    }

    // Map pages to physical pages
    for (ChunkGenerator gen(base_vaddr, aligned_size, TheISA::PageBytes); !gen.done(); gen.next()) {
        Addr page_vaddr = pageTable.addrToPage(gen.addr());
        Addr page_paddr;
        if (page_vaddr <= base_vaddr) {
            page_paddr = base_paddr - (base_vaddr - page_vaddr);
        } else {
            page_paddr = base_paddr + (page_vaddr - base_vaddr);
        }
        DPRINTF(CudaGPUPageTable, "  Trying to allocate page at vaddr %x with addr %x\n", page_vaddr, gen.addr());
        pageTable.insert(page_vaddr, page_paddr);
    }

    DPRINTF(CudaGPUAccess, "Allocating %d bytes for GPU at address 0x%x\n", size, base_vaddr);

    return base_vaddr;
}

void CudaGPU::regStats()
{
    numKernelsStarted
        .name(name() + ".kernels_started")
        .desc("Number of kernels started");
    numKernelsCompleted
        .name(name() + ".kernels_completed")
        .desc("Number of kernels completed");
}

GPGPUSimComponentWrapper *GPGPUSimComponentWrapperParams::create() {
    return new GPGPUSimComponentWrapper(this);
}

CudaGPU::GMemory::GMemory(): name("GMemory") {
    m_gMemSize = 0;
    m_xlAllocatedBlocks = NULL;
    m_lAllocatedBlocks = NULL;
    m_sAllocatedBlocks = NULL;
    m_xlNumBlocks = 0;
    m_lNumBlocks = 0;
    m_sNumBlocks = 0;
    m_largeBlocksBytes = 0; 
    m_lastsBlock = 0;
    m_lastlBlock = 0;
    m_lastxlBlock = 0;
}

CudaGPU::GMemory::~GMemory() {
    if (m_xlAllocatedBlocks)
        delete [] m_xlAllocatedBlocks;
    if (m_lAllocatedBlocks)
        delete [] m_lAllocatedBlocks;
    if(m_sAllocatedBlocks)
        delete [] m_sAllocatedBlocks;
}

void CudaGPU::GMemory::setMem(Addr add, size_t size) {
    assert(XLGBLOCK_SIZE%SGBLOCK_SIZE == 0);
    assert(LGBLOCK_SIZE%SGBLOCK_SIZE == 0);
    size_t ca = TheISA::PageBytes - (add%TheISA::PageBytes);
    add+= ca;
    size-= ca;
    size_t align = size%SGBLOCK_SIZE;
    m_gMemStart = add;
    m_gMemSize = size-align;
    m_largeBlocksBytes = XLGBLOCK_COUNT*XLGBLOCK_SIZE + LGBLOCK_COUNT*LGBLOCK_SIZE;
    assert(size >= (m_largeBlocksBytes));
    m_xlNumBlocks = XLGBLOCK_COUNT;
    m_lNumBlocks = LGBLOCK_COUNT; 
    m_sNumBlocks = (size - m_largeBlocksBytes) / SGBLOCK_SIZE;
    assert(m_lAllocatedBlocks == NULL); // memory is already initialized
    m_xlAllocatedBlocks = new bool[m_xlNumBlocks];
    m_lAllocatedBlocks = new bool[m_lNumBlocks];
    m_sAllocatedBlocks = new bool[m_sNumBlocks];
    std::memset(m_xlAllocatedBlocks, 0, m_xlNumBlocks * sizeof (bool));
    std::memset(m_lAllocatedBlocks, 0, m_lNumBlocks * sizeof (bool));
    std::memset(m_sAllocatedBlocks, 0, m_sNumBlocks * sizeof (bool));
}

Addr CudaGPU::GMemory::mallocMem(unsigned size){
    if (size == 0) return (Addr) NULL;
    if (size > XLGBLOCK_SIZE)
        panic("Graphics memory allocation > %d are not supported", XLGBLOCK_SIZE);
    
    if(size<=SGBLOCK_SIZE){
        for (int i = 0; i < m_sNumBlocks; i++){
           int idx = (i+m_lastsBlock)%m_sNumBlocks;
           if (!m_sAllocatedBlocks[idx]) {
               m_sAllocatedBlocks[idx] = true;
               DPRINTF(CudaGPU, "allocating SGBLOCK %d\n", idx);
               Addr addr= m_gMemStart + idx * SGBLOCK_SIZE + m_largeBlocksBytes;
               assert((addr+SGBLOCK_SIZE) <= (m_gMemStart+m_gMemSize));
               m_lastsBlock = idx;
               return addr;
           }
        }
    }    

    if(size<=LGBLOCK_SIZE){
       for (int i = 0; i < m_lNumBlocks; i++){
           int idx = (i+m_lastlBlock)%m_lNumBlocks;
           if (!m_lAllocatedBlocks[idx]) {
               DPRINTF(CudaGPU, "allocating LGBLOCK %d\n", idx);
               m_lAllocatedBlocks[idx] = true;
               Addr addr = m_gMemStart + idx * LGBLOCK_SIZE + XLGBLOCK_SIZE*m_xlNumBlocks;
               assert((addr+LGBLOCK_SIZE) <= (m_gMemStart+m_gMemSize));
               m_lastlBlock = idx;
               return addr;
           }
       }
    }
   
    assert(size<=XLGBLOCK_SIZE);
    for (int i = 0; i < m_xlNumBlocks; i++){
        int idx = (i+m_lastxlBlock)%m_xlNumBlocks;
        if (!m_xlAllocatedBlocks[idx]) {
            DPRINTF(CudaGPU, "allocating XLGBLOCK %d\n", idx);
            m_xlAllocatedBlocks[idx] = true;
            Addr addr = m_gMemStart + idx * XLGBLOCK_SIZE;
            assert((addr+XLGBLOCK_SIZE) <= (m_gMemStart+m_gMemSize));
            m_lastxlBlock = idx;
            return addr;
        }
    }

    //out of memory!
    panic("Running out of graphics memory!");
    return (Addr) NULL;
}

void CudaGPU::GMemory::freeMem(Addr addr) {
    assert(addr >= m_gMemStart and addr <= (m_gMemStart + m_gMemSize));
    Addr blockAddr = addr - m_gMemStart;
    assert(blockAddr<=m_gMemSize);
    if(blockAddr>= m_largeBlocksBytes){
        blockAddr-= m_largeBlocksBytes;
        blockAddr/= SGBLOCK_SIZE;
        assert(blockAddr<m_sNumBlocks);
        assert(m_sAllocatedBlocks[blockAddr] == true);
        DPRINTF(CudaGPU, "freeing SGBLOCK %d\n", blockAddr);
        m_sAllocatedBlocks[blockAddr] = false;
        return;
    }
    
    if(blockAddr>= XLGBLOCK_SIZE*m_xlNumBlocks){
        blockAddr-= XLGBLOCK_SIZE*m_xlNumBlocks;
        unsigned blockBit = blockAddr / LGBLOCK_SIZE;
        assert(blockBit < m_lNumBlocks);
        assert(m_lAllocatedBlocks[blockBit] == true);
        DPRINTF(CudaGPU, "freeing LGBLOCK %d\n", blockBit);
        m_lAllocatedBlocks[blockBit] = false;
        return;
    }
    
    unsigned blockBit = blockAddr / XLGBLOCK_SIZE;
    assert(blockBit < m_xlNumBlocks);
    assert(m_xlAllocatedBlocks[blockBit] == true);
    DPRINTF(CudaGPU, "freeing XLGBLOCK %d\n", blockBit);
    m_xlAllocatedBlocks[blockBit] = false;
    return;
}

/**
* virtual process function that is invoked when the callback
* queue is executed.
*/
void GPUExitCallback::process()
{
    std::ostream *os = simout.find(stats_filename);
    if (!os) {
        os = simout.create(stats_filename);
    }
    gpu->gpuPrintStats(*os);
    *os << std::endl;
}

