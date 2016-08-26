// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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

#include "gpu/gpgpu-sim/cuda_gpu.hh"

#include "stream_manager.h"
#include "gpgpusim_entrypoint.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"

unsigned CUstream_st::sm_next_stream_uid = 0;

CUstream_st::CUstream_st() 
{
    m_pending = false;
    m_uid = sm_next_stream_uid++;
}

bool CUstream_st::empty()
{
    bool empty = m_operations.empty();
    return empty;
}

bool CUstream_st::busy()
{
    bool pending = m_pending;
    return pending;
}

void CUstream_st::synchronize() 
{
    // called by host thread
    bool done=false;
    do{
        done = m_operations.empty();
    } while ( !done );
}

void CUstream_st::push( const stream_operation &op )
{
    // called by host thread
    m_operations.push_back( op );
}

void CUstream_st::record_next_done()
{
    // called by gpu thread
    assert(m_pending);
    m_operations.pop_front();
    m_pending=false;
}


stream_operation CUstream_st::next()
{
    // called by gpu thread
    m_pending = true;
    stream_operation result = m_operations.front();
    return result;
}

void CUstream_st::print(FILE *fp)
{
    fprintf(fp,"GPGPU-Sim API:    stream %u has %zu operations\n", m_uid, m_operations.size() );
    std::list<stream_operation>::iterator i;
    unsigned n=0;
    for( i=m_operations.begin(); i!=m_operations.end(); i++ ) {
        stream_operation &op = *i;
        fprintf(fp,"GPGPU-Sim API:       %u : ", n++);
        op.print(fp);
        fprintf(fp,"\n");
    }
}


void stream_operation::do_operation( gpgpu_sim *gpu )
{
    if( is_noop() ) 
        return;

    assert(!m_done && m_stream);
    m_stream->setThreadContext(tc);
    if(g_debug_execution >= 3)
       printf("GPGPU-Sim API: stream %u performing ", m_stream->get_uid() );
    switch( m_type ) {
    case stream_memcpy_host_to_device:
        if(g_debug_execution >= 3)
            printf("memcpy host-to-device\n");
        gpu->gem5CudaGPU->memcpy((void *)m_host_address_src,(void *)m_device_address_dst,m_cnt, m_stream, m_type);
        break;
    case stream_memcpy_device_to_host:
        if(g_debug_execution >= 3)
            printf("memcpy device-to-host\n");
        gpu->gem5CudaGPU->memcpy((void *)m_device_address_src,m_host_address_dst,m_cnt, m_stream, m_type);
        break;
    case stream_memcpy_device_to_device:
        if(g_debug_execution >= 3)
            printf("memcpy device-to-device\n");
        gpu->gem5CudaGPU->memcpy((void *)m_device_address_src,(void *)m_device_address_dst,m_cnt, m_stream, m_type);
        break;
    case stream_memcpy_to_symbol:
        if(g_debug_execution >= 3)
            printf("memcpy to symbol\n");
        gpu->gem5CudaGPU->memcpy_symbol(m_symbol, m_host_address_src, m_cnt, m_offset, 1, m_stream);
        break;
    case stream_memcpy_from_symbol:
        if(g_debug_execution >= 3)
            printf("memcpy from symbol\n");
        printf("Received stream_memcpy_from_symbol request, which is not implemented!\n");
        abort();
        break;
    case stream_kernel_launch:
        if( gpu->can_start_kernel() ) {
        	gpu->set_cache_config(m_kernel->name());
        	printf("kernel \'%s\' transfer to GPU hardware scheduler\n", m_kernel->name().c_str() );
            if( m_sim_mode )
                gpgpu_cuda_ptx_sim_main_func( *m_kernel );
            else {
                gpu->launch( m_kernel );
                gpu->gem5CudaGPU->beginRunning(launchTime, m_stream);
            }
        }
        break;
    case stream_event: {
        printf("event update\n");
        time_t wallclock = time((time_t *)NULL);
        m_event->update( gpu_tot_sim_cycle, wallclock, gpu->gem5CudaGPU->getCurTick() );
        if (m_event->needs_unblock()) {
            gpu->gem5CudaGPU->unblockThread(NULL);
            m_event->reset();
        }
        m_stream->record_next_done();
        gpu->gem5CudaGPU->scheduleStreamEvent();
        } 
        break;
    case stream_memset:
        if(g_debug_execution >= 3)
            printf("memset\n");
        gpu->gem5CudaGPU->memset((Addr)m_device_address_dst, m_write_value, m_cnt, m_stream);
        break;
    default:
        abort();
    }
    m_done=true;
    fflush(stdout);
}

void stream_operation::print( FILE *fp ) const
{
    fprintf(fp," stream operation " );
    switch( m_type ) {
    case stream_event: fprintf(fp,"event"); break;
    case stream_kernel_launch: fprintf(fp,"kernel"); break;
    case stream_memcpy_device_to_device: fprintf(fp,"memcpy device-to-device"); break;
    case stream_memcpy_device_to_host: fprintf(fp,"memcpy device-to-host"); break;
    case stream_memcpy_host_to_device: fprintf(fp,"memcpy host-to-device"); break;
    case stream_memcpy_to_symbol: fprintf(fp,"memcpy to symbol"); break;
    case stream_memcpy_from_symbol: fprintf(fp,"memcpy from symbol"); break;
    case stream_no_op: fprintf(fp,"no-op"); break;
    case stream_memset: fprintf(fp,"memset"); break;
    }
}

stream_manager::stream_manager( gpgpu_sim *gpu, bool cuda_launch_blocking ) 
{
    m_gpu = gpu;
    m_service_stream_zero = false;
    m_cuda_launch_blocking = cuda_launch_blocking;
}

bool stream_manager::operation( bool * sim)
{
    bool check=check_finished_kernel();
    if(check)m_gpu->print_stats();
    stream_operation op =front();
    op.do_operation( m_gpu );
    // simulate a clock cycle on the GPU
    return check;
}

bool stream_manager::check_finished_kernel()
{

	unsigned grid_uid = m_gpu->finished_kernel();
	bool check=register_finished_kernel(grid_uid);
	return check;

}

bool stream_manager::register_finished_kernel(unsigned grid_uid)
{
    // called by gpu simulation thread
    if(grid_uid > 0){
    CUstream_st *stream = m_grid_id_to_stream[grid_uid];
    kernel_info_t *kernel = stream->front().get_kernel();
    assert( grid_uid == kernel->get_uid() );
    stream->record_next_done();
    m_grid_id_to_stream.erase(grid_uid);
    delete kernel;
    return true;
    }else{
    	return false;
    }
    return false;
}

stream_operation stream_manager::front() 
{
    // called by gpu simulation thread
    stream_operation result;
    if( concurrent_streams_empty() )
        m_service_stream_zero = true;
    if( m_service_stream_zero ) {
        if( !m_stream_zero.empty() ) {
            if( !m_stream_zero.busy() ) {
                result = m_stream_zero.next();
                if( result.is_kernel() ) {
                    unsigned grid_id = result.get_kernel()->get_uid();
                    m_grid_id_to_stream[grid_id] = &m_stream_zero;
                }
            }
        } else {
            m_service_stream_zero = false;
        }
    } else {
        std::list<struct CUstream_st*>::iterator s;
        for( s=m_streams.begin(); s != m_streams.end(); s++) {
            CUstream_st *stream = *s;
            if( !stream->busy() && !stream->empty() ) {
                result = stream->next();
                if( result.is_kernel() ) {
                    unsigned grid_id = result.get_kernel()->get_uid();
                    m_grid_id_to_stream[grid_id] = stream;
                }
                break;
            }
        }
    }
    return result;
}

bool stream_manager::ready()
{
    bool ready = false;
    if( concurrent_streams_empty() )
        m_service_stream_zero = true;
    if( m_service_stream_zero ) {
        if( !m_stream_zero.empty() ) {
            if( !m_stream_zero.busy() ) {
                ready = true;
            }
        } else {
            m_service_stream_zero = false;
        }
    } else {
        std::list<struct CUstream_st*>::iterator s;
        for( s=m_streams.begin(); s != m_streams.end(); s++) {
            CUstream_st *stream = *s;
            if( !stream->busy() && !stream->empty() ) {
                ready = true;
            }
        }
    }
    return ready;
}

void stream_manager::add_stream( struct CUstream_st *stream )
{
    // called by host thread
    m_streams.push_back(stream);
}

void stream_manager::destroy_stream( CUstream_st *stream )
{
    // called by host thread
    while( !stream->empty() )
        ; 
    std::list<CUstream_st *>::iterator s;
    for( s=m_streams.begin(); s != m_streams.end(); s++ ) {
        if( *s == stream ) {
            m_streams.erase(s);
            break;
        }
    }
    delete stream; 
}

bool stream_manager::concurrent_streams_empty()
{
    bool result = true;
    // called by gpu simulation thread
    std::list<struct CUstream_st *>::iterator s;
    for( s=m_streams.begin(); s!=m_streams.end();++s ) {
        struct CUstream_st *stream = *s;
        if( !stream->empty() ) {
            //stream->print(stdout);
            result = false;
        }
    }
    return result;
}

bool stream_manager::empty_protected()
{
    bool result = true;
    if( !concurrent_streams_empty() )
        result = false;
    if( !m_stream_zero.empty() )
        result = false;
    return result;
}

bool stream_manager::empty()
{
    bool result = true;
    if( !concurrent_streams_empty() ) 
        result = false;
    if( !m_stream_zero.empty() ) 
        result = false;
    return result;
}


void stream_manager::print( FILE *fp)
{
    print_impl(fp);
}
void stream_manager::print_impl( FILE *fp)
{
    fprintf(fp,"GPGPU-Sim API: Stream Manager State\n");
    std::list<struct CUstream_st *>::iterator s;
    for( s=m_streams.begin(); s!=m_streams.end();++s ) {
        struct CUstream_st *stream = *s;
        if( !stream->empty() ) 
            stream->print(fp);
    }
    if( !m_stream_zero.empty() ) 
        m_stream_zero.print(fp);
}

void stream_manager::push( stream_operation op )
{
    struct CUstream_st *stream = op.get_stream();

    if( stream && !m_cuda_launch_blocking ) {
        stream->push(op);
    } else {
        op.set_stream(&m_stream_zero);
        m_stream_zero.push(op);
    }
    if(g_debug_execution >= 3)
       print_impl(stdout);

    if (ready()) {
        m_gpu->gem5CudaGPU->scheduleStreamEvent();
    }
}

