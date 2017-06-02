/* 
 * File:   graphicsStream.hh
 * Author: ayub
 *
 * Created on June 26, 2014, 12:25 AM
 */

#ifndef GRAPHICSSTREAM_HH
#define	GRAPHICSSTREAM_HH

#include <map>
#include "libOpenglRender/UnixStream.hh"
#include "api/cuda_syscalls.hh"
#define STREAM_BUFFER_SIZE  4*1024*1024 //big enough to avoid blocking

enum gem5GraphicsCall {
    gem5_writeFully = GEM5_GPU_CALLS_START, //start where cuda calls finish
    gem5_readFully, // 101
    gem5_read, // 102
    gem5_recv, // 103
    gem5_graphics_mem, // 104
    gem5_block, // 105
    gem5_debug, // 106
    gem5_call_buffer_fail, // 107
    gem5_sim_active, // 108
    GEM5_GPU_CALLS_END // 109
};

enum gem5DebugCall {
   gmem_alloc_fail, // 1
   gmem_lock_fail, // 2
   pipe_mem_alloc_fail, // 3
   gem5_info // 4
};

class graphicsStream {
public:
    static std::map<int, std::map<int, graphicsStream*> > m_connStreams;
    static graphicsStream * get(int tid, int pid);
    
    graphicsStream(UnixStream* unixstream, int tid, int pid){
        m_stream = unixstream;
        m_tid = tid;
        m_pid = pid;
    }
    int writeFully(const void *buf, size_t len);
    const unsigned char* readFully(void *buf, size_t len);
    const unsigned char* read( void *buf, size_t *inout_len);
    int recv(void *buf, size_t len);
    void* allocBuffer(size_t size);
    int getSocketNum() {
       return m_stream->getSocketNum();
    }
    
private:
    UnixStream * m_stream;
    int m_pid;
    int m_tid;
};


#endif	/* GRAPHICSSTREAM_HH */

