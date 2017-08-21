/*
 * File:   graphicsStream.hh
 * Author: ayub
 *
 * Created on June 26, 2014, 12:25 AM
 */

#ifndef GRAPHICSSTREAM_HH
#define GRAPHICSSTREAM_HH

#include <map>
#include "api/cuda_syscalls.hh"
#include "graphics/emugl/OpenglRender/include/RenderChannel.h"

enum gem5GraphicsCall {
    gem5_write = GEM5_GPU_CALLS_START, //start where cuda calls finish
    gem5_read, // 101
    gem5_graphics_mem, // 102
    gem5_block, // 103
    gem5_debug, // 104
    gem5_call_buffer_fail, // 105
    gem5_sim_active, // 106
    gem5_get_procId, //107

    GEM5_GPU_CALLS_END // 108
};

enum gem5DebugCall {
   gmem_alloc_fail, // 0
   gmem_lock_fail, // 1
   pipe_mem_alloc_fail, // 2
   gem5_info, // 3
   pipe_mem_alloc //4
};

class graphicsStream {
public:
    static std::map<int, std::map<int, graphicsStream*> > m_connStreams;
    static graphicsStream * get(int tid, int pid);

    graphicsStream(emugl::RenderChannelPtr channel, int tid, int pid){
        m_channel = channel;
        m_tid = tid;
        m_pid = pid;
    }

    ~graphicsStream(){
    }

    uint32_t read(uint8_t* buf, size_t len);
    uint32_t write(uint8_t* buf, size_t len);

    /*int writeFully(const char *buf, size_t len);
    const unsigned char* readFully(char *buf, size_t len);
    int recv(char *buf, size_t len);*/

private:
    //IOStream* m_stream;
    emugl::RenderChannelPtr m_channel;
    int m_pid;
    int m_tid;
    int sendChannel(const void* buf, size_t len);
    int recvChannel(const void* buf, size_t len);

    // These two variables serve as a reading buffer for the guest.
    // Each time we get a read request, first we extract a single chunk from
    // the |mChannel| into here, and then copy its content into the
    // guest-supplied memory.
    // If guest didn't have enough room for the whole buffer, we track the
    // number of remaining bytes in |mDataForReadingLeft| for the next read().
    emugl::RenderChannel::Buffer m_dataForReading;
    size_t m_dataForReadingLeft = 0;
};

#endif /* GRAPHICSSTREAM_HH */
