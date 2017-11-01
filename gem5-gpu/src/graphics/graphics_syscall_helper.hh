/*
 * Copyright (c) 2012 Mark D. Hill and David A. Wood
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

#ifndef __GRAPHICS_SYSCALL_HELPER_HH__
#define __GRAPHICS_SYSCALL_HELPER_HH__

#include "base/types.hh"
#include "cpu/thread_context.hh"

typedef struct graphicscall {
    uint64_t unique_id;
    int32_t pid;
    int32_t tid;
    uint64_t total_bytes;
    uint64_t num_args;
    //pointers stored in unsigned 64
    uint64_t arg_lengths_ptr;
    uint64_t args_ptr;
    uint64_t ret_ptr;
    typedef uint64_t ARG_LEN_TYPE;
    typedef int32_t RET_LEN_TYPE;
} graphicssyscall_t;


class GraphicsSyscallHelper {
    ThreadContext* tc;
    Addr sim_params_ptr;
    graphicssyscall_t sim_params;
    graphicssyscall_t::ARG_LEN_TYPE* arg_lengths;
    unsigned char* args;

    void decode_package();
    void readBlob(Addr addr, uint8_t* p, int size, ThreadContext *tc, bool use_phys);
    //void readString(Addr addr, uint8_t* p, int size, ThreadContext *tc);
    void writeBlob(Addr addr, uint8_t* p, int size, ThreadContext *tc, bool use_phys);
  public:
    GraphicsSyscallHelper(ThreadContext* _tc, graphicssyscall_t* _call_params);
    GraphicsSyscallHelper(ThreadContext* _tc);
    ~GraphicsSyscallHelper();
    void* getParam(int index);
    void setReturn(unsigned char* retValue, size_t size);
    ThreadContext* getThreadContext() { return tc; }
    void readBlob(Addr addr, uint8_t* p, int size, bool use_phys = false) { readBlob(addr, p, size, tc, use_phys); }
    //void readString(Addr addr, uint8_t* p, int size) { readString(addr, p, size, tc); }
    void writeBlob(Addr addr, uint8_t* p, int size, bool use_phys = false) { writeBlob(addr, p, size, tc, use_phys); }
    int getPid(){return sim_params.pid;}
    int getTid(){return sim_params.tid;}
    bool hasParams() { return arg_lengths!=NULL; }
};

#endif
