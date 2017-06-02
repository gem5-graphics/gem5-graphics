/*
* Copyright (C) 2011 The Android Open Source Project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#ifndef __SOCKET_STREAM_H
#define __SOCKET_STREAM_H

#include <stdlib.h>
#include <mutex>
#include <map>
#include <vector>
#include "IOStream.h"
#define STREAM_BUFFER_SIZE  4*1024*1024 //big enough to avoid blocking

class SocketStream : public IOStream {
public:
    typedef enum { ERR_INVALID_SOCKET = -1000 } SocketStreamError;
    static const size_t MAX_ADDRSTR_LEN = 256;

    explicit SocketStream(size_t bufsize = STREAM_BUFFER_SIZE);
    virtual ~SocketStream();

    virtual int listen(char addrstr[MAX_ADDRSTR_LEN]) = 0;
    virtual SocketStream *accept() = 0;
    virtual int connect(const char* addr) = 0;

    virtual void *allocBuffer(size_t minSize);
    virtual int commitBuffer(size_t size);
    virtual const unsigned char *readFully(void *buf, size_t len);
    virtual const unsigned char *readFullyForThreads(void *buf, size_t len);
    virtual const unsigned char *read(void *buf, size_t *inout_len);

    bool valid() { return m_sock >= 0; }
    virtual int recv(void *buf, size_t len);
    virtual int writeFully(const void *buf, size_t len);
    int getSocketNum() {return m_sock;}
    static bool allRenderSocketsReady();
    static void regSocket(uint64_t m_sock);
    static void regMainSocket(uint64_t sock);
    static void incSockets();
    static void incReadySockets(int sock, bool uncond);
    static void decReadySockets(int sock, bool uncond);
    static void lockMainThread();
    static void unlockMainThread();
    static void readLock();
    static void readUnlock();
    static bool locksInit;
    static int bytesSentToMain;
    static int totalBytesSentToMain;
    static int bytesSentFromMain;
    static int currentMainReadSocket;
    static int currentMainWriteSocket;

protected:
    int            m_sock;
    size_t         m_bufsize;
    unsigned char *m_buf;
    
    static int m_numRenderSockets;
    static int m_readyRenderSockets;
    static std::mutex m_sockCount;
    static std::mutex m_mainThreadLock;
    static std::mutex m_readLock;
    static std::map<int, int> m_renderSockets;
    static std::vector<int> m_mainSockets;

    SocketStream(int sock, size_t bufSize);
};

#endif /* __SOCKET_STREAM_H */
