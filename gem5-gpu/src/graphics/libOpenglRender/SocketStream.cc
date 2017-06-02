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
#include <sys/socket.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/un.h>
#include <thread>
#include "base/misc.hh"
#include "SocketStream.hh"
#include "graphics/gem5_graphics_calls.h"
#include "graphics/graphic_calls.hh"

int SocketStream::m_numRenderSockets = 0;
int SocketStream::m_readyRenderSockets = 0;
std::mutex SocketStream::m_sockCount;
std::mutex SocketStream::m_mainThreadLock;
std::mutex SocketStream::m_readLock;
std::map<int, int> SocketStream::m_renderSockets;
std::vector<int> SocketStream::m_mainSockets;
bool SocketStream::locksInit = false;
int SocketStream::bytesSentToMain = 0;
int SocketStream::bytesSentFromMain = 0;
int SocketStream::totalBytesSentToMain = 0;
//int SocketStream::currentMainSocket = -1;
int SocketStream::currentMainReadSocket = -1;
int SocketStream::currentMainWriteSocket = -1;

SocketStream::SocketStream(size_t bufSize) :
    IOStream(bufSize),
    m_sock(-1),
    m_bufsize(bufSize),
    m_buf(NULL)
{
   m_sockCount.lock();
   if(!locksInit){
      m_mainThreadLock.lock();
      m_readLock.lock();
      locksInit = true;
   }
   m_sockCount.unlock();
}

SocketStream::SocketStream(int sock, size_t bufSize) :
    IOStream(bufSize),
    m_sock(sock),
    m_bufsize(bufSize),
    m_buf(NULL)
{
}

SocketStream::~SocketStream()
{
    if (m_sock >= 0) {
#ifdef _WIN32
        closesocket(m_sock);
#else
        ::close(m_sock);
#endif
    }
    if (m_buf != NULL) {
        free(m_buf);
        m_buf = NULL;
    }
}


void *SocketStream::allocBuffer(size_t minSize)
{
    minSize = minSize >= STREAM_BUFFER_SIZE? minSize : STREAM_BUFFER_SIZE;
    size_t allocSize = (m_bufsize < minSize ? minSize : m_bufsize);
    if (!m_buf) {
        m_buf = (unsigned char *)malloc(allocSize);
    }
    else if (m_bufsize < allocSize) {
        unsigned char *p = (unsigned char *)realloc(m_buf, allocSize);
        if (p != NULL) {
            m_buf = p;
            m_bufsize = allocSize;
        } else {
            inform("%s: realloc (%zu) failed\n", __FUNCTION__, allocSize);
            free(m_buf);
            m_buf = NULL;
            m_bufsize = 0;
        }
    }

    return m_buf;
};

int SocketStream::commitBuffer(size_t size)
{
    return writeFully(m_buf, size);
}

int SocketStream::writeFully(const void* buffer, size_t size)
{
    if (!valid()) return -1;
    size_t res = size;
    int retval = 0;

    //printf("tid=%x, writeFully sending %lu bytes from %d\n", std::this_thread::get_id(), size, m_sock);

    SocketStream::incReadySockets(m_sock, false); //let the main thread continue if waiting for threads to be ready

    if((m_renderSockets.find(m_sock) != m_renderSockets.end())){
       while(currentMainReadSocket!=m_renderSockets[m_sock]); //wait till the main socket is waiting for us
       while(bytesSentToMain); //if some other thread already sent some data to the main thread wait till all data is read
       //printf("tid=%x, sok=%d: setting bytesSentToMain= %d\n", std::this_thread::get_id(), m_sock, res);
       bytesSentToMain = res; //set how much we are going to send
       totalBytesSentToMain = res;
    }

    while (res > 0) {
       ssize_t stat = ::send(m_sock, (const char *)buffer + (size - res), res, 0);
       if (stat < 0) {
            if (errno != EINTR) {
                retval =  stat;
                inform("%s: failed: %s\n", __FUNCTION__, strerror(errno));
                break;
            }
        } else {
            res -= stat;
        }
    }

    if((m_renderSockets.find(m_sock) != m_renderSockets.end())){
       SocketStream::readLock();
       SocketStream::decReadySockets(m_sock, false); //if the main thread is ready to read then dec ready sockets
       //SocketStream::readLock();
       //printf("tid=%x, unlocking main\n", std::this_thread::get_id());
       SocketStream::unlockMainThread();
    }

    return retval;
}

const unsigned char *SocketStream::readFully(void *buf, size_t len)
{
    if (!valid()) return NULL;
    if (!buf) {
      return NULL;  // do not allow NULL buf in that implementation
    }
    size_t res = len;
    while (res > 0) {
        ssize_t stat = ::recv(m_sock, (char *)(buf) + len - res, res, 0);
        if (stat > 0) {
            res -= stat;
            continue;
        }
        if (stat == 0 || errno != EINTR) { // client shutdown or error
            return NULL;
        }
    }

    //printf("tid=%x, readFully %lu bytes, left=%lu socket %d\n", std::this_thread::get_id(), len, res, m_sock);
    if((m_renderSockets.find(m_sock) != m_renderSockets.end())) {
       SocketStream::bytesSentFromMain = 0;
       SocketStream::currentMainWriteSocket = -1;
       unlockMainThread();
    }
    return (const unsigned char *)buf;
}


const unsigned char * SocketStream::readFullyForThreads(void *buf, size_t len){
   size_t res = len;
   while (res > 0) {
      ssize_t stat = ::recv(m_sock, (char *)(buf) + len - res, res, 0);
      if (stat > 0) {
         res -= stat;
         continue;
      }
      if (stat == 0 || errno != EINTR) { // client shutdown or error
         return NULL;
      }
   }
   return (const unsigned char*) buf;
}

const unsigned char *SocketStream::read( void *buf, size_t *inout_len)
{
    if (!valid()) return NULL;
    if (!buf) {
      return NULL;  // do not allow NULL buf in that implementation
    }


    SocketStream::incReadySockets(m_sock, false);
    //printf("tid=%x, read getting %d bytes to socket %d\n", std::this_thread::get_id(), *inout_len, m_sock);
    while(currentMainWriteSocket!=m_renderSockets[m_sock]); //wait till the main socket is writing to us
    *inout_len = bytesSentFromMain;

    readFullyForThreads(buf, *inout_len);
    /*int n;
    do {
        n = recv(buf, *inout_len);
    } while( n < 0 && errno == EINTR );
    */


    SocketStream::decReadySockets(m_sock, false);
    currentMainWriteSocket = -1;
    bytesSentFromMain = 0;
    unlockMainThread();
    /*if (n > 0) {
        *inout_len = n;
        return (const unsigned char *)buf;
    } */

     return (const unsigned char *)buf;
    //return NULL;
}

int SocketStream::recv(void *buf, size_t len)
{
    if (!valid()) return int(ERR_INVALID_SOCKET);
    int res = 0;
    while(true) {
        res = ::recv(m_sock, (char *)buf, len, 0);
        if (res < 0) {
            if (errno == EINTR) {
                continue;
            }
        }
        break;
    }
    return res;
}

void SocketStream::incReadySockets(int sock, bool uncond){
   m_sockCount.lock();
   if((m_renderSockets.find(sock) != m_renderSockets.end()) or uncond){
      m_readyRenderSockets++;
      //printf("inc ready sockets to %d\n", m_readyRenderSockets);
      DPRINTF(GraphicsCalls,"tid=%x, sock:%d, incReadySockets to %d\n", std::this_thread::get_id(), sock, m_readyRenderSockets);
   }
   m_sockCount.unlock();
}

void SocketStream::decReadySockets(int sock, bool uncond){
   m_sockCount.lock();
   if((m_renderSockets.find(sock) != m_renderSockets.end()) or uncond){
      m_readyRenderSockets--;
      //printf("dec ready sockets to %d\n", m_readyRenderSockets);
      DPRINTF(GraphicsCalls,"tid=%x, sock:%d, decReadySockets to %d\n", std::this_thread::get_id(), sock, m_readyRenderSockets);
   }
   m_sockCount.unlock();
}

void SocketStream::incSockets(){
    m_sockCount.lock();
    m_numRenderSockets++;
    DPRINTF(GraphicsCalls,"tid=%x, new client socket, m_numRenderSockets=%d\n",std::this_thread::get_id(), m_numRenderSockets);
    m_sockCount.unlock();
}

void SocketStream::regSocket(uint64_t sock){
   static int sockCount = 0;
   sockCount++;
   bool socketAdded = false;
    while(true) {
       m_sockCount.lock();
       if(m_mainSockets.size() >= sockCount)
          socketAdded = true;
       m_sockCount.unlock();
       if(socketAdded) break;
    }
    m_sockCount.lock();
    m_renderSockets[sock] = m_mainSockets[sockCount-1];
    DPRINTF(GraphicsCalls,"tid=%x, sock=%d mapped to %d, new client, m_numRenderSockets=%d\n",
          std::this_thread::get_id(), sock, m_mainSockets[sockCount-1], m_numRenderSockets);
    m_sockCount.unlock();
}


void SocketStream::regMainSocket(uint64_t sock){
    m_sockCount.lock();
    m_mainSockets.push_back(sock);
    m_sockCount.unlock();
}

bool SocketStream::allRenderSocketsReady(){ 
   bool res = false;
   m_sockCount.lock();
   assert(m_numRenderSockets >= m_readyRenderSockets);
   if(m_numRenderSockets == m_readyRenderSockets)
   //if((m_numRenderSockets == m_readyRenderSockets))
      res = true;
   m_sockCount.unlock();
   return res;
}

void SocketStream::lockMainThread(){
   m_mainThreadLock.lock();
}

void SocketStream::unlockMainThread(){
   m_mainThreadLock.unlock();
}

void SocketStream::readLock(){
   m_readLock.lock();
}

void SocketStream::readUnlock(){
   m_readLock.unlock();
}
