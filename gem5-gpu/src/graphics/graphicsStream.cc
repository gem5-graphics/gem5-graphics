#include "graphicsStream.hh"
#include "graphic_calls.hh"
#include <stdlib.h>

std::map<int, std::map<int, graphicsStream*> > graphicsStream::m_connStreams;

graphicsStream* graphicsStream::get(int tid, int pid) {
    if (m_connStreams[pid][tid] == NULL) {
        UnixStream * unix_stream = new UnixStream((size_t)STREAM_BUFFER_SIZE);
        char rendererAddress[256];
        android_gles_server_path(rendererAddress, sizeof(rendererAddress));
        inform("connecting at port %s \n",rendererAddress);
        if (unix_stream->connect(rendererAddress)!=0){
            inform("ThreadId=%d, failed to open Unix Stream connection with renderer\n", tid);
            exit(1);
        }
        SocketStream::incSockets();
        graphicsStream * gstream = new graphicsStream(unix_stream, tid, pid);
        SocketStream::regMainSocket(unix_stream->getSocketNum());
        m_connStreams[pid][tid] = gstream;
    }
    //now it should be either already existing or just has been established
    return m_connStreams[pid][tid];
}

int graphicsStream::writeFully(const void *buf, size_t len){
    return m_stream->writeFully(buf,len);
}

const unsigned char* graphicsStream::readFully(void *buf, size_t len){
    return m_stream->readFully(buf,len);
}

const unsigned char* graphicsStream::read( void *buf, size_t *inout_len){
    return m_stream->read(buf,inout_len);
}


int graphicsStream::recv(void *buf, size_t len){
    return m_stream->recv(buf,len);
}

void * graphicsStream::allocBuffer(size_t size){
    return m_stream->allocBuffer(size);
}
