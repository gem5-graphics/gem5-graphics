#include "base/trace.hh"
#include "graphicsStream.hh"
#include "graphics/emugl/opengles.h"
#include "debug/GraphicsCalls.hh"
#include <stdlib.h>

std::map<int, std::map<int, graphicsStream*> > graphicsStream::m_connStreams;

graphicsStream* graphicsStream::get(int tid, int pid) {
    if (m_connStreams[pid][tid] == NULL) {
        inform("connecting to a new channel\n");
        const auto& renderer = android_getOpenglesRenderer();
        emugl::RenderChannelPtr cnl= renderer->createRenderChannel(nullptr);
        if (cnl==nullptr){
            inform("ThreadId=%d, failed to open render channel\n");
            exit(1);
        }
        graphicsStream * gstream = new graphicsStream(cnl, tid, pid);
        m_connStreams[pid][tid] = gstream;
    }

    //now it should either exist already or just has been established
    return m_connStreams[pid][tid];
}

uint32_t graphicsStream::read(uint8_t* buffData, size_t buffSize){
    DPRINTF(GraphicsCalls, "%s: reading to guest buffer=%p, size=%d\n", __func__, buffData, buffSize);

    int len = 0;
    size_t buffOffset = 0;

    while (buffOffset != buffSize){
      if (m_dataForReadingLeft == 0) {
        // No data left, read a new chunk from the channel.
        for (;;) {
          //auto result = m_channel->tryRead(&m_dataForReading);
          auto result = m_channel->readFromHost(&m_dataForReading, true);
          if (result == emugl::RenderChannel::IoResult::Ok) {
            m_dataForReadingLeft = m_dataForReading.size();
            DPRINTF(GraphicsCalls, "%s: read %zu bytes to guest\n", __func__, m_dataForReadingLeft);
            break;
          }
          /*else {
            return 0;
          }*/
        }
      }

      const size_t curSize =
          std::min(buffSize - buffOffset, m_dataForReadingLeft);
      memcpy(buffData + buffOffset,
             m_dataForReading.data() +
             (m_dataForReading.size() - m_dataForReadingLeft),
             curSize);

      len += curSize;
      m_dataForReadingLeft -= curSize;
      buffOffset += curSize;
    }

    return len;
}

uint32_t graphicsStream::write(uint8_t* buffData, size_t len){
    DPRINTF(GraphicsCalls, "%s: writing to host buffer=%p, size=%d\n", __func__, buffData, len);

    // the total bytes to send.
    int count = len;

    // Copy everything into a single ChannelBuffer.
    emugl::RenderChannel::Buffer outBuffer;
    outBuffer.resize_noinit(count);
    auto ptr = outBuffer.data();
    memcpy(ptr, buffData, len);

    DPRINTF(GraphicsCalls, "%s: writing %d bytes to host\n", __func__, count);
    // Send it through the channel.
   //auto result = m_channel->tryWrite(std::move(outBuffer));
   auto result = m_channel->writeToHost(std::move(outBuffer));
   //if (result != emugl::RenderChannel::IoResult::Ok) {
   if (!result){
        //return 0;
        fatal("%s: tryWrite() failed with %d\n", __func__, (int)result);
    }

    return count;
}
