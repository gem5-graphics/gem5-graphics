#include "sim/simulate.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "graphics/graphics_syscall_helper.hh"
#include "graphics/gem5_graphics_calls.h"
#include "graphics/serialize_graphics.hh"

extern unsigned g_active_device;
extern "C" bool GPGPUSimSimulationActive();
extern renderData_t g_renderData;

gem5GraphicsCalls_t gem5GraphicsCalls_t::gem5GraphicsCalls;

void gem5GraphicsCalls_t::executeGraphicsCommand(ThreadContext *tc, uint64_t gpusysno, uint64_t call_params) {
    init_gem5_graphics();

    GraphicsSyscallHelper helper(tc, (graphicssyscall_t*) call_params);

    uint32_t buf_val = *((uint32_t*) helper.getParam(0));
    uint32_t buf_len = *((uint32_t*) helper.getParam(1));
    uint64_t buf_val64 = buf_val;


#define CALL_GSERIALIZE_CMD \
    checkpointGraphics::serializeGraphicsCommand(\
    pid, tid, gpusysno, (uint8_t*) buf_val64, buf_len );\
    
#define CALL_GSERIALIZE \
    checkpointGraphics::serializeGraphicsCommand(\
    pid, tid, gpusysno, iobuffer, buf_len );\

    DPRINTF(GraphicsCalls, "gem5pipe: command %d with buffer address %x and length of %d from pid= %d and tid= %d\n", gpusysno, buf_val, buf_len, helper.getPid(), helper.getTid());

    //================================================
    //================================================
    //graphics memory and blocking management 
    int tid = helper.getTid();
    int pid = helper.getPid();

    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    switch (gpusysno) {
        case gem5_graphics_mem:
        {
            CALL_GSERIALIZE_CMD;
            DPRINTF(GraphicsCalls, "gem5pipe: receiving graphics memory address addr=%x, len=%d\n", buf_val, buf_len);
            //setting the tc for graphics
            if (buf_val != 0)
                cudaGPU->setGraphicsMem(pid, (Addr) buf_val, (unsigned) buf_len);
            return;
        }
            break;
        case gem5_block:
        {
            CALL_GSERIALIZE_CMD;
            //we cannot issue more graphics calls while other are rendering
            uint32_t blockT = cudaGPU->isStreamManagerEmpty() ? 0 : 1;
            DPRINTF(GraphicsCalls, "returning a graphics block flag value of %d\n", blockT);
            helper.setReturn((uint8_t*) & blockT, sizeof (uint32_t));
            return;
        }
            break;
        case gem5_debug:
        case gem5_call_buffer_fail:
        {
            CALL_GSERIALIZE_CMD;
            DPRINTF(GraphicsCalls, "gem5pipe: receiving a graphics debug call %d with val=%x, len=%d\n",
                    gpusysno, buf_val, buf_len);
            return;
        }
            break;
        case gem5_sim_active:
        {
            CALL_GSERIALIZE_CMD;
            uint32_t active = GPGPUSimSimulationActive() ? 1 : 0;
            DPRINTF(GraphicsCalls, "returning a sim_active flag value = %d\n", active);
            helper.setReturn((uint8_t*) & active, sizeof (uint32_t));
            return;
        }
            break;
        default: break;
    }

    //================================================
    //================================================
    //graphics communication
    cudaGPU->setGraphicsTC(tc, pid);
    int32_t pkt_opcode;
    uint8_t * bufferVal = new uint8_t[buf_len];
    helper.readBlob(buf_val, bufferVal, buf_len);
    memcpy(&pkt_opcode, bufferVal, 4);

    DPRINTF(GraphicsCalls, "gem5pipe: packet opcode is %d\n", pkt_opcode);

    graphicsStream * stream = graphicsStream::get(tid, pid);
    g_renderData.setTcInfo(pid, tid);

    //buffer will be used to send/recv stream data
    uint8_t * iobuffer = new uint8_t[buf_len];

    DPRINTF(GraphicsCalls, "gem5pipe: iobuffer value = %x\n", (uint64_t) iobuffer);
    //if we are calling writeFully/commit then we copy the content from the android buffer and send it with the same len value
    if (gpusysno == gem5_writeFully) {
        DPRINTF(GraphicsCalls, "gem5pipe:  gpusysno writeFully\n");
        helper.readBlob(buf_val, iobuffer, buf_len);
        CALL_GSERIALIZE;
        assert(SocketStream::bytesSentFromMain == 0);
        assert(SocketStream::currentMainWriteSocket == -1);
        SocketStream::bytesSentFromMain = buf_len;
        SocketStream::currentMainWriteSocket = stream->getSocketNum();
        uint32_t ret = stream->writeFully(iobuffer, buf_len);
        DPRINTF(GraphicsCalls, "gem5pipe:  writeFully returned %d\n", ret);
        SocketStream::lockMainThread();
        while(!SocketStream::allRenderSocketsReady()); //wait till all other threads are waiting
        helper.setReturn((uint8_t*) & ret, sizeof (uint32_t));
    } else if (gpusysno == gem5_readFully) {
       DPRINTF(GraphicsCalls, "gem5pipe: gpusysno readFully\n");
       if(buf_len > 0) {
          SocketStream::currentMainReadSocket = stream->getSocketNum();
          const uint8_t* ret = stream->readFully(iobuffer, buf_len);
          SocketStream::currentMainReadSocket = -1;
          CALL_GSERIALIZE;
          int newByteCount = SocketStream::bytesSentToMain - buf_len;
          assert(newByteCount >= 0);
          bool cond = (SocketStream::bytesSentToMain == SocketStream::totalBytesSentToMain) and (buf_len > 0);
          if(cond){
             SocketStream::readUnlock();
             SocketStream::lockMainThread();
          }
          while(!SocketStream::allRenderSocketsReady());
          SocketStream::bytesSentToMain = newByteCount;

          helper.writeBlob(buf_val, iobuffer, buf_len);
          DPRINTF(GraphicsCalls, "gem5pipe: readfully buffer value is %d \n", *((uint32_t*) iobuffer));
          DPRINTF(GraphicsCalls, "gem5pipe: readFully returned %x\n", (uint64_t) ret);
          if (ret == iobuffer)
             helper.setReturn((uint8_t*) & buf_val, sizeof (uint32_t));
          else if (ret == NULL) {
             uint32_t nret = (uint32_t) NULL;
             helper.setReturn((uint8_t*) & nret, sizeof (uint32_t));
          } else {
             DPRINTF(GraphicsCalls, "gem5pipe: unexpected return value for readFully\n");
             exit(1);
          }
       }
    } else if (gpusysno == gem5_read) {
        panic("Unexpected stream read from guest system\n");
    } else if (gpusysno == gem5_recv) {
        panic("Unexpected stream recv from guest system\n");
    } else {
        panic("Unexpected stream command\n");
    }

    android_repaint();
    delete [] iobuffer;
}
