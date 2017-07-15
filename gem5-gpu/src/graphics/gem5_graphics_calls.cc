#include "sim/simulate.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "graphics/graphics_syscall_helper.hh"
#include "graphics/gem5_graphics_calls.h"
#include "graphics/serialize_graphics.hh"
#include "graphics/mesa_gpgpusim.h"
#include "graphics/emugl/opengles.h"
#include "base/output.hh"
#include "base/framebuffer.hh"
#include "base/bitmap.hh"

extern unsigned g_active_device;
extern "C" bool GPGPUSimSimulationActive();
extern renderData_t g_renderData;

gem5GraphicsCalls_t gem5GraphicsCalls_t::gem5GraphicsCalls;

#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401

//FIXME: should be configurable
#define SCREEN_WIDTH  1920
#define SCREEN_HEIGHT 1080

static void onNewGpuFrame(void* opaque,
                          int width,
                          int height,
                          int ydir,
                          int format,
                          int type,
                          unsigned char* pixels) {
    assert(ydir == -1);
    assert(format == GL_RGBA);
    assert(type == GL_UNSIGNED_BYTE);

    static int fnum = 0;
    static OutputDirectory* outputDir = NULL;
    static OutputStream* picOut = simout.create("gem5pipe.framebuffer.bmp", true);
    static uint64_t lastFbHash = 0;
    static FrameBuffer fb(width, height);

    fnum++;
    fb.copyIn((const uint8_t*) pixels, PixelConverter::rgba8888_le);

    // skip identical frames
    uint64_t newFbHash = fb.getHash();
    if(newFbHash == lastFbHash){
      return;
    }
    lastFbHash = newFbHash;

    if(!outputDir){
      std::string dirName = "frames_gem5pipe";
      simout.remove(dirName);
      outputDir = simout.createSubdirectory(dirName);

    }

    std::stringstream ss;
    ss << "fb." << std::setw(9) << std::setfill('0') << fnum-1 << "." << curTick() << ".bmp.gz";

    Bitmap bitmap(&fb);

    OutputStream *fb_out(outputDir->create(ss.str(), true));
    //(*fb_out->stream()).write((const char*) pixels, width*height*4);
    bitmap.write(*fb_out->stream());
    picOut->stream()->seekp(0);
    bitmap.write(*picOut->stream());
    outputDir->close(fb_out);
}


void gem5GraphicsCalls_t::init_gem5_graphics(){
  static bool init= false;
  if(!init){
    init=true;

    DPRINTF(GraphicsCalls,"initializing renderer process\n");

    int major, minor;
    major = 2;
    minor = 0;

    if(!(0==android_initOpenglesEmulation() and
         0==android_startOpenglesRenderer(SCREEN_WIDTH, SCREEN_HEIGHT, true, 25, &major, &minor)))
    {
        fatal("couldn't initialize openglesEmulation and/or starting openglesRenderer");
    }

    android_setPostCallback(onNewGpuFrame, NULL);
  }
}

void gem5GraphicsCalls_t::executeGraphicsCommand(ThreadContext *tc, uint64_t gpusysno, uint64_t call_params) {
    init_gem5_graphics();

    GraphicsSyscallHelper helper(tc, (graphicssyscall_t*) call_params);

    uint32_t buf_val;
    uint32_t buf_len;
    if(helper.hasParams()){
      buf_val = *((uint32_t*) helper.getParam(0));
      buf_len = *((uint32_t*) helper.getParam(1));
    } else {
      buf_val = buf_len = 0;
    }

    uint64_t buf_val64 = buf_val;


#define CALL_GSERIALIZE_CMD \
    checkpointGraphics::serializeGraphicsCommand(\
    pid, tid, gpusysno, (uint8_t*) buf_val64, buf_len );\

#define CALL_GSERIALIZE \
    checkpointGraphics::serializeGraphicsCommand(\
    pid, tid, gpusysno, iobuffer, buf_len );\


    //================================================
    //================================================
    //graphics memory and blocking management
    int tid = helper.getTid();
    int pid = helper.getPid();

    DPRINTF(GraphicsCalls, "gem5pipe: command %lu with buffer address %x and length of %d from pid= %d and tid= %d\n", gpusysno, buf_val, buf_len, pid, tid);

    //get gpu model, null returned if gpu not enabled
    CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
    switch (gpusysno) {
        case gem5_graphics_mem:
        {
            CALL_GSERIALIZE_CMD;
            DPRINTF(GraphicsCalls, "gem5pipe: receiving graphics memory address addr=%x, len=%d\n", buf_val, buf_len);
            //setting the tc for graphics
            if (buf_val != 0){
              //check if the gpu model is enabled
              if(cudaGPU!=NULL)
                cudaGPU->setGraphicsMem(pid, (Addr) buf_val, (unsigned) buf_len);
            }
            return;
        }
            break;
        case gem5_block:
        {
            CALL_GSERIALIZE_CMD;
            //we cannot issue more graphics calls while other are rendering
            uint32_t blockT = cudaGPU==NULL? 0 : (cudaGPU->isStreamManagerEmpty() ? 0 : 1);
            DPRINTF(GraphicsCalls, "returning a graphics block flag value of %d\n", blockT);
            helper.setReturn((uint8_t*) & blockT, sizeof (uint32_t));
            return;
        }
            break;
        case gem5_debug:
        case gem5_call_buffer_fail:
        {
            CALL_GSERIALIZE_CMD;
            DPRINTF(GraphicsCalls, "gem5pipe: receiving a graphics debug call %lu with val=%x, len=%d\n",
                    gpusysno, buf_val, buf_len);
            return;
        }
            break;
        case gem5_sim_active:
        {
            CALL_GSERIALIZE_CMD;
            uint32_t active = cudaGPU==NULL? 0: GPGPUSimSimulationActive() ? 1 : 0;
            DPRINTF(GraphicsCalls, "returning a sim_active flag value = %d\n", active);
            helper.setReturn((uint8_t*) & active, sizeof (uint32_t));
            return;
        }
            break;
        case gem5_get_procId:
        {
            CALL_GSERIALIZE_CMD;
            uint64_t lpid = pid;
            DPRINTF(GraphicsCalls, "returning a gem5_get_procId = 0x%lx\n", lpid);
            helper.setReturn((uint8_t*) &lpid, sizeof (uint64_t));
            return;
        }
            break;
        default: break;
    }

    //================================================
    //================================================
    //graphics communication
    if(cudaGPU)
      cudaGPU->setGraphicsTC(tc, pid);
    int32_t pkt_opcode;
    uint8_t * bufferVal = new uint8_t[buf_len];
    helper.readBlob(buf_val, bufferVal, buf_len);
    memcpy(&pkt_opcode, bufferVal, 4);


    graphicsStream * stream = graphicsStream::get(tid, pid);

    DPRINTF(GraphicsCalls, "stream:%p, gem5pipe: packet opcode is %d\n", stream,  pkt_opcode);

    g_renderData.setTcInfo(pid, tid);

    //buffer will be used to send/recv stream data
    uint8_t * iobuffer = new uint8_t[buf_len];

    DPRINTF(GraphicsCalls, "gem5pipe: iobuffer value = %lx\n", (uint64_t) iobuffer);
    if (gpusysno == gem5_write) {
        DPRINTF(GraphicsCalls, "gem5pipe:  gpusysno write\n");
        helper.readBlob(buf_val, iobuffer, buf_len);
        CALL_GSERIALIZE;
        uint32_t ret = stream->write(iobuffer, buf_len);
        DPRINTF(GraphicsCalls, "gem5pipe:  write returned %d\n", ret);
        helper.setReturn((uint8_t*) & ret, sizeof (uint32_t));
    } else if (gpusysno == gem5_read) {
       DPRINTF(GraphicsCalls, "gem5pipe: gpusysno read\n");
       if(buf_len > 0) {
          uint32_t ret = stream->read(iobuffer, buf_len);
          CALL_GSERIALIZE;

          helper.writeBlob(buf_val, iobuffer, buf_len);
          DPRINTF(GraphicsCalls, "gem5pipe: read buffer value is %p \n",  iobuffer);
          DPRINTF(GraphicsCalls, "gem5pipe: read returned %lx\n", (uint64_t) ret);

          helper.setReturn((uint8_t*) & ret, sizeof (uint32_t));
       }
    } else {
        panic("Unexpected stream command: gpusysno=%d\n", gpusysno);
    }
    delete [] iobuffer;
}
