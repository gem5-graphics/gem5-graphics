#ifndef __GEM5_GRAPHICS_CALLS_HH__
#define __GEM5_GRAPHICS_CALLS_HH__

#include <mutex>
#include "graphics/graphicsStream.hh"
#include "cpu/thread_context.hh"
#include "debug/GraphicsCalls.hh"

class gem5GraphicsCalls_t {
public:
   static gem5GraphicsCalls_t gem5GraphicsCalls;
   static void setFrameBufferSize(int bufferWidth, int bufferHeight){
     _frameBufferWidth = bufferWidth;
     _frameBufferHeight = bufferHeight;
   }
   static int getFrameBufferWidth() { return _frameBufferWidth;}
   static int getFrameBufferHeight() { return _frameBufferHeight;}
   void init_gem5_graphics();
   void executeGraphicsCommand(ThreadContext *tc, uint64_t gpusysno, uint64_t call_params);

   void static RemoveFrameDir(){
      simout.remove(_dirName, true);
   }

   static OutputDirectory* CreateFrameDir(){
     RemoveFrameDir();
     return simout.createSubdirectory(_dirName);
   }

private:
   static int _frameBufferWidth;
   static int _frameBufferHeight;
   static std::string _dirName;
};

#endif
