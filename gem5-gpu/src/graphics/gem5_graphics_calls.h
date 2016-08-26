#ifndef __GEM5_GRAPHICS_CALLS_HH__
#define __GEM5_GRAPHICS_CALLS_HH__

#include <mutex>
#include "graphics/graphicsStream.hh"
#include "cpu/thread_context.hh"

class gem5GraphicsCalls_t {
public:
   static gem5GraphicsCalls_t gem5GraphicsCalls;
   void executeGraphicsCommand(ThreadContext *tc, uint64_t gpusysno, uint64_t call_params);
};

#endif
