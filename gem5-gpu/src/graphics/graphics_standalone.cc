/*
 * Copyright (c) 2018 University of British Columbia
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
 *
 * Authors: Ayub A. Gubran
 */
#include "graphics/graphics_standalone.hh"

#include <dlfcn.h>

#include "base/misc.hh"
#include "base/statistics.hh"
#include "debug/GraphicsStandalone.hh"
#include "sim/sim_events.hh"
#include "sim/stats.hh"
#include "sim/system.hh"


GraphicsStandalone::GraphicsStandalone(const Params *p) :
      ClockedObject(p),
      tickEvent(this),
      traceStarted(false),
      traceDone(false),
      tracePath(p->trace_path),
      traceThread(NULL)
{
   if(tracePath.size() == 0)
      panic("No graphics trace is specified");

   schedule(tickEvent, 0);
}

GraphicsStandalone::~GraphicsStandalone(){
   if(traceThread != NULL)
      delete traceThread;
}

void
GraphicsStandalone::init()
{
}

void
GraphicsStandalone::tick()
{
   if(traceDone){
      panic("need to check gpu has no work left");
      exitSimLoop("Done with trace\n");
      return;
   } 

   if(not traceStarted){
      //start api trace
      DPRINTF(GraphicsStandalone, "starting trace %s\n", tracePath.c_str());
      traceThread = new std::thread(&GraphicsStandalone::runTrace, this, std::ref(tracePath));
      traceStarted = true;
   }

   schedule(tickEvent, clockEdge(Cycles(1)));
}


//extern int run_retrace(int argc, char** argv);

void GraphicsStandalone::runTrace(const std::string& pTracePath){
   const char* libpath = std::getenv("APITRACE_LIB_PATH");
   if(libpath == NULL){
      panic("APITRACE_LIB_PATH is not set!\n");
   }
   int (*run_retrace)(int argc, char** argv);
   void* handle = dlopen(libpath, RTLD_LAZY | RTLD_GLOBAL );
   if(!handle){
      panic("Couldn't open retrace library at %s\n", libpath);
   }
   *(void**)(&run_retrace) = dlsym(handle, "run_retrace");
   if(!run_retrace){
      dlclose(handle);
      panic("Couldn't find function run_retrace at %s\n", libpath);
   }
   char* argv [2];
   argv[0] = new char[100];
   argv[1] = new char[2048];
   strcpy(argv[0], "glretrace");
   strcpy(argv[1], pTracePath.c_str());
   run_retrace(2, argv);
   dlclose(handle);
   traceDone = true;
   delete argv[0];
   delete argv[1];
}


GraphicsStandalone *GraphicsStandaloneParams::create() {
   return new GraphicsStandalone(this);
}

