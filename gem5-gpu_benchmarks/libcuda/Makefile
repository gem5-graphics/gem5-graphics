# This version of the CUDA runtime is included for use with gem5-gpu. The
# following copyright should appear before other original copyright notices:
#
# Copyright (c) 2012-2014 Mark D. Hill and David A. Wood
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2009 by Tor M. Aamodt, Ali Bakhoda and the 
# University of British Columbia
# Vancouver, BC  V6T 1Z4
# All Rights Reserved.
# 
# THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
# TERMS AND CONDITIONS.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# 
# NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
# are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
# (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
# benchmarks/template/ are derived from the CUDA SDK available from 
# http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
# src/intersim/ are derived from Booksim (a simulator provided with the 
# textbook "Principles and Practices of Interconnection Networks" available 
# from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
# the corresponding legal terms and conditions set forth separately (original 
# copyright notices are left in files from these sources and where we have 
# modified a file our copyright notice appears before the original copyright 
# notice).  
# 
# Using this version of GPGPU-Sim requires a complete installation of CUDA 
# which is distributed seperately by NVIDIA under separate terms and 
# conditions.  To use this version of GPGPU-Sim with OpenCL requires a
# recent version of NVIDIA's drivers which support OpenCL.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the University of British Columbia nor the names of
# its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
#  
# 5. No nonprofit user may place any restrictions on the use of this software,
# including as modified by the user, by any other authorized user.
# 
# 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
# Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
# Vancouver, BC V6T 1Z4


CUDART_VERSION:=$(shell nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//' | sed 's/\./ /' | awk '{printf("%02u%02u", 10*int($$1), 10*$$2);}')

ifeq ($(OPENGL_SUPPORT),1) 
	GL = -DOPENGL_SUPPORT
endif

GNUC_CPP0X := $(shell gcc --version | perl -ne 'if (/gcc\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')
ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS = -std=c++0x
endif


CPP = g++
CC = gcc
AR = ar
ifeq ($(ARCH),ARM32)
	CPP = arm-linux-gnueabihf-g++-4.6
	CC = arm-linux-gnueabihf-gcc-4.6
	AR = arm-linux-gnueabihf-ar
endif

DEBUG ?= 0
RELEASE ?= 0
ifeq ($(RELEASE),1)
ifeq ($(DEBUG),1)
$(error Cannot build both debug and release)
endif
	CXXFLAGS += -DNDEBUG
endif

ifeq ($(DEBUG),1)
	CXXFLAGS += -Wall -g -fPIC $(GL) -D__DEBUG__
else
	CXXFLAGS += -O3 -g -Wall -fPIC $(GL)
endif

MEM_DEBUG ?= 0
ifeq ($(MEM_DEBUG),1)
	CXXFLAGS += -D__MEM_DEBUG__
endif

NO_TOUCH_PAGES ?= 0
ifeq ($(NO_TOUCH_PAGES),1)
	CXXFLAGS += -DNO_TOUCH_PAGES
endif

# NOTE: You may need to change this path if you are not using the
# canonical gem5-gpu directory structure. This directory includes
# m5op.h, m5ops.h, and the m5op_<arch>.S assembly files
M5OP_PATH = ../../gem5/util/m5
INCLUDES = -I./ -I$(CUDAHOME)/include -I$(M5OP_PATH)

PROG = cuda
M5OP = m5op_x86
ifeq ($(ARCH),ARM32)
	M5OP = m5op_arm
endif

CXX_SRCS = cuda_runtime_api.cc
CXXFLAGS += -DCUDART_VERSION=$(CUDART_VERSION)

.PHONY: clean

OBJS = $(CXX_SRCS:%.cc=%.o)


#--- Make rules ---
lib$(PROG).a: $(OBJS) $(M5OP).o
	$(AR) rcs lib$(PROG).a $(OBJS)
	$(AR) rcs lib$(M5OP).a $(M5OP).o

%.o: %.cc
	$(CPP) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(M5OP).o:
	$(CPP) $(CXXFLAGS) $(INCLUDES) -c $(M5OP_PATH)/$(M5OP).S -o $@

clean:
	rm -f *.o
	rm -f lib$(PROG).a lib$(M5OP).a
