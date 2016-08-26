GNUC_CPP0X := $(shell gcc --version | perl -ne 'if (/gcc\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')
ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS = -std=c++0x
endif


CPP = g++
CC = gcc
CREATELIBRARY    =  1
DEBUG ?= 0
ifeq ($(DEBUG),1)
	CXXFLAGS +=  -Wall -g -fPIC $(GL)
else
	CXXFLAGS += -O3 -g -Wall -fPIC $(GL)
endif

M5OP = m5op_x86

OPENCL = opencl
OPENCL_SRCS = opencl_runtime_api.cc
OPENCL_OBJS = $(OPENCL_SRCS:%.cc=%.o) $(M5OP).o
OPENCLHOME ?= /usr

CXXFLAGS =

.PHONY: clean

#--- Make rules ---
lib$(OPENCL).a: $(OPENCL_OBJS)
	ar rcs lib$(M5OP).a $(M5OP).o
	ar rcs lib$(OPENCL).a $(OPENCL_OBJS)

%.o: %.cc
	$(CPP) $(CXXFLAGS) -I./ -I$(OPENCLHOME)/include  -c $< -o $@

clean:
	rm -f *.o
	rm -f lib$(OPENCL).a lib$(M5OP).a
