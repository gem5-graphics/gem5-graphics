include $(GPGPUSIM_MESA_ROOT)/Mesa-7.11.2_GPGPU-Sim/configs/linux_defines

MESA_DEFS =   $(DEFINES)
CC 	  =   g++-4.4
CCFLAGS   =   -c -fPIC -std=c++0x ${MESA_DEFS}
ODFLAGS   =   -O3 -g 
LIB_PATHS =   -L$(GPGPUSIM_ROOT)/lib/$(GPGPUSIM_CONFIG) -lcudart 
INC_PATHS =   -I$(GPGPUSIM_MESA_ROOT)/Mesa-7.11.2_GPGPU-Sim/src/mesa 
INC_PATHS +=  -I$(GPGPUSIM_MESA_ROOT)/Mesa-7.11.2_GPGPU-Sim/src/mapi 
INC_PATHS +=  -I$(GPGPUSIM_MESA_ROOT)/Mesa-7.11.2_GPGPU-Sim/src/mesa/swrast
INC_PATHS +=  -I$(CUDA_INSTALL_PATH)/include
INC_PATHS +=  -I$(GPGPUSIM_ROOT)/src

all: library

debug: ODFLAGS = -g
debug: library 

library : libmesagpgpusim.so 

libmesagpgpusim.so: libmesagpgpusim.so.1
	ln -sf libmesagpgpusim.so.1 libmesagpgpusim.so

libmesagpgpusim.so.1: libmesagpgpusim.so.1.0.1
	ln -sf libmesagpgpusim.so.1.0.1 libmesagpgpusim.so.1

libmesagpgpusim.so.1.0.1: mesa_gpgpusim.o gpgpusim_calls.o mesa_calls.o 
	${CC} ${ODFLAGS} ${LIB_PATHS} -shared -Wl,-soname,libmesagpgpusim.so.1 -o libmesagpgpusim.so.1.0.1  mesa_gpgpusim.o gpgpusim_calls.o mesa_calls.o

mesa_calls.o: mesa_calls.cpp mesa_gpgpusim.h
	${CC} ${ODFLAGS} ${CCFLAGS} ${INC_PATHS} mesa_calls.cpp -o mesa_calls.o

gpgpusim_calls.o: gpgpusim_calls.cpp mesa_gpgpusim.h
	${CC} ${ODFLAGS} ${CCFLAGS} ${INC_PATHS} gpgpusim_calls.cpp -o gpgpusim_calls.o

mesa_gpgpusim.o: mesa_gpgpusim.cpp mesa_gpgpusim.h gpgpusim_calls.cpp cuda_calls.h
	${CC} ${ODFLAGS} ${CCFLAGS} ${INC_PATHS} mesa_gpgpusim.cpp -o mesa_gpgpusim.o 
clean:
	rm -rf *o libmesagpgpusim.so*
