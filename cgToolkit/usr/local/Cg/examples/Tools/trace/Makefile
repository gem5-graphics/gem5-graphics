# Makefile for GNU make

.PHONY: all trace traceCg traceCgGL traceGL traceGLUT .DEFAULT 

UNAME := $(shell uname)

all: trace traceCg traceCgGL traceGL traceGLUT 

trace:
	$(MAKE) -f Makefile.trace trace

traceCg:
	$(MAKE) -f Makefile.traceCg traceCg

ifneq ($(UNAME), Darwin)
traceCgGL:
	$(MAKE) -f Makefile.traceCgGL traceCgGL
endif

ifneq ($(UNAME), Darwin)
traceGL:
	$(MAKE) -f Makefile.traceGL traceGL
endif

traceGLUT:
	$(MAKE) -f Makefile.traceGLUT traceGLUT

.DEFAULT:
	$(MAKE) -f Makefile.trace $@
	$(MAKE) -f Makefile.traceCg $@
ifneq ($(UNAME), Darwin)
	$(MAKE) -f Makefile.traceCgGL $@
endif
ifneq ($(UNAME), Darwin)
	$(MAKE) -f Makefile.traceGL $@
endif
	$(MAKE) -f Makefile.traceGLUT $@

