# Makefile for GNU make

.PHONY: all cginfo cgfxcat trace traceCg traceCgGL traceGL traceGLUT .DEFAULT 

UNAME := $(shell uname)

all: cginfo cgfxcat trace traceCg traceCgGL traceGL traceGLUT 

cginfo:
	$(MAKE) -C cginfo -f Makefile.cginfo cginfo

cgfxcat:
	$(MAKE) -C cgfxcat -f Makefile.cgfxcat cgfxcat

trace:
	$(MAKE) -C trace -f Makefile.trace trace

traceCg:
	$(MAKE) -C trace -f Makefile.traceCg traceCg

ifneq ($(UNAME), Darwin)
traceCgGL:
	$(MAKE) -C trace -f Makefile.traceCgGL traceCgGL
endif

ifneq ($(UNAME), Darwin)
traceGL:
	$(MAKE) -C trace -f Makefile.traceGL traceGL
endif

traceGLUT:
	$(MAKE) -C trace -f Makefile.traceGLUT traceGLUT

.DEFAULT:
	$(MAKE) -C cginfo -f Makefile.cginfo $@
	$(MAKE) -C cgfxcat -f Makefile.cgfxcat $@
	$(MAKE) -C trace -f Makefile.trace $@
	$(MAKE) -C trace -f Makefile.traceCg $@
ifneq ($(UNAME), Darwin)
	$(MAKE) -C trace -f Makefile.traceCgGL $@
endif
ifneq ($(UNAME), Darwin)
	$(MAKE) -C trace -f Makefile.traceGL $@
endif
	$(MAKE) -C trace -f Makefile.traceGLUT $@

