# Makefile for GNU make

.PHONY: all cgfx_procfx .DEFAULT 

all: cgfx_procfx 

cgfx_procfx:
	$(MAKE) -f Makefile.cgfx_procfx cgfx_procfx

.DEFAULT:
	$(MAKE) -f Makefile.cgfx_procfx $@

