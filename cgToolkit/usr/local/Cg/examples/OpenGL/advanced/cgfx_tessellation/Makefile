# Makefile for GNU make

.PHONY: all cgfx_tessellation .DEFAULT 

all: cgfx_tessellation 

cgfx_tessellation:
	$(MAKE) -f Makefile.cgfx_tessellation cgfx_tessellation

.DEFAULT:
	$(MAKE) -f Makefile.cgfx_tessellation $@

