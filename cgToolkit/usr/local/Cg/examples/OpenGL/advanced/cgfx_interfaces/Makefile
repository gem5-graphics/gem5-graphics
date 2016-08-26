# Makefile for GNU make

.PHONY: all cgfx_interfaces .DEFAULT 

all: cgfx_interfaces 

cgfx_interfaces:
	$(MAKE) -f Makefile.cgfx_interfaces cgfx_interfaces

.DEFAULT:
	$(MAKE) -f Makefile.cgfx_interfaces $@

