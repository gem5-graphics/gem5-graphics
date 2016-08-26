# Makefile for GNU make

.PHONY: all cgfx_bumpdemo cgfx_bumpdemo2 .DEFAULT 

all: cgfx_bumpdemo cgfx_bumpdemo2 

cgfx_bumpdemo:
	$(MAKE) -f Makefile.cgfx_bumpdemo cgfx_bumpdemo

cgfx_bumpdemo2:
	$(MAKE) -f Makefile.cgfx_bumpdemo2 cgfx_bumpdemo2

.DEFAULT:
	$(MAKE) -f Makefile.cgfx_bumpdemo $@
	$(MAKE) -f Makefile.cgfx_bumpdemo2 $@

