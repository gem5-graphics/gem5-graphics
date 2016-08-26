# Makefile for GNU make

.PHONY: all tess_simple .DEFAULT 

all: tess_simple 

tess_simple:
	$(MAKE) -f Makefile.tess_simple tess_simple

.DEFAULT:
	$(MAKE) -f Makefile.tess_simple $@

