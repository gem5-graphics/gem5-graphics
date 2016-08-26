# Makefile for GNU make

.PHONY: all cgfxcat .DEFAULT 

UNAME := $(shell uname)

all: cgfxcat 

cgfxcat:
	$(MAKE) -f Makefile.cgfxcat cgfxcat

.DEFAULT:
	$(MAKE) -f Makefile.cgfxcat $@

