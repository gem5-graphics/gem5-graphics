# Makefile for GNU make

.PHONY: all generic_attribs .DEFAULT 

all: generic_attribs 

generic_attribs:
	$(MAKE) -f Makefile.generic_attribs generic_attribs

.DEFAULT:
	$(MAKE) -f Makefile.generic_attribs $@

