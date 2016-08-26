# Makefile for GNU make

.PHONY: all cginfo .DEFAULT 

UNAME := $(shell uname)

all: cginfo 

cginfo:
	$(MAKE) -f Makefile.cginfo cginfo

.DEFAULT:
	$(MAKE) -f Makefile.cginfo $@

