# Makefile for GNU make

.PHONY: all include_string .DEFAULT 

all: include_string 

include_string:
	$(MAKE) -f Makefile.include_string include_string

.DEFAULT:
	$(MAKE) -f Makefile.include_string $@

