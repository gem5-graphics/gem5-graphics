# Makefile for GNU make

.PHONY: all interpolation_modifier .DEFAULT 

all: interpolation_modifier 

interpolation_modifier:
	$(MAKE) -f Makefile.interpolation_modifier interpolation_modifier

.DEFAULT:
	$(MAKE) -f Makefile.interpolation_modifier $@

