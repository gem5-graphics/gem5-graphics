# Makefile for GNU make

.PHONY: all custom_state_assignments .DEFAULT 

all: custom_state_assignments 

custom_state_assignments:
	$(MAKE) -f Makefile.custom_state_assignments custom_state_assignments

.DEFAULT:
	$(MAKE) -f Makefile.custom_state_assignments $@

