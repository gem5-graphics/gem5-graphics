# Makefile for GNU make

.PHONY: all gs_md2bump gs_md2shadowvol gs_md2shadow .DEFAULT 

all: gs_md2bump gs_md2shadowvol gs_md2shadow 

gs_md2bump:
	$(MAKE) -f Makefile.gs_md2bump gs_md2bump

gs_md2shadowvol:
	$(MAKE) -f Makefile.gs_md2shadowvol gs_md2shadowvol

gs_md2shadow:
	$(MAKE) -f Makefile.gs_md2shadow gs_md2shadow

.DEFAULT:
	$(MAKE) -f Makefile.gs_md2bump $@
	$(MAKE) -f Makefile.gs_md2shadowvol $@
	$(MAKE) -f Makefile.gs_md2shadow $@

