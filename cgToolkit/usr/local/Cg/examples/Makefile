# Makefile for GNU make

# The Direct3D9 examples do not compile with GNU make; use the Visual Studio projects instead.
SUBDIRS := 
SUBDIRS += Tools
SUBDIRS += OpenGL

define SPAWN_MAKE
	$(MAKE) -C $(1) -f Makefile $@

endef

.PHONY: all .DEFAULT

all:
	$(foreach dir,$(SUBDIRS),$(call SPAWN_MAKE,$(dir)))

.DEFAULT:
	$(foreach dir,$(SUBDIRS),$(call SPAWN_MAKE,$(dir)))
