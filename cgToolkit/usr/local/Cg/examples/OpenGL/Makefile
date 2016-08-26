# Makefile for GNU make

SUBDIRS := 
SUBDIRS += glew
SUBDIRS += basic
SUBDIRS += advanced

define SPAWN_MAKE
	$(MAKE) -C $(1) -f Makefile $@

endef

.PHONY: all .DEFAULT

all:
	$(foreach dir,$(SUBDIRS),$(call SPAWN_MAKE,$(dir)))

.DEFAULT:
	$(foreach dir,$(SUBDIRS),$(call SPAWN_MAKE,$(dir)))
