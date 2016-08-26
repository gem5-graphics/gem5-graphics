#!/bin/bash

for i in cmem diverge global icache1 icache2 icache3 icache4 shared sync texture2 texture4; do cd $i; make gem5-fusion; cd ..; done
