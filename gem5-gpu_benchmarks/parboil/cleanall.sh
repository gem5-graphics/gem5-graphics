#!/bin/bash

cd common; make clean; cd ..

for i in bfs cutcp fft histo lbm mm mri-gridding mri-q sad spmv stencil tpacf; do cd $i; make clean-gem5-fusion; cd ..; done
