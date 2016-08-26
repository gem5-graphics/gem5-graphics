#!/bin/bash

for i in backprop bfs cfd gaussian heartwall hotspot kmeans lavaMD leukocyte lud mummergpu nn nw particlefilter pathfinder srad streamcluster; do cd $i; make clean-gem5-fusion; make clean; cd ..; done
