#!/bin/bash

for i in backprop bfs cfd gaussian heartwall hotspot kmeans lavaMD leukocyte lud mummergpu nn nw particlefilter pathfinder srad streamcluster; do cd $i; make; make gem5-fusion; cd ..; done
cd particlefilter
mv gem5_fusion_particlefilter_naive gem5_fusion_particlefilter_naive.backup
make clean-gem5-fusion; make gem5-fusion BUILD=float;
mv gem5_fusion_particlefilter_naive.backup gem5_fusion_particlefilter_naive
cd ..
