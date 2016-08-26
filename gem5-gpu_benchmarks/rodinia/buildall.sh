#!/bin/bash

extra=""
if [ ! -z "$1" ]
then
	extra="$1"
fi

for i in backprop bfs btree cell cfd dwt gaussian heartwall hotspot kmeans lavaMD leukocyte lud mummergpu myocyte nn nw particlefilter pathfinder srad streamcluster
do
	cd $i
	make gem5-fusion $extra
	cd ..
done
cd particlefilter
mv gem5_fusion_particlefilter_naive gem5_fusion_particlefilter_naive.backup
make clean-gem5-fusion
make gem5-fusion BUILD=float $extra
mv gem5_fusion_particlefilter_naive.backup gem5_fusion_particlefilter_naive
cd ..
