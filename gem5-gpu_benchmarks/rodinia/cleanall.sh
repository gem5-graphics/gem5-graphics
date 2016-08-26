#!/bin/bash

for i in backprop bfs btree cell cfd dwt gaussian heartwall hotspot kmeans lavaMD leukocyte lud mummergpu myocyte nn nw particlefilter pathfinder srad streamcluster
do
	cd $i
	make clean-gem5-fusion
	make clean
	cd ..
done

cd particlefilter
make clean-gem5-fusion BUILD=float
make clean BUILD=float
cd ..
