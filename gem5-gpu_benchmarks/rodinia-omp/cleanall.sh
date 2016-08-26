#!/bin/bash

for i in backprop bfs cfd heartwall hotspot kmeans lavaMD lud nn nw particlefilter pathfinder srad streamcluster
do
	if [ -d $i ]
	then
		cd $i
		make clean
		cd ..
	fi
done
