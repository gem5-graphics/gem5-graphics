#!/bin/bash

for bench in bfs bh dmr mst pta sp sssp
do
	echo $bench
	pushd . >& /dev/null
	cd $bench
	make clean; make clean-gem5-fusion
	popd >& /dev/null
done
