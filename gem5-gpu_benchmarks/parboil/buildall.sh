#!/bin/bash

export PARBOIL_ROOT=`pwd`

cd common; make; cd ..

extra=""
if [ ! -z "$1" ]
then
	extra="$1"
fi

for i in bfs cutcp fft histo lbm mri-gridding mri-q sad spmv stencil tpacf
do
	pushd . >& /dev/null
	cd $i
	make gem5-fusion $extra
	popd >& /dev/null
done
