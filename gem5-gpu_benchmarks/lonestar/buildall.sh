#!/bin/bash

extraparams=""
buildparams=""
if [ ! -z "$1" ]
then
	extraparams="$1"
	buildparams="gem5-fusion"
fi

function savebin {
	if [ "$2" == "" ]
	then
		backupbench="$1"
	else
		backupbench="gem5_fusion_$1"
	fi
	if [ -e "$backupbench" ]
	then
		echo "Saving bin $backupbench"
		mv $backupbench $backupbench.bak
	fi
}

function restorebin {
	if [ "$2" == "" ]
	then
		backupbench="$1"
	else
		backupbench="gem5_fusion_$1"
	fi
	if [ -e "$backupbench.bak" ]
	then
		echo "Restoring bin $backupbench"
		mv $backupbench.bak $backupbench
	fi
}

for bench in bfs bh dmr mst pta sp sssp
do
	echo $bench
	pushd . >& /dev/null
	cd $bench
	make clean; make clean-gem5-fusion
	make $buildparams $extraparams
	if [ "$bench" == "bfs" ]
	then
		savebin "bfs_ls" $buildparams
		make clean; make clean-gem5-fusion
		make $buildparams VARIANT=ATOMIC $extraparams
		make clean; make clean-gem5-fusion
		make $buildparams VARIANT=WLA $extraparams
		make clean; make clean-gem5-fusion
		make $buildparams VARIANT=WLC $extraparams
		make clean; make clean-gem5-fusion
		make $buildparams VARIANT=WLC_GB $extraparams
		make clean; make clean-gem5-fusion
		make $buildparams VARIANT=WLW $extraparams
		restorebin "bfs_ls" $buildparams
	elif [ "$bench" == "sssp" ]
	then
		savebin "sssp_ls" $buildparams
		make clean; make clean-gem5-fusion
		make $buildparams VARIANT=WLC $extraparams
		make clean; make clean-gem5-fusion
		make $buildparams VARIANT=WLN $extraparams
		restorebin "sssp_ls" $buildparams
	fi
	popd >& /dev/null
done
