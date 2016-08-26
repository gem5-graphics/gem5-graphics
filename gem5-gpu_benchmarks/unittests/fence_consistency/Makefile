default:
	make -f Makefile.default

clean:
	make -f Makefile.default clean

gem5-fusion:
	make -f Makefile.gem5-fusion
	mv gem5_fusion_fence_consistency gem5_fusion_fence_consistency_nofence
	make -f Makefile.gem5-fusion clean

	make -f Makefile.gem5-fusion FENCEFLAGS="-DFENCE_TYPE=SYNCTHREADS"
	mv gem5_fusion_fence_consistency gem5_fusion_fence_consistency_barsync
	make -f Makefile.gem5-fusion clean

	make -f Makefile.gem5-fusion FENCEFLAGS="-DFENCE_TYPE=CTAMEMBAR"
	mv gem5_fusion_fence_consistency gem5_fusion_fence_consistency_ctamemfence
	make -f Makefile.gem5-fusion clean

	make -f Makefile.gem5-fusion FENCEFLAGS="-DFENCE_TYPE=GLOBALMEMBAR"
	mv gem5_fusion_fence_consistency gem5_fusion_fence_consistency_glmemfence
	make -f Makefile.gem5-fusion clean

	make -f Makefile.gem5-fusion FENCEFLAGS="-DFENCE_TYPE=SYSTEMMEMBAR"
	mv gem5_fusion_fence_consistency gem5_fusion_fence_consistency_sysmemfence
	make -f Makefile.gem5-fusion clean

clean-gem5-fusion:
	make -f Makefile.gem5-fusion clean
