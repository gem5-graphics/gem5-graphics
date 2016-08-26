* To build the system you will need: CUDA 3.1 or higher (tested under 3.1). Once installed please set the CUDAHOME env variable in setMesa_GPGPU.
* Please check the prerequisites for gem5, gpgpusim and Mesa3D: 
..* http://gem5.org/Dependencies; 
..* https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/README; 
..* and http://www.mesa3d.org/install.html.

* To build the simulator:
..1. Uncomment and set CUDAHOME in setMesa_GPGPU.
..2. Source setMesa_GPGPU.
..3. Compile Mesa3D: go to ./MesaMesa-7.11.2_GPGPU-Sim and run: make linux-x86-64, or, make linux-x86-64-debug.
..4. Go to ./shader_to_ptx/arb_to_ptx and run make to build the arbToPtx binary.
..5. Now to build gem5 run the following command in ./gem5: 
scons build/ARM_VI_hammer_GPU/gem5.debug --default=ARM EXTRAS=../gem5-gpu/src:../gpgpu-sim/ PROTOCOL=VI_hammer GPGPU_SIM=True -j8

..6. I included a gem5 checkpoint that you can use to test the build. Run the following command in under ./android_test_cp:
../../../gem5-gpu/configs/soc_arm.py -b android --kernel=vmlinux.smp.mouse.arm --frame-capture -r 1 --restore-with-cpu=timing --cpu-type=timing --max-checkpoints=0 --kernel_stats --g_start_frame=455 --g_end_frame=455 --num-dirs=1 --total-mem-size=2112MB --g_skip_cp_frames=1 --g_depth_shader=1

...the command should load the Android checkpoint and render a frame from the Android desktop. To check output go to m5out/gpgpusimFrameDumps for rendering results, and check stats.txt for gem5 performance data. Modify the run command or/and config files to run gem5 under different modes/configurations.
