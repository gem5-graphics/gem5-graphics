* To build the system you will need: Nvidia CUDA Toolkit 3.1 or higher (tested under 3.1). Once installed please set the CUDAHOME env variable in setMesa_GPGPU.

* Please check the prerequisites for gem5, gpgpusim and Mesa3D: 
   * gem5: http://gem5.org/Dependencies.
   * gpgpusim: https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/README.
   * and Mesa 3D: http://www.mesa3d.org/install.html.
   * also you need: the SDL library: http://www.libsdl.org/download-1.2.php and imagemagick.
 
   Under Ubuntu to install gem5, gpgpusim and Mesa 3D dependencies and imagemagick you may use the following command to install most dependencies:

   apt-get install git g++ python build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev scons swig m4 autoconf automake libtool curl make g++ unzip python-pydot flex bison xutils libx11-dev libxt-dev libxmu-dev libxi-dev libgl1-mesa-dev python-dev imagemagick

* To build the simulator:
   1. Uncomment and set CUDAHOME in setMesa_GPGPU.

   2. Source setMesa_GPGPU.

   3. Compile Mesa3D: go to ./MesaMesa-7.11.2_GPGPU-Sim and run: make linux-x86-64, or, make linux-x86-64-debug.

   4. Go to ./shader_to_ptx/arb_to_ptx and run make to build the arbToPtx binary.

   5. Now to build gem5 run the following command in ./gem5 (NOTE: an update will be pushed soon for the .opt and .fast builds):
   scons build/ARM_VI_hammer_GPU/gem5.debug --default=ARM EXTRAS=../gem5-gpu/src:../gpgpu-sim/ PROTOCOL=VI_hammer GPGPU_SIM=True -j8

   6. You will need to run an Android image that was modified to work with our simulator, you may download it from [here](http://www.ece.ubc.ca/~ayoubg/files/android_images.tar.xz). Also you will need Android emulator libraries, you can download the binary from [here](http://www.ece.ubc.ca/~ayoubg/files/android_libs.tar.gz).
   
   You can use an already made checkpoint,  taken after Android booted, available [here](http://www.ece.ubc.ca/~ayoubg/files/android_test_cp.tar.gz).
   
   Untar android_images.tar.xz, android_libs.tar.gz and android_test_cp.tar.gz and place anroid_images, android_libs and android_test_cp under ./gem5-graphics.
   NOTE: we will add instructions soon to for how to create a system image, the code has been already been posted [here](https://github.com/ayoubg)

   7. Run the following command under ./android_test_cp: ../../../gem5-gpu/configs/soc_arm.py -b android --kernel=vmlinux.smp.mouse.arm --frame-capture -r 1 --restore-with-cpu=timing --cpu-type=timing --max-checkpoints=0 --kernel_stats --g_start_frame=455 --g_end_frame=455 --num-dirs=1 --total-mem-size=2112MB --g_skip_cp_frames=1 --g_depth_shader=1

   this command will load the checkpoint and render a frame of Android desktop. To check output go to m5out/gpgpusimFrameDumps for rendering results, and check stats.txt for gem5 performance data. More documentation will be posted later but you may try to modify the run command or/and config files to run gem5 under different modes/configurations. 
