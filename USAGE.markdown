# Building instructions

<span style="color:blue">*Aug-12 2019: This is a beta version of the instructions, I'll be posting further updates in the coming weeks and months to the code base and the instructions here. Future updates will include bug fixes, example workloads, automation scripts, config files, Android images for the full-system mode. For now if you've questions email me at ayoubg@ece.ubc.ca*</span>

*Note: the older version (no longer supported) of this project which is used in the first use-case of the [Emerald](https://dl.acm.org/citation.cfm?id=3322221) paper is available [here](https://github.com/ayoubg/gem5-graphics_v1). The older version features a less detailed graphics model for the GPU and uses the Ruby memory model instead of gem5's classic model.*


## Prerequisites
Please check the prerequisites for gem5, gpgpusim and Mesa3D.
* gem5: http://gem5.org/Dependencies;
* gpgpusim: https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/master/README;
* Mesa 3D: http://www.mesa3d.org/install.html;
* also, you will need imagemagick.

Under Ubuntu to install gem5, gpgpusim and Mesa 3D dependencies and imagemagick you may use the following command to install most (if not all) dependencies, note that you still need to install CUDA and use its path as described later. 

```
apt-get install git g++ python build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev scons swig m4 autoconf automake libtool curl make g++ unzip python-pydot flex bison xutils libx11-dev libxt-dev libxmu-dev libxi-dev libgl1-mesa-dev python-dev imagemagick libpng-dev
```

    
## Building Emerald
1. `$ mkdir emerald` 
2. `$ cd emerald` 

now start with apitrace 

3. `$ git clone https://github.com/gem5-graphics/apitrace` 
4. `$ cd apitrace` 
5. `$ mkdir build` 
6. `$ cd build` 
7. `$ cmake ..` 
8. `$ make -j4` 
9. `$ cd ../..` 

clone emerald:

10. `$ git clone https://github.com/gem5-graphics/gem5-graphics.git` 
11. `$ cd gem5-graphics` 


Update your **setEnvironment**, namely set your **CUDAHOME**, **NVIDIA_CUDA_SDK_LOCATION**, **APITRACE_LIB_PATH** (you can ignore M5_PATH for now). 

Your **APITRACE_LIB_PATH** is the path to `apitrace/build/retraces/libglretrace.so`.

now source your env

12. `$source setEnvironment`

Now build mesa in OGL mode

13. `$ cd mesa`
14. `$./autogen.sh --enable-gallium-swrast --with-gallium-drivers=swrast --disable-gallium--llvm --disable-dri --disable-gbm --disable-egl` 
15. `$ make -j4`

copying libGL (related to some building bug to be fixed)

16. `$ cp lib/gallium/libGL.so lib/gallium/libswrast_dri.so`

Build gem5 with ARM:

17. `$ cd ../gem5`
Choose one of gem5 builds (most likely you want debug or opt)
18. `$ scons build/ARM/gem5.{debug, opt, fast,...}  EXTRAS=../gem5-gpu/src:../gpgpu-sim -j4`



Test your build, try to render a cube trace (download  it from [here]((https://drive.google.com/open?id=1q1vdk1beR-4l3oU7VTJAHU3S2dCWHUeJ)):
`$ build/ARM/gem5.debug ../gem5-gpu/configs/graphics_standalone.py --gtrace={PATH TO textured_cube.trace} --g_start_frame=2 --g_end_frame=2`

Now check that Emerald produced an image (note some pixels may appear missing because they are still in the cache as the image is read from DRAM memory, if you want, you can remove caches in gem5-gpu/configs/GPUConfig.py):
`$ xdg-open m5out/gpgpusimFrameDumps/gpgpusimBuffer_post_frame2_drawcall0_1024x768_-1.-1.bgra.jpg`


### Common issues
* libEGL warning "DRI2: failed to open swrast…"
  * Solution: install mesa dri package (libgl1-mesa-dri)
* GCC version:
  * Try using GCC 5.5.
* protobuf errors:
  * Try to install it from source (https://github.com/protocolbuffers/protobuf)
