
Cg 2.0 demo using CgFX

cgfx_boxfilter - implement a CgFX box filter effect

Use the space bar to cycle through a set of CgFX techniques for
downsampling texture images.  The boxfilter techniques are both
fast and high-quality.  The bilinear version of the boxfilter
techniques exploit conventional bilinear texture sampling to
approximately double the boxfilter throughput compared to a technique
that fetches every sample within the box filter with a separate
texture fetch.  The bilinear version performs a carefully computed
weighted texture fetch to sample 4 pixels in a single texture fetch.

Use your mouse to stretch smaller the demon image.  Notice how the
boxfilter techniques have substantially less aliasing in the small
downsampled image than the bilinear-only techniques.

You should expect that if the image is up-sampled (made larger)
with one of the box filter modes for the image to look blocky.  This
is the behavior of a boxfilter.  The point of the example however
is the sampling quality when images are down-sampled with a boxfilter
compared to bilinear downsampling.

CAVEATS:  The "boxfilter_nv40" and "boxfilter_bilinear_nv40"
techniques in this demo does NOT work correctly on Mac OS X 10.4
(Tiger) on GeForce 7x00 GPUs due to Mac OpenGL driver bugs.  This
problem is fixed by OpenGL driver updates in Mac OS X 10.5 (Leopard).
