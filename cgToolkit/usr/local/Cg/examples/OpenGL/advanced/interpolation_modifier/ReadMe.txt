
Cg 2.0 example of interpolation modifiers on semantics of fragment
interpolants

The dot-separated FLAT, NOPERSPECTIVE, and CENTROID semantic suffixes
modify how interpolation operates for the GeForce 8 "gp4fp" fragment
profile.

The semantic TEXCOORD0 provides conventional, perspective-correct
interpolation.

The semantic TEXCOORD0.FLAT provides flat (constant) interpolation using
OpenGL's provoking vertex.  For triangle strips and independent triangles,
this last vertex forming the triangle.

The semantic TEXCOORD0.NOPERSPECTIVE provide interpolation without
perspective correction.

The semantic TEXCOORD0.CENTROID provides centroid interpolation where the
interpolation center of a fragment is the centroid (average) position for
the fragment's covered samples.  This mode only applies when used with
a multisample framebuffer.  While centroid sampling avoids artifacts
where conventional interpolation at the fragment center might lead to
extrapolation because the fragment center is not covered by the primitive,
centroid sampling can result incorrect partial derivatives because the
separation between fragment centers might not necessarily be a unit pixel.
For this reason, avoid using centroid interpolation for parameters used
to access mipmapped textures or with the ddx() and ddy() Cg standard
library functions.

