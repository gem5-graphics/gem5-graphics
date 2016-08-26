
MD2 GEOMETRY PROGRAM EXAMPLES

This set of three examples demonstrates rendering a Quake2 MD2 model
using geometry programs for shadow volume extrusion and establishing a
per-triangle surface-local coordinate system for bump mapping.

Prior to geometry programs, shadow volume extrusion and establishing a
per-triangle surface-local coordinate system for bump mapping typically
required per-primitive computations performed on the CPU.  This typically
meant operations such as keyframe interpolation or mesh skinning had to be
performed on the CPU.  In these examples, the CPU has no responsibility
for per-frame triangle processing; all such processing is performed by
Cg geometry programs within the GPU.

md2bump

  This Cg 2.0 example extends the OpenGL/basic/16_keyframe_interpolation
  keyframe interpolation example by adding a geometry shader that
  computes a reasonable surface local basis for each triangle based
  on the model's skin texture coordinates.  The geometry program
  then transforms the object-space view and light vectors, which are
  computed by the vertex program, to the primitive's surface local space.
  Then the fragment program access a normal map texture and computes a
  reasonable bump-mapped lighting model accounting for ambient, diffuse,
  and specular contributions using textured normals, texture decal,
  and textured glossiness.

md2shadowvol

  This Cg 2.0 example visualizes shadow volume extrusion for Zfail shadow
  volumes using a geometry program to generate the shadow volume from
  a triangle mesh with adjacency.

md2shadow

  This CgFX 2.0 demo combines the prior demos to render an animated MD2
  model with stenciled shadow volumes for shadowing and bump-mapping
  for per-pixel lighting.

Controls:

  In each demo, use the middle mouse button to rotate the light and the
  left button to rotate the eye's view.  The right mouse button provides
  a pop-up menu.  The space bar toggles the model's keyframe animation.
  The 'w' key toggles wireframe rendering.

Credit:

  The "Perelith Knight" model was designed by James Green.

