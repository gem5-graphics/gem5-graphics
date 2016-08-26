
This Cg example demonstrates new Cg standard library support for 2D
texture arrays.  Cg 2.0 introduces new sampler2DARRAY and sampler1DARRAY
types accessing texture arrays.

The program requires your GPU to support the OpenGL EXT_texture_array
extension.

A texture array is an indexable collection of one- and two-dimensional
images of identical size and format, arranged in layers.  A texture array
is accessed via a single texture unit using a single coordinate vector.
A single layer is selected, and that layer is then accessed as though
it were a conventional 1D or 2D texture.

Texture arrays are useful to minimize frequent texture binds between
batches of primitives.  For example, a collection of objects, each with
its own skin texture, can be rendered in a single batch using the texture
array index to select the correct skin for each object.

Space bar animates texture array offsets used in rendering scenes.

'+' increments the texture array offset.

'-' decrements the texture array offset.
