/*
 * Mesa 3-D graphics library
 * Version:  7.6
 *
 * Copyright (C) 2009  VMware, Inc.  All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * BRIAN PAUL BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#ifndef META_H
#define META_H

/** Return offset in bytes of the field within a vertex struct */
#define OFFSET(FIELD) ((void *) offsetof(struct vertex, FIELD))


/**
 * Flags passed to _mesa_meta_begin().
 */
/*@{*/
#define META_ALL              ~0x0
#define META_ALPHA_TEST        0x1
#define META_BLEND             0x2  /**< includes logicop */
#define META_COLOR_MASK        0x4
#define META_DEPTH_TEST        0x8
#define META_FOG              0x10
#define META_PIXEL_STORE      0x20
#define META_PIXEL_TRANSFER   0x40
#define META_RASTERIZATION    0x80
#define META_SCISSOR         0x100
#define META_SHADER          0x200
#define META_STENCIL_TEST    0x400
#define META_TRANSFORM       0x800 /**< modelview, projection, clip planes */
#define META_TEXTURE        0x1000
#define META_VERTEX         0x2000
#define META_VIEWPORT       0x4000
#define META_CLAMP_FRAGMENT_COLOR 0x8000
#define META_CLAMP_VERTEX_COLOR 0x10000
#define META_CONDITIONAL_RENDER 0x20000
/*@}*/


/**
 * State which we may save/restore across meta ops.
 * XXX this may be incomplete...
 */
struct save_state
{
   GLbitfield SavedState;  /**< bitmask of META_* flags */

   /** META_ALPHA_TEST */
   GLboolean AlphaEnabled;
   GLenum AlphaFunc;
   GLclampf AlphaRef;

   /** META_BLEND */
   GLbitfield BlendEnabled;
   GLboolean ColorLogicOpEnabled;

   /** META_COLOR_MASK */
   GLubyte ColorMask[MAX_DRAW_BUFFERS][4];

   /** META_DEPTH_TEST */
   struct gl_depthbuffer_attrib Depth;

   /** META_FOG */
   GLboolean Fog;

   /** META_PIXEL_STORE */
   struct gl_pixelstore_attrib Pack, Unpack;

   /** META_PIXEL_TRANSFER */
   GLfloat RedBias, RedScale;
   GLfloat GreenBias, GreenScale;
   GLfloat BlueBias, BlueScale;
   GLfloat AlphaBias, AlphaScale;
   GLfloat DepthBias, DepthScale;
   GLboolean MapColorFlag;

   /** META_RASTERIZATION */
   GLenum FrontPolygonMode, BackPolygonMode;
   GLboolean PolygonOffset;
   GLboolean PolygonSmooth;
   GLboolean PolygonStipple;
   GLboolean PolygonCull;

   /** META_SCISSOR */
   struct gl_scissor_attrib Scissor;

   /** META_SHADER */
   GLboolean VertexProgramEnabled;
   struct gl_vertex_program *VertexProgram;
   GLboolean FragmentProgramEnabled;
   struct gl_fragment_program *FragmentProgram;
   struct gl_shader_program *VertexShader;
   struct gl_shader_program *GeometryShader;
   struct gl_shader_program *FragmentShader;
   struct gl_shader_program *ActiveShader;

   /** META_STENCIL_TEST */
   struct gl_stencil_attrib Stencil;

   /** META_TRANSFORM */
   GLenum MatrixMode;
   GLfloat ModelviewMatrix[16];
   GLfloat ProjectionMatrix[16];
   GLfloat TextureMatrix[16];
   GLbitfield ClipPlanesEnabled;

   /** META_TEXTURE */
   GLuint ActiveUnit;
   GLuint ClientActiveUnit;
   /** for unit[0] only */
   struct gl_texture_object *CurrentTexture[NUM_TEXTURE_TARGETS];
   /** mask of TEXTURE_2D_BIT, etc */
   GLbitfield TexEnabled[MAX_TEXTURE_UNITS];
   GLbitfield TexGenEnabled[MAX_TEXTURE_UNITS];
   GLuint EnvMode;  /* unit[0] only */

   /** META_VERTEX */
   struct gl_array_object *ArrayObj;
   struct gl_buffer_object *ArrayBufferObj;

   /** META_VIEWPORT */
   GLint ViewportX, ViewportY, ViewportW, ViewportH;
   GLclampd DepthNear, DepthFar;

   /** META_CLAMP_FRAGMENT_COLOR */
   GLenum ClampFragmentColor;

   /** META_CLAMP_VERTEX_COLOR */
   GLenum ClampVertexColor;

   /** META_CONDITIONAL_RENDER */
   struct gl_query_object *CondRenderQuery;
   GLenum CondRenderMode;

   /** Miscellaneous (always disabled) */
   GLboolean Lighting;
};


/**
 * Temporary texture used for glBlitFramebuffer, glDrawPixels, etc.
 * This is currently shared by all the meta ops.  But we could create a
 * separate one for each of glDrawPixel, glBlitFramebuffer, glCopyPixels, etc.
 */
struct temp_texture
{
   GLuint TexObj;
   GLenum Target;         /**< GL_TEXTURE_2D or GL_TEXTURE_RECTANGLE */
   GLsizei MinSize;       /**< Min texture size to allocate */
   GLsizei MaxSize;       /**< Max possible texture size */
   GLboolean NPOT;        /**< Non-power of two size OK? */
   GLsizei Width, Height; /**< Current texture size */
   GLenum IntFormat;
   GLfloat Sright, Ttop;  /**< right, top texcoords */
};


/**
 * State for glBlitFramebufer()
 */
struct blit_state
{
   GLuint ArrayObj;
   GLuint VBO;
   GLuint DepthFP;
};


/**
 * State for glClear()
 */
struct clear_state
{
   GLuint ArrayObj;
   GLuint VBO;
};


/**
 * State for glCopyPixels()
 */
struct copypix_state
{
   GLuint ArrayObj;
   GLuint VBO;
};


/**
 * State for glDrawPixels()
 */
struct drawpix_state
{
   GLuint ArrayObj;

   GLuint StencilFP;  /**< Fragment program for drawing stencil images */
   GLuint DepthFP;  /**< Fragment program for drawing depth images */
};


/**
 * State for glBitmap()
 */
struct bitmap_state
{
   GLuint ArrayObj;
   GLuint VBO;
   struct temp_texture Tex;  /**< separate texture from other meta ops */
};


/**
 * State for _mesa_meta_generate_mipmap()
 */
struct gen_mipmap_state
{
   GLuint ArrayObj;
   GLuint VBO;
   GLuint FBO;
};

#define MAX_META_OPS_DEPTH      2
/**
 * All per-context meta state.
 */
struct gl_meta_state
{
   /** Stack of state saved during meta-ops */
   struct save_state Save[MAX_META_OPS_DEPTH];
   /** Save stack depth */
   GLuint SaveStackDepth;

   struct temp_texture TempTex;

   struct blit_state Blit;    /**< For _mesa_meta_BlitFramebuffer() */
   struct clear_state Clear;  /**< For _mesa_meta_Clear() */
   struct copypix_state CopyPix;  /**< For _mesa_meta_CopyPixels() */
   struct drawpix_state DrawPix;  /**< For _mesa_meta_DrawPixels() */
   struct bitmap_state Bitmap;    /**< For _mesa_meta_Bitmap() */
   struct gen_mipmap_state Mipmap;    /**< For _mesa_meta_GenerateMipmap() */
};


extern void
_mesa_meta_init(struct gl_context *ctx);

extern void
_mesa_meta_free(struct gl_context *ctx);

extern void
_mesa_meta_BlitFramebuffer(struct gl_context *ctx,
                           GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
                           GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1,
                           GLbitfield mask, GLenum filter);

extern void
_mesa_meta_Clear(struct gl_context *ctx, GLbitfield buffers);

extern void
_mesa_meta_CopyPixels(struct gl_context *ctx, GLint srcx, GLint srcy,
                      GLsizei width, GLsizei height,
                      GLint dstx, GLint dsty, GLenum type);

extern void
_mesa_meta_DrawPixels(struct gl_context *ctx,
                      GLint x, GLint y, GLsizei width, GLsizei height,
                      GLenum format, GLenum type,
                      const struct gl_pixelstore_attrib *unpack,
                      const GLvoid *pixels);

extern void
_mesa_meta_Bitmap(struct gl_context *ctx,
                  GLint x, GLint y, GLsizei width, GLsizei height,
                  const struct gl_pixelstore_attrib *unpack,
                  const GLubyte *bitmap);

extern GLboolean
_mesa_meta_check_generate_mipmap_fallback(struct gl_context *ctx, GLenum target,
                                          struct gl_texture_object *texObj);

extern void
_mesa_meta_GenerateMipmap(struct gl_context *ctx, GLenum target,
                          struct gl_texture_object *texObj);

extern void
_mesa_meta_CopyTexImage1D(struct gl_context *ctx, GLenum target, GLint level,
                          GLenum internalFormat, GLint x, GLint y,
                          GLsizei width, GLint border);

extern void
_mesa_meta_CopyTexImage2D(struct gl_context *ctx, GLenum target, GLint level,
                          GLenum internalFormat, GLint x, GLint y,
                          GLsizei width, GLsizei height, GLint border);

extern void
_mesa_meta_CopyTexSubImage1D(struct gl_context *ctx, GLenum target, GLint level,
                             GLint xoffset,
                             GLint x, GLint y, GLsizei width);

extern void
_mesa_meta_CopyTexSubImage2D(struct gl_context *ctx, GLenum target, GLint level,
                             GLint xoffset, GLint yoffset,
                             GLint x, GLint y,
                             GLsizei width, GLsizei height);

extern void
_mesa_meta_CopyTexSubImage3D(struct gl_context *ctx, GLenum target, GLint level,
                             GLint xoffset, GLint yoffset, GLint zoffset,
                             GLint x, GLint y,
                             GLsizei width, GLsizei height);

extern void
_mesa_meta_CopyColorTable(struct gl_context *ctx,
                          GLenum target, GLenum internalformat,
                          GLint x, GLint y, GLsizei width);

extern void
_mesa_meta_CopyColorSubTable(struct gl_context *ctx,GLenum target, GLsizei start,
                             GLint x, GLint y, GLsizei width);

extern void
_mesa_meta_CopyConvolutionFilter1D(struct gl_context *ctx, GLenum target,
                                   GLenum internalFormat,
                                   GLint x, GLint y, GLsizei width);

extern void
_mesa_meta_CopyConvolutionFilter2D(struct gl_context *ctx, GLenum target,
                                   GLenum internalFormat, GLint x, GLint y,
                                   GLsizei width, GLsizei height);


#endif /* META_H */
