#ifndef __MD2FILE_H__
#define __MD2FILE_H__

/* Copyright NVIDIA Corporation, 2000. */

/* Structures and constants for parsing Quake 2 ".md2" files. */

#define MD2_MAX_TRIANGLES		4096
#define MD2_MAX_VERTICES		2048
#define MD2_MAX_TEXCOORDS		2048
#define MD2_MAX_FRAMES			512
#define MD2_MAX_SKINS			32
#define MD2_MAX_FRAMESIZE		(MD2_MAX_VERTICES * 4 + 128)

typedef struct { 
  int magic; 
  int version; 
  int skinWidth; 
  int skinHeight; 
  int frameSize; 
  int numSkins; 
  int numVertices; 
  int numTexCoords; 
  int numTriangles; 
  int numGlCommands; 
  int numFrames; 
  int offsetSkins; 
  int offsetTexCoords; 
  int offsetTriangles; 
  int offsetFrames; 
  int offsetGlCommands; 
  int offsetEnd; 
} Md2Header;  /* File format structure! */

typedef struct {
  unsigned char vertex[3];
  unsigned char lightNormalIndex;
} Md2AliasTriangleVertex;

typedef struct {
  float vertex[3];
  float normal[3];
} Md2TriangleVertex;

typedef struct {
  short vertexIndices[3];
  short textureIndices[3];
} Md2Triangle;

typedef struct {
  short s, t;
} Md2TextureCoordinate;

typedef struct {
  float scale[3];
  float translate[3];
  char name[16];
  Md2AliasTriangleVertex alias_vertices[1];
} Md2AliasFrame;

typedef struct {
  char name[16];
  Md2TriangleVertex *vertices;
} Md2Frame;

typedef struct {
  float s, t;
  int vertexIndex;
} Md2CommandVertex;

#define NUMVERTEXNORMALS 162

#endif /* __MD2FILE_H__ */
