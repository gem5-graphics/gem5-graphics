#ifndef __MD2_H__
#define __MD2_H__

/* Copyright NVIDIA Corporation, 2000. */

/* $Id: //sw/main/apps/OpenGL/mjk/md2shader/md2.h#18 $ */

#include <stdlib.h>

//#include "matrix.h"

#include "md2file.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct {
  short vertexIndex;
  short triangle;
  short edge;
  short prev;
  short next[2];
  int active;
  float maxSqArea;
} Md2Boundary;

typedef struct {
  float p[4];  /* Plane equation the triangle in object space. */
} Md2FrameTrianglePlane;

typedef struct {
  short adjacentTriangle[3];
  /* Bits 0:1 is edge number of adjacent triangle 0 */
  /* Bits 2:3 is edge number of adjacent triangle 1 */
  /* Bits 4:5 is edge number of adjacent triangle 2 */
  unsigned char adjacentTriangleEdges;
  unsigned char openEdgeMask;
} Md2TriangleEdgeInfo;

#define SET_ADJACENT_EDGE(x, n, e) (x) |= ((e) << (2*n))
#define ADJACENT_EDGE(x, n)        (((x) >> (2*(n))) & 0x3)

typedef struct {
  Md2Header header;
#if 0
  Md2Skin *skins;
#endif
  Md2TextureCoordinate *texCoords;
  Md2Triangle *triangles;
  Md2Frame *frames;
  Md2TriangleEdgeInfo *edgeInfo;
  Md2FrameTrianglePlane *framePlane;
  char *filename;
#if 0
  int *glCommandBuffer;
#endif
} Md2Model;

extern float md2VertexNormals[NUMVERTEXNORMALS][3];

extern void md2FreeModel(Md2Model *model);
extern Md2Model *md2ReadModel(const char *filename);

extern void md2EliminateTrivialDegenerateTriangles(Md2Model *model);
extern void md2ComputeTriangleEdgeInfo(Md2Model *model);
extern void md2CheckForBogusAdjacency(Md2Model *model);
extern void md2EliminateAdjacentDegenerateTriangles(Md2Model *model);
extern void md2CloseOpenTriangleGroups(Md2Model *model);
extern void md2ComputeFrameTrianglePlanes(Md2Model *model);

#ifdef  __cplusplus
}
#endif

#endif /* __MD2_H__ */
