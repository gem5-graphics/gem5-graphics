
/* md2edge.c - edge sharing for MD2 model triangles */

/* Copyright NVIDIA Corporation, 2000. */

/* $Id: //sw/main/apps/OpenGL/mjk/md2shader/md2edge.c#17 $ */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef __APPLE__
#include <GLUT/glut.h>  // for GLfloat
#else
#include <GL/glut.h>  // for GLfloat
#endif

#include "md2.h"
#include "vecutil.h"

extern const char *myProgramName;

/* sameVertex - determine if two vertices are identical.  To be
   identical, the two vertices must have the same respective
   X,Y,Z values in each and every frame. */
static int
sameVertex(Md2Model *model, int v0, int v1)
{
  int i;

  for (i=0; i<model->header.numFrames; i++) {
    Md2TriangleVertex *v = model->frames[i].vertices;  
    
    if (v[v0].vertex[0] != v[v1].vertex[0] ||
        v[v0].vertex[1] != v[v1].vertex[1] ||
        v[v0].vertex[2] != v[v1].vertex[2]) {
      return 0;
    }
  }
  return 1;
}

static void
joinTriangles(Md2TriangleEdgeInfo *edgeInfo,
              int tri1, unsigned int edge1,
              int tri2, unsigned int edge2)
{
  assert(edge1 < 3);
  assert(tri1 >= 0);
  assert(edge2 < 3);
  assert(tri2 >= 0);

  edgeInfo[tri1].adjacentTriangle[edge1] = tri2;
  edgeInfo[tri1].adjacentTriangleEdges &= ~(0x3 << (2*edge1));
  edgeInfo[tri1].adjacentTriangleEdges |= edge2 << (2*edge1);

  edgeInfo[tri2].adjacentTriangle[edge2] = tri1;
  edgeInfo[tri2].adjacentTriangleEdges &= ~(0x3 << (2*edge2));
  edgeInfo[tri2].adjacentTriangleEdges |= edge1 << (2*edge2);
}

static void
matchWithTriangleSharingEdge(Md2Model *model,
                             int triangle, int edge,
                             int v0, int v1, int otherv)
{
  const int numTriangles = model->header.numTriangles;
  int doubleTri = -1;
  int otherEdge = 0;
  int i;
  
  /* Match shared edges based on vertex numbers (relatively fast). */
  for (i = triangle+1; i < numTriangles; i++) {
    Md2Triangle *t = &model->triangles[i];
    
    if (t->vertexIndices[0] == v0) {
      if (t->vertexIndices[2] == v1) {
        if (model->edgeInfo[i].adjacentTriangle[2] < 0) {
          if (t->vertexIndices[1] == otherv) {
            if (doubleTri < 0) {
              doubleTri = i;
              otherEdge = 2;
            }
          } else {
            joinTriangles(model->edgeInfo, i, 2, triangle, edge);
            return;
          }
        }
      }
    }
    if (t->vertexIndices[1] == v0) {
      if (t->vertexIndices[0] == v1) {
        if (model->edgeInfo[i].adjacentTriangle[0] < 0) {
          if (t->vertexIndices[2] == otherv) {
            if (doubleTri < 0) {
              doubleTri = i;
              otherEdge = 0;
            }
          } else {
            joinTriangles(model->edgeInfo, i, 0, triangle, edge);
            return;
          }
        }
      }
    }
    if (t->vertexIndices[2] == v0) {
      if (t->vertexIndices[1] == v1) {
        if (model->edgeInfo[i].adjacentTriangle[1] < 0) {
          if (t->vertexIndices[0] == otherv) {
            if (doubleTri < 0) {
              doubleTri = i;
              otherEdge = 1;
            }
          } else {
            joinTriangles(model->edgeInfo, i, 1, triangle, edge);
            return;
          }
        }
      }
    }
  }
  
  /* Match shared edges based on vertex XYZ values (slow check). */
  for (i = triangle+1; i < model->header.numTriangles; i++) {
    Md2Triangle *t = &model->triangles[i];
    
    if (sameVertex(model, t->vertexIndices[0], v0)) {
      if (sameVertex(model, t->vertexIndices[2], v1)) {
        if (model->edgeInfo[i].adjacentTriangle[2] < 0) {
          if (t->vertexIndices[0] == otherv) {
            if (doubleTri < 0) {
              doubleTri = i;
              otherEdge = 2;
            }
          } else {
            joinTriangles(model->edgeInfo, i, 2, triangle, edge);
            return;
          }
        }
      }
    }
    if (sameVertex(model, t->vertexIndices[1], v0)) {
      if (sameVertex(model, t->vertexIndices[0], v1)) {
        if (model->edgeInfo[i].adjacentTriangle[0] < 0) {
          if (t->vertexIndices[0] == otherv) {
            if (doubleTri < 0) {
              doubleTri = i;
              otherEdge = 0;
            }
          } else {
            joinTriangles(model->edgeInfo, i, 0, triangle, edge);
            return;
          }
        }
      }
    }
    if (sameVertex(model, t->vertexIndices[2], v0)) {
      if (sameVertex(model, t->vertexIndices[1], v1)) {
        if (model->edgeInfo[i].adjacentTriangle[1] < 0) {
          if (t->vertexIndices[0] == otherv) {
            if (doubleTri < 0) {
              doubleTri = i;
              otherEdge = 1;
            }
          } else {
            joinTriangles(model->edgeInfo, i, 1, triangle, edge);
            return;
          }
        }
      }
    }
  }
  
  /* Only connect a triangle to a triangle with the exact
     same three vertices as a last resort. */
  if (doubleTri >= 0) {
    joinTriangles(model->edgeInfo, doubleTri, otherEdge, triangle, edge);
    return;
  }
}

static float
polygonArea(Md2Model *model, Md2Boundary *boundaryList, int boundaryIndex)
{
  Md2TriangleVertex *v;
  float d01[3], d02[3], prod[3];
  float norm, maxNorm;
  int v0, v1, v2;
  int i;
  
  /* Get the vertices of the triangle along the boundary. */
  v0 = boundaryList[boundaryIndex].vertexIndex;
  v1 = boundaryList[boundaryList[boundaryIndex].next[0]].vertexIndex;
  v2 = boundaryList[boundaryList[boundaryIndex].next[1]].vertexIndex;

  /* Compute the area of the triangle in the first frame. */
  v = model->frames[0].vertices;  
  v3sub(v[v0].vertex, v[v1].vertex, d01);
  v3sub(v[v0].vertex, v[v2].vertex, d02);
  v3cross(d01, d02, prod);
  maxNorm = v3sqlength(prod);
  
  /* For each frame in the model (beyond the first)... */
  for (i=1; i<model->header.numFrames; i++) {

    /* Compute the area of the triangle in the ith frame. */
    v = model->frames[i].vertices;  
    v3sub(v[v0].vertex, v[v1].vertex, d01);
    v3sub(v[v0].vertex, v[v2].vertex, d02);
    v3cross(d01, d02, prod);
    norm = v3sqlength(prod);

    /* Is this the maximum area that we have encountered? */
    if (norm > maxNorm) {
      maxNorm = norm;
    }
  }

  /* Return the maximum triangle area among all frames. */
  return maxNorm;
}

static int
addNewTriangle(Md2Model *model)
{
  int newTriIndex = model->header.numTriangles;

  /* Add another triangle to the model and resize the
     model's triangle-sized lists. */
  model->header.numTriangles++;
  model->triangles = (Md2Triangle*)
    realloc(model->triangles,
      sizeof(Md2Triangle) * model->header.numTriangles);
  model->edgeInfo = (Md2TriangleEdgeInfo*)
    realloc(model->edgeInfo,
      sizeof(Md2TriangleEdgeInfo) * model->header.numTriangles);

  return newTriIndex;
}

static void
fixOpenTriangle(Md2Model *model, Md2Boundary *boundaryList, int boundaryIndex)
{
  Md2Triangle *newTri;
  int newTriIndex;
  int b0 = boundaryIndex;
  int bp = boundaryList[b0].prev;
  int b1 = boundaryList[b0].next[0];
  int b2 = boundaryList[b0].next[1];

  assert(boundaryList[b1].next[0] == b2);
  assert(boundaryList[bp].next[0] == b0);
  assert(boundaryList[bp].next[1] == b1);
  
  newTriIndex = addNewTriangle(model);
  newTri = &model->triangles[newTriIndex];

  /** Initialize the new triangle **/

  /* Assign the new vertices. */
  newTri->vertexIndices[0] = boundaryList[b2].vertexIndex;
  newTri->vertexIndices[1] = boundaryList[b1].vertexIndex;
  newTri->vertexIndices[2] = boundaryList[b0].vertexIndex;

  /* Bogus texture indices are fine since triangle is (hopefully)
     interior to the model. */
  newTri->textureIndices[0] = 0;
  newTri->textureIndices[1] = 0;
  newTri->textureIndices[2] = 0;

  /* Mark edge 2 unconnected */
  model->edgeInfo[newTriIndex].adjacentTriangle[2] = -1;
  model->edgeInfo[newTriIndex].adjacentTriangleEdges = 0x3 << 4;

  /* Make sure edges we are joining are currently unconnected. */
  assert(model->edgeInfo[boundaryList[b1].triangle].adjacentTriangle[boundaryList[b1].edge] == -1);
  assert(model->edgeInfo[boundaryList[b0].triangle].adjacentTriangle[boundaryList[b0].edge] == -1);

  /* Join the triangles with the new triangle. */
  joinTriangles(model->edgeInfo, 
                newTriIndex, 0, 
                boundaryList[b1].triangle, boundaryList[b1].edge);
  joinTriangles(model->edgeInfo, 
                newTriIndex, 1, 
                boundaryList[b0].triangle, boundaryList[b0].edge);

  /** Update the boundary list based on the addition of the new triangle. **/

  boundaryList[b0].triangle = newTriIndex;
  boundaryList[b0].edge = 2;
  boundaryList[b0].next[0] = b2;
  boundaryList[b0].next[1] = boundaryList[b2].next[0];
  boundaryList[b0].maxSqArea = polygonArea(model, boundaryList, b0);

  boundaryList[bp].next[1] = b2;

  boundaryList[b1].active = 0;

  boundaryList[b2].prev = b0;
}

static void
findOpenBoundary(Md2Model *model, int triangle, int edge,
                 int *boundaryVertices, Md2Boundary *boundaryList)
{
  short v0, v;
  int nextEdge;
  int otherTriangle;
  int count;
  int i;

  if (model->edgeInfo[triangle].openEdgeMask & 1<<edge) {
    return;
  }

  count = 0;

  assert(model->edgeInfo[triangle].adjacentTriangle[edge] == -1);

  model->edgeInfo[triangle].openEdgeMask |= 1<<edge;
  
  v0 = model->triangles[triangle].vertexIndices[edge];
  boundaryList[count].vertexIndex = v0;
  boundaryList[count].triangle = triangle;
  boundaryList[count].edge = edge;
  count++;

  nextEdge = (edge+1)%3;
  v = model->triangles[triangle].vertexIndices[nextEdge];
  while (!sameVertex(model, v, v0)) {
    otherTriangle = model->edgeInfo[triangle].adjacentTriangle[nextEdge];
    while (otherTriangle >= 0) {
      for (i=0; i<3; i++) {
        if (model->edgeInfo[otherTriangle].adjacentTriangle[i] == triangle) {
          assert(sameVertex(model,
	           model->triangles[otherTriangle].vertexIndices[(i+1)%3], v));

          triangle = otherTriangle;
      
          nextEdge = (i+1)%3;
          break;
        }
      }
      assert(i<3);
      otherTriangle = model->edgeInfo[triangle].adjacentTriangle[nextEdge];
    }

    /* Mark this edge as processed to avoid reprocessing
       the boundary multiple times. */
    model->edgeInfo[triangle].openEdgeMask |= 1<<nextEdge;

    boundaryList[count].vertexIndex = v;
    boundaryList[count].triangle = triangle;
    boundaryList[count].edge = nextEdge;
    count++;

    nextEdge = (nextEdge+1)%3;
    v = model->triangles[triangle].vertexIndices[nextEdge];
  }
  *boundaryVertices = count;
}

static void
fixOpenBoundary(Md2Model *model,
                int count, Md2Boundary *boundaryList)
{
  int b0, b1, b2;
  int i;

  if (count == 1) {
    /* Ugh, a degenerate triangle with two (or perhaps three) 
       identical vertices tricking us into thinking that there
       is an open edge.  Hopefully these should be eliminated
       by an earlier "eliminate" pass, but such triangles are
       harmless. */
    return;
  }

  assert(count >= 3);

  if (count == 3) {
    /* Often a common case.  Save bookkeeping and close the triangle
       boundary immediately. */
    b0 = 0;
    b1 = 1;
    b2 = 2;
  } else {
    float maxMaxSqArea;
    int numActive;
    int minIndex = 0;
    
    boundaryList[0].prev = count-1;
    boundaryList[0].next[0] = 1;
    boundaryList[0].next[1] = 2;
    boundaryList[0].active = 1;

    for (i=1; i<count-2; i++) {
      boundaryList[i].prev = i-1;
      boundaryList[i].next[0] = i+1;
      boundaryList[i].next[1] = i+2;
      boundaryList[i].active = 1;
    }

    boundaryList[i].prev = i-1;
    boundaryList[i].next[0] = i+1;
    boundaryList[i].next[1] = 0;
    boundaryList[i].active = 1;

    boundaryList[i+1].prev = i;
    boundaryList[i+1].next[0] = 0;
    boundaryList[i+1].next[1] = 1;
    boundaryList[i+1].active = 1;
    
    boundaryList[0].maxSqArea = polygonArea(model, boundaryList, 0);
    maxMaxSqArea = boundaryList[0].maxSqArea;
    
    for (i=1; i<count; i++) {
      boundaryList[i].maxSqArea = polygonArea(model, boundaryList, i);
      if (boundaryList[i].maxSqArea > maxMaxSqArea) {
        maxMaxSqArea = boundaryList[i].maxSqArea;
      }
    }

    /* If triangles are formed from adjacent edges along the
       boundary, at least front-facing such triangle should
       be front-facing (ie, have a non-negative area). */
    /* XXX Would this apply to Mobeous strips and Kline bottles? */
    assert(maxMaxSqArea >= 0.0);

    maxMaxSqArea = 2.0f * maxMaxSqArea;
    
    numActive = count;
    
    while (numActive > 3) {
      float min;

      min = maxMaxSqArea;
      for (i=0; i<count; i++) {
        if (boundaryList[i].active) {
          if (boundaryList[i].maxSqArea < min) {
            if (boundaryList[i].maxSqArea >= 0.0) {
              min = boundaryList[i].maxSqArea;
              minIndex = i;
            }
          }
        }
      }
      assert(min < maxMaxSqArea);
      fixOpenTriangle(model, boundaryList, minIndex);

      /* Newly created triangle formed from adjacent edges
         along the boundary could be larger than the
         previous largest triangle. */
      if (boundaryList[minIndex].maxSqArea > maxMaxSqArea) {
        maxMaxSqArea = 2.0f * boundaryList[minIndex].maxSqArea;
      
      }
      numActive--;
    }

    for (i=0; i<count; i++) {
      if (boundaryList[i].active) {
        minIndex = i;
        break;
      }
    }
    assert(i < count);

    b0 = minIndex;
    b1 = boundaryList[b0].next[0];
    b2 = boundaryList[b0].next[1];

    assert(boundaryList[b0].prev == b2);
    assert(boundaryList[b1].prev == b0);
    assert(boundaryList[b1].next[0] == b2);
    assert(boundaryList[b1].next[1] == b0);
    assert(boundaryList[b2].prev == b1);
    assert(boundaryList[b2].next[0] == b0);
    assert(boundaryList[b2].next[1] == b1);
  }

  /* Place final "keystone" triangle to fill completely
     the open boundary. */
  {
    Md2Triangle *newTri;
    int newTriIndex = model->header.numTriangles;

    model->header.numTriangles++;
    model->triangles = (Md2Triangle*)
      realloc(model->triangles,
        sizeof(Md2Triangle) * model->header.numTriangles);
    model->edgeInfo = (Md2TriangleEdgeInfo*)
      realloc(model->edgeInfo,
        sizeof(Md2TriangleEdgeInfo) * model->header.numTriangles);

    newTri = &model->triangles[newTriIndex];
    newTri->vertexIndices[0] = boundaryList[b2].vertexIndex;
    newTri->vertexIndices[1] = boundaryList[b1].vertexIndex;
    newTri->vertexIndices[2] = boundaryList[b0].vertexIndex;
    /* Bogus texture indices are fine since triangle is (hopefully)
       interior to themodel. */
    newTri->textureIndices[0] = 0;
    newTri->textureIndices[1] = 0;
    newTri->textureIndices[2] = 0;

    /* Join keystone triangle. */
    joinTriangles(model->edgeInfo,
                  newTriIndex, 0,
                  boundaryList[b1].triangle, boundaryList[b1].edge);
    joinTriangles(model->edgeInfo, 
                  newTriIndex, 1,
                  boundaryList[b0].triangle, boundaryList[b0].edge);
    joinTriangles(model->edgeInfo,
                  newTriIndex, 2,
                  boundaryList[b2].triangle, boundaryList[b2].edge);
  }
}

static void
findAndFixOpenTriangleGroups(Md2Model *model, int triangle)
{
  int count;
  Md2Boundary *boundaryList = (Md2Boundary*)
    malloc((1+2*model->header.numTriangles)*sizeof(Md2Boundary));
  if (boundaryList == NULL) {
    return;
  }
  if (model->edgeInfo[triangle].adjacentTriangle[0] < 0) {
    findOpenBoundary(model, triangle, 0, &count, boundaryList);
    fixOpenBoundary(model, count, boundaryList);
  }
  if (model->edgeInfo[triangle].adjacentTriangle[1] < 0) {
    findOpenBoundary(model, triangle, 1, &count, boundaryList);
    fixOpenBoundary(model, count, boundaryList);
  }
  if (model->edgeInfo[triangle].adjacentTriangle[2] < 0) {
    findOpenBoundary(model, triangle, 2, &count, boundaryList);
    fixOpenBoundary(model, count, boundaryList);
  }
  free(boundaryList);
}

/* md2EliminateTrivialDegenerateTriangles - throw away triangles
   in the model that are trivially degenerate.  Triangles are
   trivially degenerate if they have two (ie, the triangle is
   really a line) or three identical vertices (ie, the triangle
   is really a point).

   A non-trivial degenerate triangle has effectively zero area,
   but does not duplicate vertices.  The vertices are in a line,
   but no two vertices are identical.  To avoid floating-point
   accuracy issues (and to avoid a lot of computation), a
   subsequent "eliminate" routine kills non-trivial degenerate
   triangles based on adjacency information.

   This routine is dedicated to the jupiter.md2 model.  What
   an utterly crappy model. */
void
md2EliminateTrivialDegenerateTriangles(Md2Model *model)
{
  int lineCount = 0, pointCount = 0;
  int i;

  /* Work backwards so we can kill degenerate triangles by
     replacing the dengenerate triangle with the last triangle
     in the list.  For this to work, we must make sure that
     the last triangles in the list have already been verified
     not to be degenerate. */
  for (i = model->header.numTriangles-1; i >= 0; i--) {
    Md2Triangle *t = &model->triangles[i];
    int v0 = t->vertexIndices[0];
    int v1 = t->vertexIndices[1];
    int v2 = t->vertexIndices[2];
    int dupCount = 0;

    if (sameVertex(model, v0, v1)) {
      dupCount++;
    }
    if (sameVertex(model, v1, v2)) {
      dupCount++;
    }
    if (sameVertex(model, v2, v0)) {
      dupCount++;
    }
    if (dupCount > 0) {
      assert(dupCount != 2);
      model->triangles[i] = model->triangles[model->header.numTriangles-1];
      model->header.numTriangles--;
      if (dupCount == 3) {
        pointCount++;
      } else {
        lineCount++;
      }
    }
  }
  if (lineCount) {
    printf("%s: %d line-degenerate triangles removed from \"%s\"\n",
      myProgramName,
      lineCount, model->filename);
  }
  if (pointCount) {
    printf("%s: %d point-degenerate triangles removed from \"%s\"\n",
      myProgramName,
      pointCount, model->filename);
  }
}

static void
reconnectSharedEdges(Md2Model *model, int isTri, int wasTri)
{
  int tri;
  int i, j;
  int count;
  
  for (i=0; i<3; i++) {
    tri = model->edgeInfo[wasTri].adjacentTriangle[i];
    
    if (tri >= 0) {
      count = 0;
      for (j=0; j<3; j++) {
        if (model->edgeInfo[tri].adjacentTriangle[j] == wasTri) {
          model->edgeInfo[tri].adjacentTriangle[j] = isTri;
          count++;
        }
        if (model->edgeInfo[tri].adjacentTriangle[j] == isTri) {
          count++;
        }
      }
      assert(count > 0);
    }
  }
}

void
possiblyReconnectTriangle(Md2Model *model, int tri, int isTri, int wasTri)
{
  int j;

  for (j=0; j<3; j++) {
    if (model->edgeInfo[tri].adjacentTriangle[j] == wasTri) {
      model->edgeInfo[tri].adjacentTriangle[j] = isTri;
    }
  }
}

static int
eliminateAdjacentDegeneratePair(Md2Model *model, int badTri, int otherBadTri,
                                int goodTri)
{
  int otherGoodTri = 0;
  int numTris;
  int i, j;

  assert(badTri < model->header.numTriangles);
  assert(otherBadTri < model->header.numTriangles);
  assert(goodTri < model->header.numTriangles);

  /* The other good triangle is the triangle adjacent to the other
     bad triangle but which is not the bad triangle. */
  for (i=0; i<3; i++) {
    if (model->edgeInfo[otherBadTri].adjacentTriangle[i] != badTri) {
      otherGoodTri = model->edgeInfo[otherBadTri].adjacentTriangle[i];
      break;
    }
  }
  assert(i < 3);
  
  /* Fix the good triangle so that both edges adjacent to the
     bad triangle are now adjacent to the other good triangle. */
  for (i=0; i<3; i++) {
    if (model->edgeInfo[goodTri].adjacentTriangle[i] == badTri) {
      model->edgeInfo[goodTri].adjacentTriangle[i] = otherGoodTri;
    }
  }
  
  /* Fix the other good triangle so that both edges adjacent to the
     other bad triangle are now adjacent to the good triangle. */
  for (i=0; i<3; i++) {
    if (model->edgeInfo[otherGoodTri].adjacentTriangle[i] == otherBadTri) {
      model->edgeInfo[otherGoodTri].adjacentTriangle[i] = goodTri;
    }
  }

  /* Decrement the model's triangle count by 2.  Then copy
     non-degenerate triangles from the end of the triangle
     list to the slots once used by the eliminated triangles.
     Be sure to copy the edgeInfo data structure too.  Also
     if goodTri is one of the last two triangles, be careful
     to make sure it gets copied. */

  model->header.numTriangles -= 2;
  numTris = model->header.numTriangles;
  
  if (goodTri < numTris) {
    model->triangles[badTri] = model->triangles[numTris+1];
    model->edgeInfo[badTri]  = model->edgeInfo[numTris+1];    
    model->triangles[otherBadTri] = model->triangles[numTris];
    model->edgeInfo[otherBadTri]  = model->edgeInfo[numTris];   
    reconnectSharedEdges(model, badTri, numTris+1);
    reconnectSharedEdges(model, otherBadTri, numTris);
    /* We are moving two triangles and they each might be
       connected to each other.  Possibly reconnect the
       edges appropriately if so. */
    possiblyReconnectTriangle(model, badTri, otherBadTri, numTris);
    possiblyReconnectTriangle(model, otherBadTri, badTri, numTris+1);
  } else {
    if (goodTri == numTris+1) {
      if (badTri < numTris) {
        model->triangles[badTri] = model->triangles[numTris+1];
        model->edgeInfo[badTri]  = model->edgeInfo[numTris+1];
        model->triangles[otherBadTri] = model->triangles[numTris];
        model->edgeInfo[otherBadTri]  = model->edgeInfo[numTris];  
        reconnectSharedEdges(model, badTri, numTris+1);
        possiblyReconnectTriangle(model, badTri, otherBadTri, numTris);

        if (otherBadTri < numTris) {
          reconnectSharedEdges(model, otherBadTri, numTris);
          possiblyReconnectTriangle(model, otherBadTri, badTri, numTris+1);
        }

        goodTri = badTri;
      } else {
        assert(otherBadTri < numTris);
        model->triangles[otherBadTri] = model->triangles[numTris+1];
        model->edgeInfo[otherBadTri]  = model->edgeInfo[numTris+1];
        model->triangles[badTri] = model->triangles[numTris];
        model->edgeInfo[badTri]  = model->edgeInfo[numTris];           
        reconnectSharedEdges(model, otherBadTri, numTris+1);
        possiblyReconnectTriangle(model, otherBadTri, badTri, numTris);

        if (badTri < numTris) {
          reconnectSharedEdges(model, badTri, numTris);
          possiblyReconnectTriangle(model, badTri, otherBadTri, numTris+1);
        }

        goodTri = otherBadTri;
      }
    } else {
      assert(goodTri == numTris);
      if (badTri < numTris) {
        model->triangles[badTri] = model->triangles[numTris];
        model->edgeInfo[badTri]  = model->edgeInfo[numTris];
        model->triangles[otherBadTri] = model->triangles[numTris+1];
        model->edgeInfo[otherBadTri]  = model->edgeInfo[numTris+1];  
        reconnectSharedEdges(model, badTri, numTris);
        possiblyReconnectTriangle(model, badTri, otherBadTri, numTris+1);

        if (otherBadTri < numTris) {
          reconnectSharedEdges(model, otherBadTri, numTris+1);
          possiblyReconnectTriangle(model, otherBadTri, badTri, numTris);
        }

        goodTri = badTri;
      } else {
        assert(otherBadTri < numTris);
        model->triangles[otherBadTri] = model->triangles[numTris];
        model->edgeInfo[otherBadTri]  = model->edgeInfo[numTris];    
        model->triangles[badTri] = model->triangles[numTris+1];
        model->edgeInfo[badTri]  = model->edgeInfo[numTris+1];  
        reconnectSharedEdges(model, otherBadTri, numTris);
        possiblyReconnectTriangle(model, otherBadTri, badTri, numTris+1);

        if (badTri < numTris) {
          reconnectSharedEdges(model, badTri, numTris+1);
          possiblyReconnectTriangle(model, badTri, otherBadTri, numTris);
        }

        goodTri = otherBadTri;
      }
    }
  }
  
  assert(goodTri < model->header.numTriangles);

  /* Patch up the edge info for the two relocated triangles. */
  for (i=model->header.numTriangles-1; i >= 0; i--) {
    for (j=0; j<3; j++) {
      assert(model->edgeInfo[i].adjacentTriangle[j] <
             model->header.numTriangles);
    }
  }

#ifndef NDEBUG
  for (i=model->header.numTriangles-1; i >= 0; i--) {
    for (j=0; j<3; j++) {
      assert(model->edgeInfo[i].adjacentTriangle[j] <
             model->header.numTriangles);
    }
  }
  for (i=model->header.numTriangles-1; i >= 0; i--) {
    if ((model->edgeInfo[i].adjacentTriangle[0] ==
         model->edgeInfo[i].adjacentTriangle[1]) &&
        (model->edgeInfo[i].adjacentTriangle[1] ==
	 model->edgeInfo[i].adjacentTriangle[2]) &&
        (model->edgeInfo[i].adjacentTriangle[0] != -1)) {
      assert(model->edgeInfo[model->edgeInfo[i].adjacentTriangle[0]].adjacentTriangle[0] == i);
      assert(model->edgeInfo[model->edgeInfo[i].adjacentTriangle[0]].adjacentTriangle[1] == i);
      assert(model->edgeInfo[model->edgeInfo[i].adjacentTriangle[0]].adjacentTriangle[2] == i);
      printf("badness\n");
    }
  }
#endif

  /* Two degenerate triangles eliminated. */
  return 2;
}

static int
findAndFixAdjacentDegeneratePair(Md2Model *model, int tri)
{
  int t0, t1, t2;
  
  t0 = model->edgeInfo[tri].adjacentTriangle[0];
  t1 = model->edgeInfo[tri].adjacentTriangle[1];
  t2 = model->edgeInfo[tri].adjacentTriangle[2];
  
  /* Trivially degnerate triangles should have already been eliminated. */
  assert(t0 != tri);
  assert(t1 != tri);
  assert(t2 != tri);

  if ((t0 == t1) && (t1 == t2)) {
    if (t0 >= 0) {
      assert(model->edgeInfo[t0].adjacentTriangle[0] == tri);
      assert(model->edgeInfo[t0].adjacentTriangle[1] == tri);
      assert(model->edgeInfo[t0].adjacentTriangle[2] == tri);
    }
    return 0;
  }
  
  if (t0 == t1) {
    if (t0 >= 0) {
      return eliminateAdjacentDegeneratePair(model, tri, t0, t2);
    }
  }
  if (t1 == t2) {
    if (t1 >= 0) {
      return eliminateAdjacentDegeneratePair(model, tri, t1, t0);
    }
  }
  if (t2 == t0) {
    if (t2 >= 0) {
      return eliminateAdjacentDegeneratePair(model, tri, t2, t1);
    }
  }
  return 0;
}

void
md2EliminateAdjacentDegenerateTriangles(Md2Model *model)
{
  int count = 0;
  int loopCount;
  int i;

  /* Eliminating two degenerate triangle pairs may
     not be the end of the story if the two "good" triangles
     that get connected are also degenerate.  Loop to
     handle this unlikely event. */
  do {
    loopCount = count;
    for (i = 0; i<model->header.numTriangles; i++) {
      count += findAndFixAdjacentDegeneratePair(model, i);
    }
  } while (count > loopCount);

  if (count) {
    printf("%s: eliminated %d adjacent degenerate triangles from \"%s\"\n",
      myProgramName,
      count, model->filename);
  }
}

void
md2CloseOpenTriangleGroups(Md2Model *model)
{
  int groups = 0;
  int i;
  
  for (i = model->header.numTriangles-1; i >= 0; i--) {
    if (model->edgeInfo[i].adjacentTriangle[0] < 0 ||
      model->edgeInfo[i].adjacentTriangle[1] < 0 ||
      model->edgeInfo[i].adjacentTriangle[2] < 0) {
      findAndFixOpenTriangleGroups(model, i);
      groups++;
    } 
  }
  if (groups > 0) {
    printf("%s: had to close %d open triangle groups in \"%s\" model\n",
      myProgramName,
      groups, model->filename);
  }
}

void
md2ComputeTriangleEdgeInfo(Md2Model *model)
{
  const int numTriangles = model->header.numTriangles;
  int i;

  /* Allocate edge information for all triangles in the model.
     Note that the connectivity is assumed to be the same
     among all frames. */
  model->edgeInfo = (Md2TriangleEdgeInfo *)
    malloc(sizeof(Md2TriangleEdgeInfo) * numTriangles);

  /* Initialize edge information as if all triangles are
     fully disconnected. */
  for (i = 0; i < numTriangles; i++) {
    model->edgeInfo[i].adjacentTriangle[0] = -1;  /* Vertex 0,1 edge */
    model->edgeInfo[i].adjacentTriangle[1] = -1;  /* Vertex 1,2 edge */
    model->edgeInfo[i].adjacentTriangle[2] = -1;  /* Vertex 2,0 edge */

    model->edgeInfo[i].adjacentTriangleEdges = (0x3 << 0) | 
                                               (0x3 << 2) | 
                                               (0x3 << 4);

    model->edgeInfo[i].openEdgeMask = 0;
  }

  for (i = 0; i < numTriangles; i++) {
    Md2Triangle *t = &model->triangles[i];

    if (model->edgeInfo[i].adjacentTriangle[0] < 0) {
      matchWithTriangleSharingEdge(model, i, 0,
        t->vertexIndices[0], t->vertexIndices[1], t->vertexIndices[2]);
    }
    if (model->edgeInfo[i].adjacentTriangle[1] < 0) {
      matchWithTriangleSharingEdge(model, i, 1,
        t->vertexIndices[1], t->vertexIndices[2], t->vertexIndices[0]);
    }
    if (model->edgeInfo[i].adjacentTriangle[2] < 0) {
      matchWithTriangleSharingEdge(model, i, 2,
        t->vertexIndices[2], t->vertexIndices[0], t->vertexIndices[1]);
    }
  }
}

void
md2CheckForBogusAdjacency(Md2Model *model)
{
  const int numTriangles = model->header.numTriangles;
  int i, k;
  unsigned char j;

  for (i = 0; i < numTriangles; i++) {
    for (j=0; j<3; j++) {
      int mutuallyAdjacentCount0, mutuallyAdjacentCount1;
      int adjacentTriangle;
#ifndef NDEBUG  /* Only used by assert macros. */
      unsigned char adjacentTriangleEdges =
        model->edgeInfo[i].adjacentTriangleEdges;
      int adjacentTriangleSharedEdge =
        ADJACENT_EDGE(adjacentTriangleEdges, j);
#endif
      
      adjacentTriangle = model->edgeInfo[i].adjacentTriangle[j];

      if (adjacentTriangle >= 0) {
        
        assert(adjacentTriangleSharedEdge < 3);
        assert(model->edgeInfo[adjacentTriangle].adjacentTriangle[adjacentTriangleSharedEdge] == i);
        assert(ADJACENT_EDGE(model->edgeInfo[adjacentTriangle].adjacentTriangleEdges, adjacentTriangleSharedEdge) == j);
        
        if (adjacentTriangle == i) {
          printf("warning: triangle %d should not be adjacent to itself!\n", i);
        }
        
        mutuallyAdjacentCount0 = 0;
        for (k=0; k<3; k++) {
          if (model->edgeInfo[i].adjacentTriangle[j] ==
	      model->edgeInfo[i].adjacentTriangle[k]) {
            mutuallyAdjacentCount0++;
          }
        }
        
        mutuallyAdjacentCount1 = 0;
        for (k=0; k<3; k++) {
          if (model->edgeInfo[adjacentTriangle].adjacentTriangle[k] == i) {
            mutuallyAdjacentCount1++;
          }
        }
        
        if (mutuallyAdjacentCount0 != mutuallyAdjacentCount1) {
          printf("warning: triangles %d and %d should be "
	    "mutually adjacent but are not!\n",
            i, adjacentTriangle);
        }
      } else {
        assert(adjacentTriangle == -1);
        assert(adjacentTriangleSharedEdge == 3);
      }
    }
  }
}

static void
makePlane(float p[4], 
          const float v0[3], const float v1[3], const float v2[3])
{
  GLfloat vec0[3], vec1[3];

  /* Need 2 vectors to find cross product. */
  vec0[0] = v1[0] - v0[0];
  vec0[1] = v1[1] - v0[1];
  vec0[2] = v1[2] - v0[2];

  vec1[0] = v2[0] - v0[0];
  vec1[1] = v2[1] - v0[1];
  vec1[2] = v2[2] - v0[2];

  /* Use cross product to get A, B, and C of plane equation */
  p[0] =   vec0[1] * vec1[2] - vec0[2] * vec1[1];
  p[1] = -(vec0[0] * vec1[2] - vec0[2] * vec1[0]);
  p[2] =   vec0[0] * vec1[1] - vec0[1] * vec1[0];
  p[3] = -(p[0] * v0[0] + p[1] * v0[1] + p[2] * v0[2]);
}

/* Instead of trying to compute the plane equation for each interpolated
   triangle, we pre-compute the plane equation for every triangle in
   every key frame.  Since the key frame interpolation is simply linear,
   we can interpolate the two plane equations from corresponding key
   frame triangles and reduce our per-frame math.

   md2ComputeFrameTrianglePlanes pre-computes the plane equation for
   every triangle in every key frame. */
void
md2ComputeFrameTrianglePlanes(Md2Model *model)
{
  Md2Frame *frame;
  Md2FrameTrianglePlane *framePlane;
  const int numTriangles = model->header.numTriangles;
  const int numFrames = model->header.numFrames;
  int i, j;
  
  /* Allocate enough space for a plane equation for every
     triangle in every frame of the model. */
  model->framePlane = (Md2FrameTrianglePlane *)
    malloc(numFrames * numTriangles * sizeof(Md2FrameTrianglePlane));

  framePlane = model->framePlane;
  frame = model->frames;

  /* For each frame... */
  for (i=0; i<numFrames; i++) {
    Md2TriangleVertex *vertices = frame->vertices;
    Md2Triangle *t = model->triangles;

    /* For each triangle... */
    for (j=0; j<numTriangles; j++) {
      /* Get the vertex position of each vertex of triangle for the frame. */
      const float *v0 = vertices[t->vertexIndices[0]].vertex;
      const float *v1 = vertices[t->vertexIndices[1]].vertex;
      const float *v2 = vertices[t->vertexIndices[2]].vertex;

      /* Compute and stash the plane equation for the triangle
         in this particular frame. */
      makePlane(framePlane->p, v0, v1, v2);
      framePlane++;
      t++;
    }
    frame++;
  }
}

void md2ComputeAdjacencyInfo(Md2Model *model)
{
  md2EliminateTrivialDegenerateTriangles(model);
  md2ComputeTriangleEdgeInfo(model);
  md2CheckForBogusAdjacency(model);
#if 0
  md2EliminateAdjacentDegenerateTriangles(model);
#endif
  md2CloseOpenTriangleGroups(model);
}
