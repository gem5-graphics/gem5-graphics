
// md2render.cpp - class to help load and render Quake2 MD2 models via OpenGL

// Copyright NVIDIA Corporation, 2006

#ifdef _MSC_VER
/* Suppress Visual C++ 6 warning about 'Some STL template class' :
   identifier was truncated to '255' characters in the debug information */
#pragma warning(disable: 4786)
#endif

#include <assert.h>

#include <map>
using namespace std;

#include "md2.h"
#include "loadtex.h"
#include "md2render.h"

struct TexCoord {
  GLshort s;
  GLshort t;
};

struct VertexNormal {
  GLfloat x, y, z;
  GLfloat nx, ny, nz;
};

struct VertexData {
  GLushort ndx;
  GLushort newndx;
  TexCoord tc;
};

struct ModelRenderInfo {
  struct {
    GLuint texCoordArray;
    GLuint vertexNormalArray;   
    GLuint elementArray;
  } bufferObject;
  struct {
    GLuint decal;
    GLuint normalMap;
    GLuint heightMap;
  } textureObject;

  int arrayIndicesPerFrame;
  int frameCount;
  int arrayElementsPerObject;
  int refcnt;

  ModelRenderInfo(const Md2Model *model, bool withAdjacency = false);
  ~ModelRenderInfo();

  void bindArrayElementsForGL();
  void bindTexCoordsForGL(int texunit);
  void bindPositionsAndNormalsForGL(int frameA, int frameB, int normals);
  void drawModel();
  void drawModelAdj();

  void ref() { refcnt++; }
  void deref() { refcnt--; if (refcnt == 0) { delete this; } }
};

// MD2 files have a separate position/normal index and texture
// coordinate set index per vertex.  This routine re-indexes the
// vertex set so every vertex has a unique index to its 
// position/normal and texture for easy rendering with vertex
// arrays stored in vertex buffer objects.
//
// Once this re-indexing is performed, three buffer objects
// are created: a per-model texture coordinate set buffer object,
// a buffer object with all the position/normal pairs for the
// full sequence of frames, and finally an element arrays for
// all of the model's independent triangles.
ModelRenderInfo::ModelRenderInfo(const Md2Model *model, bool withAdjacency)
{
  typedef pair<int,int> PosTexIndex;
  typedef map<PosTexIndex, VertexData, less<PosTexIndex> > vertexMapType;
  // map old ndx to new ndx, considering just position (not texture coordinates)
  typedef map</*old*/int,/*new*/int> positionMapType;

  vertexMapType vertexMap;
  positionMapType posMap;

  const int numFrames = model->header.numFrames;
  const int numTriangles = model->header.numTriangles;

  int i;
  vertexMapType::iterator m;

  refcnt = 1;
  frameCount = numFrames;
  arrayElementsPerObject = 3*numTriangles;

  // Make a map from each unique <vertexIndex,textureIndex> pair.
  for (i=0; i<numTriangles; i++) {
    const Md2Triangle &tri = model->triangles[i];

    for (int j=0; j<3; j++) {
      PosTexIndex pti(tri.vertexIndices[j], tri.textureIndices[j]);
      VertexData vd;
      vd.ndx = tri.vertexIndices[j];
      vd.tc.s = model->texCoords[tri.textureIndices[j]].s;
      vd.tc.t = model->texCoords[tri.textureIndices[j]].t;
      vd.newndx = 0;  /* Satisfy gcc's need to see this initialized. */
      vertexMap[pti] = vd;
    }
  }
  // Assign "new" indices to each unique <vertexIndex,textureIndex> pair
  // and count unique arrayIndicesPerFrame.
  arrayIndicesPerFrame = 0;
  for (m = vertexMap.begin(); m != vertexMap.end(); ++m) {
      m->second.newndx = arrayIndicesPerFrame;
        posMap[m->second.ndx] = arrayIndicesPerFrame;
      arrayIndicesPerFrame++;
  }
  // Should be no more vertices in posMap than model has unique vertices.
  assert(posMap.size() <= (unsigned int)model->header.numVertices);

  // Generate unused buffer object names for three buffer objects.
  GLuint bufferObjs[3];
  glGenBuffers(3, bufferObjs);
  bufferObject.texCoordArray = bufferObjs[0];
  bufferObject.vertexNormalArray = bufferObjs[1];
  bufferObject.elementArray = bufferObjs[2];

  GLuint texObjs[3];
  glGenTextures(3, texObjs);
  textureObject.decal = texObjs[0];
  textureObject.normalMap = texObjs[1];
  textureObject.heightMap = texObjs[2];

  // Allocate a single texture coordinate array for model and load into
  // buffer object.
  TexCoord *texArray = new TexCoord[arrayIndicesPerFrame];
  int newndx = 0;
  for (m = vertexMap.begin(); m != vertexMap.end(); ++m) {
    texArray[newndx].s = m->second.tc.s;
    texArray[newndx].t = 65535 - m->second.tc.t;
    newndx++;
  }
  glBindBuffer(GL_ARRAY_BUFFER, bufferObject.texCoordArray);
  glBufferData(GL_ARRAY_BUFFER, sizeof(TexCoord)*arrayIndicesPerFrame,
    texArray, GL_STATIC_DRAW);
  delete [] texArray;

  // Allocate a vertex/normal array for model's frames and load into
  // buffer object.
  VertexNormal *vertexNormalArray = new
    VertexNormal[arrayIndicesPerFrame*numFrames];
  newndx = 0;
  for (int f = 0; f < model->header.numFrames; f++) {
    Md2Frame *frame = &model->frames[f];

    for (vertexMapType::iterator m = vertexMap.begin();
         m != vertexMap.end();
         ++m) {
      assert(m->second.newndx == newndx % arrayIndicesPerFrame);
      assert(m->second.newndx < arrayIndicesPerFrame);
      int oldndx = m->second.ndx;
      vertexNormalArray[newndx].x = frame->vertices[oldndx].vertex[0];
      vertexNormalArray[newndx].y = frame->vertices[oldndx].vertex[1];
      vertexNormalArray[newndx].z = frame->vertices[oldndx].vertex[2];
      vertexNormalArray[newndx].nx = frame->vertices[oldndx].normal[0];
      vertexNormalArray[newndx].ny = frame->vertices[oldndx].normal[1];
      vertexNormalArray[newndx].nz = frame->vertices[oldndx].normal[2];
      newndx++;
    }
  }
  glBindBuffer(GL_ARRAY_BUFFER, bufferObject.vertexNormalArray);
  glBufferData(GL_ARRAY_BUFFER,
    sizeof(VertexNormal)*arrayIndicesPerFrame*numFrames,
    vertexNormalArray, GL_STATIC_DRAW);
  delete [] vertexNormalArray;

  // Allocate a element array for model and load into buffer object.
  int ndx = 0;
  int elementArrayEntries = arrayElementsPerObject;
  if (withAdjacency) {
    elementArrayEntries += 2*arrayElementsPerObject;
  }
  GLushort *elementArray = new GLushort[elementArrayEntries];
  for (i=0; i<numTriangles; i++) {
    const Md2Triangle &tri = model->triangles[i];

    // MD2 models have clockwise front-facing triangles so reverse
    // index order to match OpenGL's default counter-clockwise default
    // for glFrontFace
    for (int j=2; j>=0; j--, ndx++) {
      PosTexIndex pti(tri.vertexIndices[j], tri.textureIndices[j]);

      VertexData vd = vertexMap[pti];
      elementArray[ndx] = vd.newndx;
    }
  }
  assert(arrayElementsPerObject == ndx);
  if (withAdjacency) {
    // Now triangle adjacency
    for (i=0; i<numTriangles; i++) {
      const Md2Triangle &tri = model->triangles[i];
      const Md2TriangleEdgeInfo &edgeInfo = model->edgeInfo[i];
      const unsigned char adjacentTriangleEdges =
        edgeInfo.adjacentTriangleEdges;

      // MD2 models have clockwise front-facing triangles so reverse
      // index order to match OpenGL's default counter-clockwise default
      // for glFrontFace
      for (int j=2; j>=0; j--) {
        int newndx = posMap[tri.vertexIndices[j]];
        elementArray[ndx++] = newndx;

        int jj = (j+2)%3;

        int ii = (ADJACENT_EDGE(adjacentTriangleEdges, jj)+2)%3;
        const Md2Triangle &adjtri =
          model->triangles[edgeInfo.adjacentTriangle[jj]];
        int adjvert = adjtri.vertexIndices[ii];

        assert(tri.vertexIndices[jj] == adjtri.vertexIndices[(ii+2)%3]);
        //printf("%d,%d\n", tri.vertexIndices[j], adjtri.vertexIndices[(ii+2)%3]);
        assert(tri.vertexIndices[(jj+1)%3] == adjtri.vertexIndices[(ii+1)%3]);
        //printf("%d:%d\n", tri.vertexIndices[(j+1)%3], adjtri.vertexIndices[(ii+1)%3]);

        newndx = posMap[adjvert];
        elementArray[ndx++] = newndx;
      }
    }
    assert(elementArrayEntries == ndx);
  }
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObject.elementArray);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
    sizeof(GLushort)*elementArrayEntries, elementArray, GL_STATIC_DRAW);
  delete [] elementArray;
}

ModelRenderInfo::~ModelRenderInfo()
{
  GLuint bufferObjs[3];

  bufferObjs[0] = bufferObject.elementArray;
  bufferObjs[1] = bufferObject.texCoordArray;
  bufferObjs[2] = bufferObject.vertexNormalArray;
  glDeleteBuffers(3, bufferObjs);
}

void ModelRenderInfo::bindArrayElementsForGL()
{
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObject.elementArray);
}

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

void ModelRenderInfo::bindTexCoordsForGL(int texunit)
{
  glClientActiveTexture(GL_TEXTURE0_ARB + texunit);
  glBindBuffer(GL_ARRAY_BUFFER, bufferObject.texCoordArray);
  glTexCoordPointer(2, GL_SHORT, sizeof(GLshort)*2, BUFFER_OFFSET(0));
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
}

void ModelRenderInfo::bindPositionsAndNormalsForGL(int frameA, int frameB,
                                                   int normals)
{
  // Bind to array buffer with per-frame position/normal arrays.
  glBindBuffer(GL_ARRAY_BUFFER, bufferObject.vertexNormalArray);

  // Frame A positions.
  glVertexPointer(3, GL_FLOAT, 6*sizeof(GLfloat),
    BUFFER_OFFSET(frameA*arrayIndicesPerFrame*6*sizeof(GLfloat)));
  glEnableClientState(GL_VERTEX_ARRAY);
  // Frame B positions.
  glClientActiveTexture(GL_TEXTURE1);
  glTexCoordPointer(3, GL_FLOAT, 6*sizeof(GLfloat),
    BUFFER_OFFSET(frameB*arrayIndicesPerFrame*6*sizeof(GLfloat)));
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);

  if (normals) {
    // Frame A normals.
    glNormalPointer(GL_FLOAT, 6*sizeof(GLfloat),
      BUFFER_OFFSET(3*sizeof(GLfloat) +
                    frameA*arrayIndicesPerFrame*6*sizeof(GLfloat)));
    glEnableClientState(GL_NORMAL_ARRAY);
    // Frame B normals.
    glClientActiveTexture(GL_TEXTURE2);
    glTexCoordPointer(3, GL_FLOAT, 6*sizeof(GLfloat),
      BUFFER_OFFSET(3*sizeof(GLfloat) +
                    frameB*arrayIndicesPerFrame*6*sizeof(GLfloat)));
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  }
}

void ModelRenderInfo::drawModel()
{
  glDrawElements(GL_TRIANGLES, arrayElementsPerObject,
    GL_UNSIGNED_SHORT, BUFFER_OFFSET(0));
}

void ModelRenderInfo::drawModelAdj()
{
  glDrawElements(GL_TRIANGLES_ADJACENCY_EXT,
    2*arrayElementsPerObject, GL_UNSIGNED_SHORT,
    BUFFER_OFFSET(arrayElementsPerObject*sizeof(GLushort)));
}

extern "C" {

MD2render *createMD2render(Md2Model *model)
{
  ModelRenderInfo *mri = new ModelRenderInfo(model);

  return (MD2render*) mri;
}

MD2render *createMD2renderWithAdjacency(Md2Model *model)
{
  ModelRenderInfo *mri = new ModelRenderInfo(model, true);

  return (MD2render*) mri;
}

void drawMD2render(MD2render *m, int frameA, int frameB)
{
  ModelRenderInfo *mri = reinterpret_cast<ModelRenderInfo *>(m);

  mri->bindArrayElementsForGL();
  mri->bindTexCoordsForGL(0);
  mri->bindPositionsAndNormalsForGL(frameA, frameB, 1);
  mri->drawModel();
}

void drawMD2renderWithAdjacency(MD2render *m, int frameA, int frameB)
{
  ModelRenderInfo *mri = reinterpret_cast<ModelRenderInfo *>(m);

  mri->bindArrayElementsForGL();
  mri->bindTexCoordsForGL(0);
  mri->bindPositionsAndNormalsForGL(frameA, frameB, 1);
  mri->drawModelAdj();
}

} // extern "C"
