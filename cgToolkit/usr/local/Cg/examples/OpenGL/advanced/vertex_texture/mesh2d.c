
/* mesh2d.c - efficient 2D mesh rendering */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

extern const char *programName;

#include <GL/glew.h>

typedef struct {
  int xsteps;
  int ysteps;
  GLfloat *data;
  GLuint *ndxs;
} Mesh2D;

static size_t meshAttribDataInBytes(int xsteps, int ysteps)
{
  return sizeof(GLfloat)*2*(xsteps+1)*(ysteps+1);
}

static size_t meshElementDataInBytes(int xsteps, int ysteps)
{
  return sizeof(GLuint)*ysteps*2*(xsteps+1);
}

Mesh2D createMesh2D(float x0, float x1, float y0, float y1, int xsteps, int ysteps)
{
  Mesh2D retval;
  const float dx = (x1-x0)/xsteps,
              dy = (y1-y0)/ysteps;
  int i, j;

  const size_t datalen = meshAttribDataInBytes(xsteps, ysteps),
               ndxslen = meshElementDataInBytes(xsteps, ysteps);

  GLfloat *data = malloc(datalen);
  GLuint *ndxs = malloc(ndxslen);

  if (!data || !ndxs) {
    fprintf(stderr, "%s: malloc failed\n", programName);
    exit(1);
  }

  /* Fill XY vertex buffer. */
  {
    GLfloat *loc = data + 2*((xsteps+1)*0 + 0);

    assert((char*)loc < (char*)data + datalen);
    loc[0] = x0;
    loc[1] = y0;
    for (i=1; i<xsteps; i++) {
      loc = data + 2*i;
      assert((char*)loc < (char*)data + datalen);
      loc[0] = x0 + i*dx;
      loc[1] = y0;
    }
    loc = data + 2*i;
    assert((char*)loc < (char*)data + datalen);
    loc[0] = x1;
    loc[1] = y0;
  }
  for (j=1; j<ysteps; j++) {
    GLfloat *loc = data + 2*((xsteps+1)*j + 0);

    assert((char*)loc < (char*)data + datalen);
    loc[0] = x0;
    loc[1] = y0 + j*dy;
    for (i=1; i<xsteps; i++) {
      loc = data + 2*((xsteps+1)*j + i);
      assert((char*)loc < (char*)data + datalen);
      loc[0] = x0 + i*dx;
      loc[1] = y0 + j*dy;
    }
    loc = data + 2*((xsteps+1)*j + i);
    assert((char*)loc < (char*)data + datalen);
    loc[0] = x1;
    loc[1] = y0 + j*dy;
  }
  {
    GLfloat *loc = data + 2*((xsteps+1)*j + 0);

    assert((char*)loc < (char*)data + datalen);
    loc[0] = x0;
    loc[1] = y1;
    for (i=1; i<xsteps; i++) {
      loc = data + 2*((xsteps+1)*j + i);
      assert((char*)loc < (char*)data + datalen);
      loc[0] = x0 + i*dx;
      loc[1] = y1;
    }
    loc = data + 2*((xsteps+1)*j + i);
    assert((char*)loc < (char*)data + datalen);
    loc[0] = x1;
    loc[1] = y1;
  }

  /* Fill array element buffer. */
  for (j=0; j<ysteps; j++) {
    for (i=0; i<=xsteps; i++) {
      GLuint *ndx = ndxs + j*2*(xsteps+1) + i*2;

      assert((char*)ndx < (char*)ndxs + ndxslen);
      ndx[0] = (j+1)*(xsteps+1) + i;
      ndx[1] = j*(xsteps+1) + i;
    }
  }

  retval.data = data;
  retval.ndxs = ndxs;
  retval.xsteps = xsteps;
  retval.ysteps = ysteps;
  return retval;
}

void freeMesh2D(Mesh2D mesh)
{
  free(mesh.data);
  free(mesh.ndxs);
}

typedef struct {
  GLuint elementArray;
  GLuint xyArray;
  int elementsPerTriStrpRow;
  int triStripRows;
} Mesh2D_GL;

Mesh2D_GL createMesh2D_GL(Mesh2D mesh)
{
  Mesh2D_GL retval;
  GLuint bufs[2];

  glGenBuffers(2, bufs);
  retval.elementArray = bufs[0];
  retval.xyArray = bufs[1];

  retval.elementsPerTriStrpRow = 2*(mesh.xsteps+1);
  retval.triStripRows = mesh.ysteps;

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, retval.elementArray);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
    meshElementDataInBytes(mesh.xsteps, mesh.ysteps), mesh.ndxs, GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, retval.xyArray);
  glBufferData(GL_ARRAY_BUFFER,
    meshAttribDataInBytes(mesh.xsteps, mesh.ysteps), mesh.data, GL_STATIC_DRAW);

  return retval;
}

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

void bindMesh2D_GL(Mesh2D_GL mesh)
{
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.elementArray);
  glBindBuffer(GL_ARRAY_BUFFER, mesh.xyArray);
  glVertexPointer(2, GL_FLOAT, 2*sizeof(GLfloat), BUFFER_OFFSET(0));
  glEnableClientState(GL_VERTEX_ARRAY);
}

void renderMesh2D_GL(Mesh2D_GL mesh)
{
  int i;

  for (i=0; i<mesh.triStripRows; i++) {
    glDrawElements(GL_TRIANGLE_STRIP, mesh.elementsPerTriStrpRow,
      GL_UNSIGNED_INT, BUFFER_OFFSET(i*4*mesh.elementsPerTriStrpRow));
  }
}

void freeMesh2D_GL(Mesh2D_GL mesh)
{
  GLuint bufs[2];

  bufs[0] = mesh.elementArray;
  bufs[0] = mesh.xyArray;
  glDeleteBuffers(2, bufs);
}
