
/* mesh2d.h - efficient 2D mesh rendering */

typedef struct {
  int xsteps;
  int ysteps;
  GLfloat *data;
  GLuint *ndxs;
} Mesh2D;

Mesh2D createMesh2D(float x0, float x1, float y0, float y1, int xsteps, int ysteps);
void freeMesh2D(Mesh2D mesh);

typedef struct {
  GLuint elementArray;
  GLuint xyArray;
  int elementsPerTriStrpRow;
  int triStripRows;
} Mesh2D_GL;

Mesh2D_GL createMesh2D_GL(Mesh2D mesh);
void bindMesh2D_GL(Mesh2D_GL mesh);
void renderMesh2D_GL(Mesh2D_GL mesh);
void freeMesh2D_GL(Mesh2D_GL mesh);
