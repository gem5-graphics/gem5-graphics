
/* md2file.c - Quake 2 MD2 model loader */

#include <string.h>
#include <stdio.h>
#include <math.h>

#include "md2.h"

void
md2FreeModel(Md2Model *model)
{
  if (model) {
    if (model->texCoords) {
      free(model->texCoords);
    }
    if (model->triangles) {
      free(model->triangles);
    }
    if (model->frames) {
      int i;

      for (i = 0; i < model->header.numFrames; i++) {
        if (model->frames[i].vertices) {
          free(model->frames[i].vertices);
        }
      }
      free(model->frames);
    }
    if (model->filename) {
      free(model->filename);
    }
    free(model);
  }
}

static void
generateAutomaticNormals(Md2Model *model)
{
  int i, j;
  
  for (j = 0; j < model->header.numFrames; j++) {
    Md2Frame *f = &model->frames[j];
    Md2TriangleVertex *v = f->vertices;
    
    for (i = 0; i < model->header.numVertices; i++) {
      v[i].normal[0] = 0.0;
      v[i].normal[1] = 0.0;
      v[i].normal[2] = 0.0;
    }
    
    for (i = 0; i < model->header.numTriangles; i++) {
      Md2Triangle *t = &model->triangles[i];
      Md2TriangleVertex *v0, *v1, *v2;
      float vec0[3], vec1[3], n[3];
      float invLen;

      v0 = &v[t->vertexIndices[0]];
      v1 = &v[t->vertexIndices[1]];
      v2 = &v[t->vertexIndices[2]];
      
      /* Need 2 vectors to find cross product. */
      vec0[0] = v1->vertex[0] - v0->vertex[0];
      vec0[1] = v1->vertex[1] - v0->vertex[1];
      vec0[2] = v1->vertex[2] - v0->vertex[2];
      
      vec1[0] = v2->vertex[0] - v0->vertex[0];
      vec1[1] = v2->vertex[1] - v0->vertex[1];
      vec1[2] = v2->vertex[2] - v0->vertex[2];

      n[0] =   vec0[1] * vec1[2] - vec0[2] * vec1[1];
      n[1] = -(vec0[0] * vec1[2] - vec0[2] * vec1[0]);
      n[2] =   vec0[0] * vec1[1] - vec0[1] * vec1[0];
      invLen = (float) (-1.0/sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]));
      n[0] *= invLen;
      n[1] *= invLen;
      n[2] *= invLen;

      v0->normal[0] += n[0];
      v0->normal[1] += n[1];
      v0->normal[2] += n[2];
      v1->normal[0] += n[0];
      v1->normal[1] += n[1];
      v1->normal[2] += n[2];
      v2->normal[0] += n[0];
      v2->normal[1] += n[1];
      v2->normal[2] += n[2];
    }

    for (i = 0; i < model->header.numVertices; i++) {
      float dot, invLen;

      dot = v[i].normal[0] * v[i].normal[0] +
            v[i].normal[1] * v[i].normal[1] +
            v[i].normal[2] * v[i].normal[2];
      invLen = (float)(1.0/sqrt(dot));
      v[i].normal[0] *= invLen;
      v[i].normal[1] *= invLen;
      v[i].normal[2] *= invLen;
    }
  }
}

/* Return 2x the area of a 2D triangle.  Return a "double" so
   there is less chance of precision issues since we are
   generally interested in the sign of the area. */
static double
area2(float *v0,
      float *v1,
      float *v2)
{
  /* An equivalent algebraic expression for twice the area of a
     2D triangle is:
     
       v0[0]*v1[1] - v1[0]*v0[1] + 
       v1[0]*v2[1] - v2[0]*v1[1] + 
       v2[0]*v0[1] - v0[0]*v2[1]

     However this is 6 multiplies and 5 add/subtracs rather
     than the area2 implementations 2 multiplies and 5
     add/subtracts.  Moreover, the chosen implementation is
     less prone to overflows. */
  return (v1[0]-v0[0])*(v2[1]-v0[1]) - (v2[0]-v0[0])*(v1[1]-v0[1]);
}

/* Some models that I have encountered appear to have flipped normals.
   The Bobafett and Grey Alien models are examples of such models.
   This heuristic tries to determine when a model has bogus normals
   that should be flipped.  The algorithm is to sample each triangle
   on a pseudo-random (modulo 17) frame and test the area of the
   triangle flattened into the Z=0 plane (in model space) with the
   summed vertex normal Z components.  If the area and the normal
   component sum have the same sign, that is considered an indication that
   the normals for the model need to be flipped.  If more triangles
   appear to need flipping than not, then flip all the normals in
   the model.  Note that if a model doesn't backface cull properly,
   this heuristic would not work, but the facingness of the polygons
   seems reliable for the MD2 models that I've encountered. */
static int
flipNormalsIfProbablyNeeds(Md2Model *model)
{
  int lovesMe = 0, lovesMeNot = 0;
  unsigned int frame = 0;
  double a;
  float nz;
  int i, j;

  /* Sample each triangle on pseudo-random frames. */
  for (i = 0; i < model->header.numTriangles; i++, frame += 17) {
    Md2Frame *f = &model->frames[frame % model->header.numFrames];
    Md2Triangle *t = &model->triangles[i % model->header.numTriangles];
    Md2TriangleVertex *v0, *v1, *v2;

    v0 = &f->vertices[t->vertexIndices[0]];
    v1 = &f->vertices[t->vertexIndices[1]];
    v2 = &f->vertices[t->vertexIndices[2]];
    /* Compute the area of the polygon (in model space) projected
       onto the Z=0 plane. */
    a = area2(v0->vertex, v1->vertex, v2->vertex);
    /* Sum the Z components of the vertex normals. */
    nz = v0->normal[2] + v1->normal[2] + v2->normal[2];
    /* Don't count cases where the polygon area or normal Z sum are zero.
       Count cases where the polygon area and normal Z sum have opposite
       signs as an indication that we don't need to flip normals, but when
       the signs are the same, count that as a sign that we may need to
       flip normals. */
    if (a > 0) {
      if (nz > 0) {
        lovesMeNot++;
      } else if (nz < 0) {
        lovesMe++;
      }
    } else if (a < 0) {
      if (nz > 0) {
        lovesMe++;
      } else if (nz < 0) {
        lovesMeNot++;
      }
    }
  }

  /* If more normals seem wrong than not wrong... */
  if (lovesMeNot > lovesMe) {
    /* Then flip all the normals. */
    for (i = 0; i < model->header.numFrames; i++) {
      for (j = 0; j < model->header.numVertices; j++) {
        model->frames[i].vertices[j].normal[0] *= -1;
        model->frames[i].vertices[j].normal[1] *= -1;
        model->frames[i].vertices[j].normal[2] *= -1;
      }
    }
  }
  return 0;
}

/* assume _WIN32 (Windows) is always little-endian */
#if defined(__LITTLE_ENDIAN__) || defined(_WIN32)
/* target is already little endian so no swapping is needed to read little-endian data */
#else
static const unsigned int nativeIntOrder = 0x03020100;
static const unsigned short nativeShortOrder = 0x0100;

#define LE_INT32_BYTE_OFFSET(a) (((unsigned char*)&nativeIntOrder)[a])
#define LE_INT16_BYTE_OFFSET(a) (((unsigned char*)&nativeShortOrder)[a])
#endif

static short short_le2native(short v)
{
/* works even if little-endian target and __LITTLE_ENDIAN__ not defined */
#if defined(__LITTLE_ENDIAN__) || defined(_WIN32)
  return v;
#else
  union {
    short s;
    unsigned char b[2];
  } src, dst;

  src.s = v;
  dst.b[0] = src.b[LE_INT16_BYTE_OFFSET(0)];
  dst.b[1] = src.b[LE_INT16_BYTE_OFFSET(1)];
  return dst.s;
#endif
}

static int int_le2native(int v)
{
/* works even if little-endian target and __LITTLE_ENDIAN__ not defined */
#if defined(__LITTLE_ENDIAN__) || defined(_WIN32)
  return v;
#else
  union {
    int i;
    unsigned char b[4];
  } src, dst;

  src.i = v;
  dst.b[0] = src.b[LE_INT32_BYTE_OFFSET(0)];
  dst.b[1] = src.b[LE_INT32_BYTE_OFFSET(1)];
  dst.b[2] = src.b[LE_INT32_BYTE_OFFSET(2)];
  dst.b[3] = src.b[LE_INT32_BYTE_OFFSET(3)];
  return dst.i;
#endif
}

static float float_le2native(float v)
{
/* works even if little-endian target and __LITTLE_ENDIAN__ not defined */
#if defined(__LITTLE_ENDIAN__) || defined(_WIN32)
  return v;
#else
  union {
    float f;
    unsigned char b[4];
  } src, dst;

  src.f = v;
  dst.b[0] = src.b[LE_INT32_BYTE_OFFSET(0)];
  dst.b[1] = src.b[LE_INT32_BYTE_OFFSET(1)];
  dst.b[2] = src.b[LE_INT32_BYTE_OFFSET(2)];
  dst.b[3] = src.b[LE_INT32_BYTE_OFFSET(3)];
  return dst.f;
#endif
}

Md2Model *
md2ReadModel(const char *filename)
{
  FILE *file;
  Md2Model *model;
  unsigned char buffer[MD2_MAX_FRAMESIZE];
  int zeroLightNormalIndexCount = 0;
  int i;
  
  model = (Md2Model *) malloc(sizeof(Md2Model));
  if (!model) {
    return NULL;
  }
  
  file = fopen(filename, "rb");
  if (!file) {
    free(model);
    return 0;
  }
 
  /* initialize model and read header */
  memset(model, 0, sizeof(Md2Model));
  fread(&model->header, sizeof(Md2Header), 1, file);

  /* Byte-swap various values in the little-endian file format. */
  model->header.magic = int_le2native(model->header.magic); 
  model->header.version = int_le2native(model->header.version); 
  model->header.skinWidth = int_le2native(model->header.skinWidth);    
  model->header.skinHeight = int_le2native(model->header.skinHeight); 
  model->header.frameSize = int_le2native(model->header.frameSize); 
  model->header.numSkins = int_le2native(model->header.numSkins);   
  model->header.numVertices = int_le2native(model->header.numVertices); 
  model->header.numTexCoords =
    int_le2native(model->header.numTexCoords); 
  model->header.numTriangles =
    int_le2native(model->header.numTriangles);   
  model->header.numGlCommands =
    int_le2native(model->header.numGlCommands); 
  model->header.numFrames = int_le2native(model->header.numFrames);    
  model->header.offsetSkins = int_le2native(model->header.offsetSkins);
  model->header.offsetTexCoords =
    int_le2native(model->header.offsetTexCoords);    
  model->header.offsetTriangles =
    int_le2native(model->header.offsetTriangles); 
  model->header.offsetFrames =
    int_le2native(model->header.offsetFrames);   
  model->header.offsetGlCommands =
    int_le2native(model->header.offsetGlCommands); 
  model->header.offsetEnd = int_le2native(model->header.offsetEnd); 

  if (model->header.magic !=
      (int) (('2' << 24) + ('P' << 16) + ('D' << 8) + 'I')) {
    fclose(file);
    free(model);
    return 0;
  }

  /* We skip the "skins" section of the MD2 file. */

  /* Read texture coordinates. */
  fseek(file, model->header.offsetTexCoords, SEEK_SET);
  if (model->header.numTexCoords > 0) {
    model->texCoords = (Md2TextureCoordinate *)
      malloc(sizeof(Md2TextureCoordinate) * model->header.numTexCoords);
    if (!model->texCoords) {
      md2FreeModel(model);
      return 0;
    }
    
    fread(model->texCoords, sizeof(Md2TextureCoordinate),
      model->header.numTexCoords, file);
  }
  /* Byte-swap various values in the little-endian file format. */
  for (i = 0; i < model->header.numTexCoords; i++) {
    model->texCoords[i].s = short_le2native(model->texCoords[i].s);
    model->texCoords[i].t = short_le2native(model->texCoords[i].t);          
  }
  
  /* Read triangles. */
  fseek(file, model->header.offsetTriangles, SEEK_SET);
  if (model->header.numTriangles > 0) {
    model->triangles = (Md2Triangle *)
      malloc(sizeof(Md2Triangle) * model->header.numTriangles);
    if (!model->triangles) {
      md2FreeModel(model);
      return 0;
    }
    
    fread(model->triangles, sizeof(Md2Triangle),
      model->header.numTriangles, file);

    /* Byte-swap various values in the little-endian file format. */
    for (i = 0; i < model->header.numTriangles; i++) {
      model->triangles[i].vertexIndices[0] =
        short_le2native(model->triangles[i].vertexIndices[0]);
      model->triangles[i].vertexIndices[1] =
        short_le2native(model->triangles[i].vertexIndices[1]);
      model->triangles[i].vertexIndices[2] =
        short_le2native(model->triangles[i].vertexIndices[2]);
      model->triangles[i].textureIndices[0] =
        short_le2native(model->triangles[i].textureIndices[0]);
      model->triangles[i].textureIndices[1] =
        short_le2native(model->triangles[i].textureIndices[1]);
      model->triangles[i].textureIndices[2] =
        short_le2native(model->triangles[i].textureIndices[2]);
    }
  }
  
  /* Read alias frames. */
  fseek(file, model->header.offsetFrames, SEEK_SET);
  if (model->header.numFrames > 0) {
    model->frames =
      (Md2Frame *) malloc(sizeof(Md2Frame) * model->header.numFrames);
    if (!model->frames) {
      md2FreeModel(model);
      return 0;
    }
    
    for (i = 0; i < model->header.numFrames; i++) {
      Md2AliasFrame *frame = (Md2AliasFrame *) buffer;
      int j;
      
      model->frames[i].vertices = (Md2TriangleVertex *)
        malloc(sizeof(Md2TriangleVertex) * model->header.numVertices);
      if (!model->frames[i].vertices) {
        md2FreeModel(model);
        return 0;
      }
      
      fread(frame, 1, model->header.frameSize, file);
      strcpy(model->frames[i].name, frame->name);

      /* Byte-swapping 32-bit values in the little-endian file format. */
      frame->scale[0] =
        float_le2native(frame->scale[0]);
      frame->scale[1] =
        float_le2native(frame->scale[1]);
      frame->scale[2] =
        float_le2native(frame->scale[2]);
      frame->translate[0] =
        float_le2native(frame->translate[0]);
      frame->translate[1] =
        float_le2native(frame->translate[1]);
      frame->translate[2] =
        float_le2native(frame->translate[2]);
                  
      for (j = 0; j < model->header.numVertices; j++) {  
        int lightNormalIndex;

        /* Why is the Z coordinate negated?  I believe that
           there is a coordinate system handedness switch
           occuring. */
        model->frames[i].vertices[j].vertex[0] = (float)
          ((int) frame->alias_vertices[j].vertex[0]) * frame->scale[0]
          + frame->translate[0];
        model->frames[i].vertices[j].vertex[1] = (float)
          ((int) frame->alias_vertices[j].vertex[2]) * frame->scale[2]
          + frame->translate[2];
        model->frames[i].vertices[j].vertex[2] = -1*
          ((float) ((int) frame->alias_vertices[j].vertex[1]) * frame->scale[1]
                        + frame->translate[1]);
        
        lightNormalIndex = frame->alias_vertices[j].lightNormalIndex;
        if (lightNormalIndex == 0) {
          zeroLightNormalIndexCount++;
        }
        model->frames[i].vertices[j].normal[0] =
          md2VertexNormals[lightNormalIndex][0];
        model->frames[i].vertices[j].normal[1] =
          md2VertexNormals[lightNormalIndex][2];
        model->frames[i].vertices[j].normal[2] =
          -md2VertexNormals[lightNormalIndex][1];
      }
    }

    if (model->header.numFrames * model->header.numVertices ==
        zeroLightNormalIndexCount) {
      generateAutomaticNormals(model);
    }
  }

  fclose(file);

  flipNormalsIfProbablyNeeds(model);

  /* We skip the "GL commands" section for stripped rendering
     of the model.  We naively send independent triangles and
     let the GPU's pre- and post-transform vertex caching do
     its thing. */

  return model;
}
