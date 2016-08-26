
/* normalmap.c - construct a normal map from a height-field and load it
                 as a mipmapped texture */

/* Copyright NVIDIA Corporation, 1999. */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <GL/glew.h>

/*** NORMAL MAP TEXTURE CONSTRUCTION ***/

/* Structure to encode a normal like an 8-bit unsigned BGRA vector. */
typedef struct {
  /* Normalized tangent space peturbed surface normal.  The
     [0,1] range of (nx,ny,nz) gets expanded to the [-1,1]
     range in the combiners.  The (nx,ny,nz) is always a
     normalized vector. */
  GLubyte nz, ny, nx;

  /* A scaling factor for the normal.  Mipmap level 0 has a constant
     magnitude of 1.0, but downsampled mipmap levels keep track of
     the unnormalized vector sum length.  For diffuse per-pixel
     lighting, it is preferable to make N' be the _unnormalized_
     vector, but for specular lighting to work reasonably, the
     normal vector should be normalized.  In the diffuse case, we
     can multiply by the "mag" to get the possibly shortened
     unnormalized length. */
  GLubyte mag;

  /* Why does "mag" make sense for diffuse lighting?

     Because sum(L dot Ni)/n == (L dot sum(Ni))/n 
  
     Think about a bumpy diffuse surface in the distance.  It should
     have a duller illumination than a flat diffuse surface in the
     distance. */

  /* On NVIDIA GPUs, the RGB8 internal format is just as memory
     efficient as the RGB8 internal texture format so keeping
     "mag" around is just as cheap as not having it. */

} Normal;

/* Convert a height field image into a normal map.  This involves
   differencing each texel with its right and upper neighboor, then
   normalizing the cross product of the two difference vectors. */
Normal *
convertHeightFieldToNormalMap(GLubyte *pixels,
                              int w, int h, int wr, int hr, float scale)
{
  int i, j;
  Normal *nmap;
  float sqlen, reciplen, nx, ny, nz;

  const float oneOver255 = 1.0f/255.0f;

  float c, cx, cy, dcx, dcy;

  nmap = (Normal*) malloc(sizeof(Normal)*w*h);

  for (i=0; i<h; i++) {
    for (j=0; j<w; j++) {
      /* Expand [0,255] texel values to the [0,1] range. */
      c = pixels[i*wr + j] * oneOver255;
      /* Expand the texel to its right. */
      cx = pixels[i*wr + (j+1)%wr] * oneOver255;
      /* Expand the texel one up. */
      cy = pixels[((i+1)%hr)*wr + j] * oneOver255;
      dcx = scale * (cx - c);
      dcy = scale * (cy - c);

      /* Normalize the vector. */
      sqlen = dcx*dcx + dcy*dcy + 1;
      reciplen = 1.0f/(float)sqrt(sqlen);
      nx = -dcx*reciplen;
      ny = -dcy*reciplen;
      nz = reciplen;

      /* Repack the normalized vector into an RGB unsigned byte
         vector in the normal map image. */
      nmap[i*w+j].nx = (GLubyte) (128 + 127*nx);
      nmap[i*w+j].ny = (GLubyte) (128 + 127*ny);
      nmap[i*w+j].nz = (GLubyte) (128 + 127*nz);

      /* The highest resolution mipmap level always has a
         unit length magnitude. */
      nmap[i*w+j].mag = 255;
    }
  }

  return nmap;
}

/* Given a normal map, create a downsampled version of the normal map
   at half the width and height.  Use a 2x2 box filter to create each
   downsample.  gluBuild2DMipmaps is not suitable because each downsampled
   texel must also be renormalized. */
Normal *
downSampleNormalMap(Normal *old, int w2, int h2, int w, int h)
{
  const float oneOver127 = 1.0f/127.0f;
  const float oneOver255 = 1.0f/255.0f;

  Normal *nmap;
  float x, y, z, l, invl;
  float mag00, mag01, mag10, mag11;
  int i, j, ii, jj;

  /* Allocate space for the downsampled normal map level. */
  nmap = (Normal*) malloc(sizeof(Normal)*w*h);

  for (i=0; i<h2; i+=2) {
    for (j=0; j<w2; j+=2) {

      /* The "%w2" and "%h2" modulo arithmetic makes sure that
         Nx1 and 1xN normal map levels are handled correctly. */

      /* Fetch the magnitude of the four vectors to be downsampled. */
      mag00 = oneOver255 * old[ (i  )    *w2 +  (j  )    ].mag;
      mag01 = oneOver255 * old[ (i  )    *w2 + ((j+1)%h2)].mag;
      mag10 = oneOver255 * old[((i+1)%w2)*w2 +  (j  )    ].mag;
      mag11 = oneOver255 * old[((i+1)%w2)*w2 + ((j+1)%h2)].mag;

      /* Sum 2x2 footprint of red component scaled back to [-1,1]
         floating point range. */
      x =  mag00 * (oneOver127 * old[ (i  )    *w2 +  (j  )    ].nx - 1.0f);
      x += mag01 * (oneOver127 * old[ (i  )    *w2 + ((j+1)%h2)].nx - 1.0f);
      x += mag10 * (oneOver127 * old[((i+1)%w2)*w2 +  (j  )    ].nx - 1.0f);
      x += mag11 * (oneOver127 * old[((i+1)%w2)*w2 + ((j+1)%h2)].nx - 1.0f);

      /* Sum 2x2 footprint of green component scaled back to [-1,1]
         floating point range. */
      y =  mag00 * (oneOver127 * old[ (i  )    *w2 +  (j  )    ].ny - 1.0f);
      y += mag01 * (oneOver127 * old[ (i  )    *w2 + ((j+1)%h2)].ny - 1.0f);
      y += mag10 * (oneOver127 * old[((i+1)%w2)*w2 +  (j  )    ].ny - 1.0f);
      y += mag11 * (oneOver127 * old[((i+1)%w2)*w2 + ((j+1)%h2)].ny - 1.0f);

      /* Sum 2x2 footprint of blue component scaled back to [-1,1]
         floating point range. */
      z =  mag00 * (oneOver127 * old[ (i  )    *w2 +  (j  )    ].nz - 1.0f);
      z += mag01 * (oneOver127 * old[ (i  )    *w2 + ((j+1)%h2)].nz - 1.0f);
      z += mag10 * (oneOver127 * old[((i+1)%w2)*w2 +  (j  )    ].nz - 1.0f);
      z += mag11 * (oneOver127 * old[((i+1)%w2)*w2 + ((j+1)%h2)].nz - 1.0f);

      /* Compute length of the (x,y,z) vector. */
      l = (float) sqrt(x*x + y*y + z*z);
      if (l == 0.0) {
        /* Ugh, a zero length vector.  Avoid division by zero and just
           use the unpeturbed normal (0,0,1). */
        x = 0.0;
        y = 0.0;
        z = 1.0;
      } else {
        /* Normalize the vector to unit length. */
        invl = 1.0f/l;
        x = x*invl;
        y = y*invl;
        z = z*invl;
      }

      ii = i >> 1;
      jj = j >> 1;

      /* Pack the normalized vector into an RGB unsigned byte vector
         in the downsampled image. */
      nmap[ii*w+jj].nx = (GLubyte) (128 + 127*x);
      nmap[ii*w+jj].ny = (GLubyte) (128 + 127*y);
      nmap[ii*w+jj].nz = (GLubyte) (128 + 127*z);

      /* Store the magnitude of the average vector in the alpha
         component so we keep track of the magntiude. */
      l = l/4.0f;
      if (l > 1.0) {
        nmap[ii*w+jj].mag = 255;
      } else {
        nmap[ii*w+jj].mag = (GLubyte) (255.0f*l);
      }
    }
  }

  free(old);

  return nmap;
}

/* Convert the supplied height-field image into a normal map (a normalized
   vector compressed to the [0,1] range in RGB and A=1.0).  Load the
   base texture level, then recursively downsample and load successive
   normal map levels (being careful to expand, average, renormalize,
   and unexpand each RGB value an also accumulate the average vector
   shortening in alpha). */
void
convertHeightFieldAndLoadNormalMapTexture(GLubyte *pixels, int w, int h,
                                          int wr, int hr, float scale)
{
  Normal *nmap;
  int level;

  nmap = convertHeightFieldToNormalMap(pixels, w, h, wr, hr, scale);

  level = 0;

  /* Load original maximum resolution normal map. */

  /* The BGRA color component ordering is fastest for NVIDIA. */
  glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA8, w, h, level,
    GL_BGRA, GL_UNSIGNED_BYTE, &nmap->nz);

  /* Downsample the normal map for mipmap levels down to 1x1. */
  while (w > 1 || h > 1) {
    int nw, nh;

    level++;

    /* Half width and height but not beyond one. */
    nw = w >> 1;
    nh = h >> 1;
    if (nw == 0) nw = 1;
    if (nh == 0) nh = 1;

    nmap = downSampleNormalMap(nmap, w, h, nw, nh);

    glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA8, nw, nh, 0,
      GL_BGRA, GL_UNSIGNED_BYTE, &nmap->nz);

    /* Make the new width and height the old width and height. */
    w = nw;
    h = nh;
  }

  free(nmap);
}

