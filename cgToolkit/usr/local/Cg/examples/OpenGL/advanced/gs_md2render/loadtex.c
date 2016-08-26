
#include <assert.h>
#include <stdlib.h>  /* for exit */

#include "gli.h"
#include "normalmap.h"

extern const char *myProgramName;

gliGenericImage *
readImage(const char *filename)
{
  FILE *file;
  gliGenericImage *image;
  int yFlip = 0;

  file = fopen(filename, "rb");
  if (file == NULL) {
    printf("%s: could not open \"%s\"\n", myProgramName, filename);
    exit(1);
  }
  image = gliReadTGA(file, filename, yFlip);
  if (image == NULL) {
    printf("%s: \"%s\" is not a TGA image\n", myProgramName, filename);
    exit(1);
  }
  fclose(file);
  return image;
}

static int
roundUp(int v)
{
  int i;

  for (i=0; i<31; i++) {
    if (v <= (1<<i)) {
      return 1<<i;
    }
  }
  return 1<<31;
}

gliGenericImage *
loadTextureDecal(gliGenericImage *image, int mipmap)
{
  int needsScaling;
  int nw, nh;

  nw = roundUp(image->width);
  nh = roundUp(image->height);

  if ((nw != image->width) || (nh != image->height)) {
    needsScaling = 1;
  } else {
    needsScaling = 0;
  }
  assert(image->format != GL_COLOR_INDEX);
  if (needsScaling) {
    gliGenericImage *nimage;
      
    nimage = gliScaleImage(image, nw, nh);
    gliFree(image);
    image = nimage;
  }

#ifdef __APPLE__
  mipmap = 0;  /* Why doesn't Apple's gluBuild2DMipmaps work correctly? */
#endif
  if (mipmap) {
    GLint status;

    glTexParameteri(GL_TEXTURE_2D,
      GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    status = gluBuild2DMipmaps(GL_TEXTURE_2D, image->internalFormat,
      nw, nh, image->format, image->type, image->pixels);
    if (status == GLU_INVALID_ENUM) {
      gliConvertImageToCoreFormat(image);
      status = gluBuild2DMipmaps(GL_TEXTURE_2D, image->internalFormat,
        nw, nh, image->format, image->type, image->pixels);
    }
    assert(status == 0);
  } else {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, image->internalFormat,
      nw, nh, 0,
      image->format, image->type, image->pixels);
  }

  return image;
}

void
loadTextureNormalMap(gliGenericImage *image, const char *filename, float scale)
{
  Normal *nmap;
  int w, h, wr = 0, hr, nw, nh, level, badSize;

  if (image == NULL) {
    printf("%s: failed to load decal skin %s\n", myProgramName, filename);
    exit(0);
  }

  w = image->width;
  h = image->height;

  badSize = 0;
  if ( (w & (w-1))) {
    if ( ((w-1) & (w-2))) {
      /* Width not 2^n or 2^+1. */
      badSize = 1;
    } else {
      /* Width is power of two plus one, use border */
      wr = w;
      w = w-1;
    }
  } else {
    /* Width is a power of two, wrap normal map width. */
    wr = w;
  }

  if ( (h & (h-1))) {
    if ( ((h-1) & (h-2))) {
      /* Height not 2^n or 2^+1. */
      badSize = 1;
    } else {
      /* Height is power of two plus one, use border */
      hr = h;
      h = h-1;
    }
  } else {
    /* Height is a power of two, wrap normal map height. */
    hr = h;
  }

  if (badSize) {
    fprintf(stderr,
      "%s: normal map \"%s\" must have 2^n or 2^n+1 dimensions,"
      " not %dx%d\n", myProgramName, filename, w, h);
    exit(1);
  }

  nmap = convertHeightFieldToNormalMap(image->pixels, w, h, wr, hr, scale);

  level = 0;

  /* Load original maximum resolution normal map. */
  glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA8, w, h, level,
    GL_BGRA_EXT, GL_UNSIGNED_BYTE, &nmap->nz);

  /* Downsample the normal map for mipmap levels down to 1x1. */
  while (w > 1 || h > 1) {
    level++;

    /* Half width and height but not beyond one. */
    nw = w >> 1;
    nh = h >> 1;
    if (nw == 0) nw = 1;
    if (nh == 0) nh = 1;

    nmap = downSampleNormalMap(nmap, w, h, nw, nh);

    glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA8, nw, nh, 0,
      GL_BGRA_EXT, GL_UNSIGNED_BYTE, &nmap->nz);

    /* Make the new width and height the old width and height. */
    w = nw;
    h = nh;
  }

  free(nmap);
}
