
#include <assert.h>
#include <stdlib.h>  /* for exit */

#include "gli.h"

extern const char *myProgramName;

gliGenericImage *
readImage(const char *filename)
{
  FILE *file;
  gliGenericImage *image;
  int yFlip = 1;

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
