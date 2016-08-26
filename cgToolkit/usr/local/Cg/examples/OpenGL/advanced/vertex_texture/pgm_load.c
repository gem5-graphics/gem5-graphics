
/* pgm.c - simple Portable Grayscale Bitmap loader */

#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>

GLuint pgm_load(const char *filename, GLenum internalFormat)
{
  FILE *file = fopen(filename, "rb");

  if (file) {
    int width, height, maxvalue;
    size_t got;
    int rc;

    /* "%*s" means skip over string. */
    rc = fscanf(file, "P5 #%*[^\n]s");
    rc += fscanf(file, "%d %d %d\n", &width, &height, &maxvalue);
    if (rc == 3 && maxvalue == 255 && width >= 1 && height >= 1) {
      size_t size = width * height;
      char *buffer = malloc(size);

      if (buffer) {
        got = fread(buffer, 1, size, file);
        if (got == size) {
          GLuint texobj;

          fclose(file);
          glGenTextures(1, &texobj);
          glBindTexture(GL_TEXTURE_2D, texobj);
          glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0,
            GL_LUMINANCE, GL_UNSIGNED_BYTE, buffer);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
          return texobj;
        } else {
          /* not enough data read */
        }
        free(buffer);
      } else {
        /* malloc failed */
      }
    } else {
      /* PGM header mismatch */
    }
    fclose(file);
  } else {
    /* failed to open file */
  }
  return 0;
}
