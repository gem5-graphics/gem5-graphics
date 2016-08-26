
/* pgm_load.h - simple Portable Grayscale Bitmap loader */

#ifndef __PGM_LOAD_H__
#define __PGM_LOAD_H__

#include <GL/glew.h>

GLuint pgm_load(const char *filename, GLenum internalFormat);

#endif /* __PGM_LOAD_H__ */
