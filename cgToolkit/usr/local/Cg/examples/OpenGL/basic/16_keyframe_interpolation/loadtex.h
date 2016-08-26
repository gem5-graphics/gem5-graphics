
#include "gli.h"

#ifdef  __cplusplus
extern "C" {
#endif

extern gliGenericImage *readImage(const char *filename);
extern gliGenericImage *loadTextureDecal(gliGenericImage *image, int mipmap);

#ifdef  __cplusplus
}
#endif
