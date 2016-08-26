
#include "gli.h"

#ifdef  __cplusplus
extern "C" {
#endif

extern gliGenericImage *readImage(const char *filename);
extern gliGenericImage *loadTextureDecal(gliGenericImage *image, int mipmap);
extern void loadTextureNormalMap(gliGenericImage *image, const char *filename, float scale);

#ifdef  __cplusplus
}
#endif
