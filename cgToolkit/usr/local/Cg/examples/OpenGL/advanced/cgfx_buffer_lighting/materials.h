
/* materials.h - static data for various materials */

typedef struct {
  float ambient[4];
  float diffuse[4];
  float specular[4];
  float shine[4];
} MaterialData;

typedef struct {
  const char *name;
  MaterialData data;
} MaterialInfo;

extern const MaterialInfo materialInfo[];
extern const int materialInfoCount;
