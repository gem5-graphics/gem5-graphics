#ifndef __md2render_h__
#define __md2render_h__

/* md2render.cpp - C API for class to help load and render Quake2 MD2 models via OpenGL */

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct _MD2render MD2render;

MD2render *createMD2render(Md2Model *model);
MD2render *createMD2renderWithAdjacency(Md2Model *model);
void drawMD2render(MD2render *m, int frameA, int frameB);
void drawMD2renderWithAdjacency(MD2render *m, int frameA, int frameB);

#ifdef  __cplusplus
}
#endif

#endif /* __md2render_h__ */
