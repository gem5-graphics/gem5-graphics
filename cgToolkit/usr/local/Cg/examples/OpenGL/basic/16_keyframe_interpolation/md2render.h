
#ifdef  __cplusplus
extern "C" {
#endif

typedef struct _MD2render MD2render;

MD2render *createMD2render(Md2Model *model);
void drawMD2render(MD2render *m, int frameA, int frameB);

#ifdef  __cplusplus
}
#endif
