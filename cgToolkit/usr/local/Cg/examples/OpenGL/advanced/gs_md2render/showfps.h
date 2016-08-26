#ifndef SHOWFPS_H
#define SHOWFPS_H

/* showfps.h - OpenGL code for rendering frames per second */

/* Call handleFPS in your GLUT display callback every frame. */

extern void handleFPS(void);
extern void toggleFPS();
extern void enableFPS();
extern void disableFPS();
extern void colorFPS(float r, float g, float b);

#endif /* SHOWFPS_H */
