//stuff that were moved from fakeglx.c
#ifndef FAKEGLX_H
#define FAKEGLX_H

#include "glxheader.h"
#include "glxapi.h"
#include "main/context.h"
#include "main/config.h"
#include "main/macros.h"
#include "main/imports.h"
#include "main/mtypes.h"
#include "main/version.h"
#include "xfonts.h"
#include "xmesaP.h"


/*
 * Our fake GLX context will contain a "real" GLX context and an XMesa context.
 *
 * Note that a pointer to a __GLXcontext is a pointer to a fake_glx_context,
 * and vice versa.
 *
 * We really just need this structure in order to make the libGL functions
 * glXGetCurrentContext(), glXGetCurrentDrawable() and glXGetCurrentDisplay()
 * work correctly.
 */
struct fake_glx_context {
   __GLXcontext glxContext;   /* this MUST be first! */
   XMesaContext xmesaContext;
};

#endif