
#include <Cg/cg.h>

extern void RegisterStates(CGcontext myCgContext);
extern void RegisterSamplerStates(CGcontext myCgContext);
extern CGstate RegisterState(const char *name, CGtype type, int size,
                             CGcontext context, int stateId);
