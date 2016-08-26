#ifndef TRACE_CG_STATE
#define TRACE_CG_STATE

#include <stdlib.h>
#include <assert.h>
#include "uthash.h"

#include <Cg/cg.h>

/* CGIncludeCallbackFuncMap - UT hash map for include callbacks. */

struct CGIncludeCallbackFuncMap
{
  size_t context;               /* UT hash map key.               */
  CGIncludeCallbackFunc cb;     /* UT hash map value.             */
  UT_hash_handle hh;            /* Makes this structure hashable. */
};

/* CGstatecallbackMap - UT hash map for state callbacks. */

struct CGstatecallbackMap
{
  size_t state;                 /* UT hash map key.               */
  CGstatecallback cb;           /* UT hash map value.             */
  UT_hash_handle hh;            /* Makes this structure hashable. */
};

/*
 * TraceCgState - Cg runtime state for trace purposes.
 *
 * Maintain a mapping between context and include callbacks.
 *
 * Maintain a mapping between state and state callbacks.
 */

struct TraceCgState
{
  struct CGIncludeCallbackFuncMap *CompilerIncludeMap;

  struct CGstatecallbackMap *StateSetMap;
  struct CGstatecallbackMap *StateResetMap;
  struct CGstatecallbackMap *StateValidateMap;
};

/* Lookup CGIncludeCallbackFunc for context. */

CGIncludeCallbackFunc
lookupCGIncludeCallbackFunc(const struct CGIncludeCallbackFuncMap *map, CGcontext context)
{
  if (context)
  {
    struct CGIncludeCallbackFuncMap *i = NULL;
    size_t key = (size_t) context;

    HASH_FIND(hh, map, &key, sizeof(size_t), i);
    return i ? i->cb : NULL;
  }
  return NULL;
}

/* Update CGIncludeCallbackFunc for context. */

int
updateCGIncludeCallbackFunc(struct CGIncludeCallbackFuncMap ** const map, CGcontext context, CGIncludeCallbackFunc cb)
{
  assert(map);

  /* Remove current mapping (if any) for context. */

  if (context)
  {
    struct CGIncludeCallbackFuncMap *i = NULL;
    size_t key = (size_t) context;

    HASH_FIND(hh, *map, &key, sizeof(size_t), i);
    if (i)
    {
      HASH_DEL(*map, i);
      free(i);
    }
  }

  /* Add new mapping for context. */

  if (context && cb)
  {
    struct CGIncludeCallbackFuncMap *i = malloc(sizeof(struct CGIncludeCallbackFuncMap));
    if (!i)
      return 0;

    i->context = (size_t) context;
    i->cb = cb;

    HASH_ADD(hh, *map, context, sizeof(size_t), i);
    return 1;
  }

  return !context && !cb;
}

/* Lookup CGstatecallback for state. */

CGstatecallback
lookupCGstatecallback(const struct CGstatecallbackMap *map, CGstate state)
{
  if (state)
  {
    struct CGstatecallbackMap *i = NULL;
    size_t key = (size_t) state;

    HASH_FIND(hh, map, &key, sizeof(size_t), i);
    return i ? i->cb : NULL;
  }
  return NULL;
}

/* Update CGstatecallback for state. */

int
updateCGstatecallback(struct CGstatecallbackMap ** const map, CGstate state, CGstatecallback cb)
{
  assert(map);

  /* Remove current mapping (if any) for state. */

  if (state)
  {
    struct CGstatecallbackMap *i = NULL;
    size_t key = (size_t) state;

    HASH_FIND(hh, *map, &key, sizeof(size_t), i);
    if (i)
    {
      HASH_DEL(*map, i);
      free(i);
    }
  }

  /* Add new mapping for state. */

  if (state && cb)
  {
    struct CGstatecallbackMap *i = malloc(sizeof(struct CGstatecallbackMap));
    if (!i)
      return 0;

    i->state = (size_t) state;
    i->cb = cb;

    HASH_ADD(hh, *map, state, sizeof(size_t), i);
    return 1;
  }

  return !state && !cb;
}

#endif
