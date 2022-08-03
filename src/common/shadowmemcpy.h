#include "stddef.h"

#define SHADOW_STEP (0x10000L)
#define SHADOW_ALLOC (1L<<24)
#define SHADOW_CYCLE (64L)
#define SHADOW_CL (64L)
#define SHADOW_CLM1 (SHADOW_CL-1)

inline void *shadow_step(void *p, size_t sets)
{
  uintptr_t ip = (uintptr_t) p;
  uintptr_t set =  ip & ((SHADOW_CYCLE - 1) * SHADOW_CL);   // should be 0x3f0000
  uintptr_t step =  ip & ((SHADOW_CYCLE - 1) * SHADOW_STEP);  // should be 0xfc0
  ip = ip & ~(0x3fffffL);
  sets = sets * SHADOW_CL;   // should compile to << 6
  set = set + SHADOW_CL;
  if (set >= sets) {
    set = 0;
    step = step + SHADOW_STEP;
    if (step >= (SHADOW_STEP*SHADOW_CYCLE)) {
      step = 0;
    }
  }
  return((void *) (ip + step + set));
}

/* these return the next pointer to use */
inline void *shadow_load(void *p, size_t sets)
{
  long int temp =  *((volatile long int *) p);
  return (shadow_step(p, sets));
}

inline void *shadow_store(void *p, size_t sets)
{
  *((long int *) p) = 0L;
  return (shadow_step(p, sets));
}

  
void *shadow_alloc();

/* these return the next pointer to use */
/* the shadow_w calls use shadow ways scheme */
/* the shadow_b calls use shadow big scheme */

/* All these accept the current shadow pointer and return the new one */
/* memcpy uses shadow load to load and mov to store */
void *shadow_memcpy(void *dst, void *src, size_t count, size_t sets);
/* All these accept the current shadow pointer and return the new one */
/* memcpy_r uses mov to load and shadow_load to "store" */
void *shadow_memcpy_r(void *dst, void *src, size_t count, size_t sets);

void *shadow_load_block(void *p, size_t length, size_t sets);
void *shadow_store_block(void *p, size_t length, size_t sets);

void shadow_pushtol3(void *p, size_t length);

