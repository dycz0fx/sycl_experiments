/* spinbarrier.h */

#ifndef SPINBARRIER_H
#define SPINBARRIER_H

#ifndef CACHELINE
#define CACHELINE 64
#endif

struct SpinBarrier {
  volatile uint64_t count;
  uint64_t barrier_size;
  uint64_t pad1[6];
  volatile uint64_t passed;
  uint64_t pad2[7];
} __attribute__ ((aligned(CACHELINE)));

typedef struct SpinBarrier SpinBarrier_t;

/* Design
 * 
 * Atomic add instructions count entries to the barrier
 * threads wait for the "passed" sequence number to change
 * The last thread in sets the counter back to 0, synchronizes
 * to make sure it happens, then changes the sequence number
 */

static inline void SpinBarrier_Init(SpinBarrier_t *p, uint64_t size)
{
  p->count = 0;
  p->passed = 0;
  p->barrier_size = size;
}

static inline void SpinBarrier_Wait(SpinBarrier_t *p)
{
  uint64_t old_passed = p->passed;
  uint64_t val = __sync_fetch_and_add(&p->count, 1);
  if (val == (p->barrier_size - 1)) {
    p->count = 0;
    __sync_synchronize();
    p->passed = old_passed + 1;
    __sync_synchronize();
  } else {
    while (p->passed == old_passed) _mm_pause();
  }
}

static inline void SpinBarrier_SetSize(SpinBarrier_t *p, uint64_t size)
{
  p->barrier_size = size;
}


#endif
