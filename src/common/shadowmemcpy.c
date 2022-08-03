#define _GNU_SOURCE             /* See feature_test_macros(7) */
#include "movget.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>
#include "shadowmemcpy.h"
#include "bigbuf.h"
#include "movgetmemcpy.h"
#include <assert.h>
#include <amfunctions.h>

// define this to print address traces
//#define PRINTLOG 1

void *shadow_alloc()
{
  void *shadow;
  bigbuf_init();

  shadow = bigbuf_alloc(SHADOW_ALLOC, SHADOW_ALLOC);
  assert(shadow != NULL);
  /*   memset(shadow, 0, SHADOW_ALLOC); handled by bigbuf */
  return(shadow);
}

void * shadow_load_block(void *buf, size_t size, size_t sets)
{
  uintptr_t p = (uintptr_t) buf;
  uintptr_t set =  p & ((SHADOW_CYCLE - 1) * SHADOW_CL);   // should be 0x3f0000
  uintptr_t step =  p & ((SHADOW_CYCLE - 1) * SHADOW_STEP);  // should be 0xfc0
  sets = sets * SHADOW_CL;   // should compile to << 6
  uint64_t temp;
  p = p & ~(0x3fffffL);
  while (size > 0) {
    //printf("%p\n", (char *) (p + step + set));
    temp = *((volatile uint64_t *)(p + step + set));
    set = set + SHADOW_CL;
    if (set >= sets) {
      set = 0;
      step = step + SHADOW_STEP;
      if (step >= (SHADOW_STEP*SHADOW_CYCLE)) {
	step = 0;
      }
    }
    size = size - SHADOW_CL;
  }
  return((void *) (p + step + set));
}

void * shadow_store_block(void *buf, size_t size, size_t sets)
{
  uintptr_t p = (uintptr_t) buf;
  uintptr_t set =  p & ((SHADOW_CYCLE - 1) * SHADOW_CL);   // should be 0x3f0000
  uintptr_t step =  p & ((SHADOW_CYCLE - 1) * SHADOW_STEP);  // should be 0xfc0
  sets = sets * SHADOW_CL;   // should compile to << 6
  p = p & ~(0x3fffffL);
  while (size > 0) {
    //printf("%p\n", (char *) (p + step + set));
    *((volatile uint64_t *)(p + step + set)) = 0;
    set = set + SHADOW_CL;
    if (set >= sets) {
      set = 0;
      step = step + SHADOW_STEP;
      if (step >= (SHADOW_STEP*SHADOW_CYCLE)) {
	step = 0;
      }
    }
    size = size - SHADOW_CL;
  }
  return((void *) (p + step + set));
}

void *shadow_memcpy(void *dst, void *src, size_t n, size_t sets)
{
  while (n > SHADOW_CLM1) {
    //printf("%p\n", src);
    __m512i temp = _mm512_load_epi32((__m512i *) src);
    src = shadow_step(src, sets);
    _mm512_storeu_si512(dst, temp);
    n -= SHADOW_CL;
    dst = (__m512i *) ((uintptr_t) dst + SHADOW_CL);
  }
  return(src);
}

void *shadow_memcpy_r(void *dst, void *src, size_t n, size_t sets)
{
  while (n > SHADOW_CLM1) {
    //printf("%p\n", src);
    __m512i temp = _mm512_load_epi32((__m512i *) src);
    _mm512_storeu_si512(dst, temp);
    dst = shadow_step(dst, sets);
    n -= SHADOW_CL;
    src = (__m512i *) ((uintptr_t) src + SHADOW_CL);
  }
  return(src);
}

void shadow_pushtol3(void *p, size_t length)
{
  uint64_t data = 0;
  for (unsigned offset = 0; offset < length; offset += SHADOW_CL){
    for (unsigned loop = 0; loop < 2; loop += 1) {
      for (unsigned stride = 0; stride < (SHADOW_STEP * SHADOW_CYCLE); stride += SHADOW_STEP) {
	//printf("%p\n", &((uint64_t *) p)[(stride+offset ) / sizeof(uint64_t)]);
	data += ((volatile uint64_t *) p)[(stride+offset ) / sizeof(uint64_t)];
      }
    }
  }
}

