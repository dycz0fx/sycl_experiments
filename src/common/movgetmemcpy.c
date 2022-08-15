#include "movget.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>

/* movgetmemcpy(void *dst, void *src, size_t count);
 *
 * Algorithm:
 * 1) align the source by using an aligned movget to a tmp then store the suffix
 * 2) use movget64b until there are less than 64 bytes remaining
 *    this will have unaligned stores
 * 3) use an aligned movget of the partial last block, to a temp, and store the prefix 
 */







void print64(void *src)
{
  uint64_t x;
  for (int qw=0; qw < 8; qw += 1) {
    x = 0;
    for (int b = 0; b < 8; b += 1)
      x |= ((uint64_t) (0xff & *((char *) src + (qw * 8) + b))) << (b * 8);
    printf(" %016lx", x);
  }
  printf("\n");
}

#define CL 64L
#define CLM1 (CL-1)

#define TEST 0
#include <assert.h>
#if TEST
void testmg64b(__m512i *dest, __m512i *src)
{
  assert(((uintptr_t) src & CLM1)==0);
  //printf("mg64b dst %p src %p", dest, src);
  //print64(src);
  memmove(dest, src, 64);  // use memmove due to overlapping in code below
}

#define MG64B testmg64b
#define MG512 _mm512_load_epi32
#else
#define MG64B _movget64b
#define MG512 _mm512_movget_load_m512i
#endif


void movget_memcpy(void *dst, void *src, size_t count)
{
  uintptr_t src_offset = (uintptr_t) src & CLM1;
#if 1
  assert(src_offset == 0);
#else
  /* FIX! should use byte pointers instead of intptr_t, but compiler complains! */
  if (src_offset != 0) {
    size_t  bytes = CL - src_offset;
    if (count < bytes) bytes = count;
    __m512i temp = MG512((__m512i *) ((uintptr_t) src & ~CLM1));
    _mm512_mask_storeu_epi8((void *) ((uintptr_t) dst - src_offset), ((1L << bytes) - 1) << src_offset, temp);
    count -= bytes;
    if (count == 0) return;
    src = (__m512i *) ((uintptr_t) src + bytes);
    dst = (__m512i *) ((uintptr_t) dst + bytes);
    assert(((uintptr_t) src & CLM1) == 0);
  }
#endif
  __m512i *ss = (__m512i *)src;
  __m512i *dd = (__m512i *)dst;
  while (count >= CL) {
    __m512i temp = _mm512_movget_load_m512i(ss);
    _mm512_storeu_si512(dd, temp);
    count -= CL;
    ss++; dd++;
  }
  if (count > 0) {
    __m512i temp = MG512(ss);
    _mm512_mask_storeu_epi8((void *) dd, (1L << count) - 1, temp);
  }
 }

void avx_memcpy(void *dst, void *src, size_t count)
{
  uintptr_t src_offset = (uintptr_t) src & CLM1;
#if 1
  assert(src_offset == 0);
#else
  /* FIX! should use byte pointers instead of intptr_t, but compiler complains! */
  if (src_offset != 0) {
    size_t  bytes = CL - src_offset;
    if (count < bytes) bytes = count;
    __m512i temp = _mm512_load_epi32((__m512i *) ((uintptr_t) src & ~CLM1));
    _mm512_mask_storeu_epi8((void *) ((uintptr_t) dst - src_offset), ((1L << bytes) - 1) << src_offset, temp);
    count -= bytes;
    if (count == 0) return;
    src = (__m512i *) ((uintptr_t) src + bytes);
    dst = (__m512i *) ((uintptr_t) dst + bytes);
  }
#endif
  __m512i *ss = (__m512i *)src;
  __m512i *dd = (__m512i *)dst;
  while (count >= CL) {
    __m512i temp = _mm512_load_epi32(ss);
    _mm512_storeu_si512(dd, temp);
    count -= CL;
    ss++; dd++;
  }
  if (count > 0) {
    __m512i temp = _mm512_load_epi32(ss);
    _mm512_mask_storeu_epi8((void *) dd, (1L << count) - 1, temp);
  }
 }

/*************************************
 Case 1 count < 64 bytes
     cannot use 64 byte stores
     movget64b the 1 or 2 source cachelines to a temp buffer
     call library memcpy
*/

/*

 Case 2 count >= 64  bytes,  2 cachelines in src buffer
     movget64b both cachelines to tmp buffer
     load unaligned and store unaligned the beginning
     if count > 64 load unaligned and store unaligned end
*/

/*
 Case 3 Source buffer contains 3 or more cachelines
     movget64b first cacheline to aligned tmp
     load aligned, store unaligned
     movget64b  last cacheline to aligned tmp
     load aligned, store unaligned
     loop movget64b the middle
*/

void movget_memcpy_noavx(void *dst, void *src, size_t count)
{
  __m512i tmp[3];
  __m512i *firstcl = (__m512i *) ((uintptr_t) src & ~CLM1);
  uintptr_t src_offset = (uintptr_t) src & CLM1;
  if (count < CL) {
    MG64B((void *) ((uintptr_t) &tmp[1] - src_offset), firstcl);
    if (src_offset + count > CL) 
      MG64B((void *) ((uintptr_t) &tmp[2] - src_offset), firstcl+1);
    //lfence();  // not needed because memcopy load follows ordinary store
    memcpy(dst, (void *) &tmp[1], count);
    return;
  }
  /* first bit from src buffer becomes tmp[1]
   */
  __m512i *lastcl = (__m512i *) (((uintptr_t) src + count) & ~CLM1);
  if ((firstcl + 1) == lastcl) {
    //printf("firstcl %p lastcl %p count %lu\n", firstcl, lastcl, count);
    MG64B((void *) ((uintptr_t) &tmp[1] - src_offset), firstcl);
    MG64B((void *) ((uintptr_t) &tmp[2] - src_offset), lastcl);
    // lfence(); // not needed  because address conflict
    MG64B(dst, &tmp[1]);
    if (count > CL) {
      MG64B((void *) ((uintptr_t) &tmp[1] - (count - CL)), &tmp[1]);
      MG64B((void *) ((uintptr_t) &tmp[2] - (count - CL)), &tmp[2]);
      /* above two MG64B have overlapping source and destinations */

      MG64B((void *) ((uintptr_t) dst + (count - CL)), &tmp[1]);
    }
    return;
  }
  /* if src is not aligned, load the CL containing the beginning into tmp
     and then store the 64 bytes containing the first partial data to dst */
  if (src_offset != 0) {
    MG64B((void *) ((uintptr_t) &tmp[1] - src_offset), (__m512i *) firstcl);
    MG64B(dst, &tmp[1]);
    dst = (void *) ((uintptr_t) dst + CL - src_offset);
    count -= CL - src_offset;
    src = (void *) ((uintptr_t) src + CL - src_offset); /* now src is aligned */
  }
  /* if the end of src is not aligned, load the last CL into tmp
     and then store the 64 bytes ending src into the end of dst
   */
  uintptr_t valid_bytes_lastcl = ((uintptr_t) src + count) & CLM1;
  if (valid_bytes_lastcl != 0) {
    uintptr_t dst_end_offset = ((uintptr_t) dst + count) & CLM1;
    MG64B((void *) ((uintptr_t) &tmp[1] - valid_bytes_lastcl), lastcl);
    MG64B((void *) ((uintptr_t) dst + count - CL), &tmp[0]);
    count -= valid_bytes_lastcl;
  }
  //printf("count %lu valid_bytes_lastcl %lu\n", count, valid_bytes_lastcl);
  assert((count & CLM1) == 0);
  while (count > 0) {
    MG64B(dst, (__m512i *) src);
    count -= CL;
    src = (__m512i *) ((uintptr_t) src + CL);
    dst = (__m512i *) ((uintptr_t) dst + CL);
  }

 }

