/* movget.c
 * block load functions using different movget instructions
 */

#include "immintrin.h"

// stop complaints about unused values in the load functions

//#pragma warning disable 593
// #pragma warning disable 869



#define LOAD_MG512A(reg, p) \
  __asm__ volatile ("vmovget %1, %0\n": "=x"(reg): "m"(p))
#define LOAD_MG256A(reg, p) \
  __asm__ volatile ("vmovget %1, %0\n": "=x"(reg): "m"(p))
#define LOAD_MG128A(reg, p) \
  __asm__ volatile ("vmovget %1, %0\n": "=x"(reg): "m"(p))

#define LOAD_MG512(reg, p) reg = _mm512_movget_load_m512i(p)
#define LOAD_MG256(reg, p) reg = _mm256_movget_load_m256i(p)
#define LOAD_MG128(reg, p) reg = _mm_movget_load_m128i(p)

void raw_load_movget_512(__m512i *src, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOAD_MG512(r0, &p[0]);
    LOAD_MG512(r1, &p[1]);
    LOAD_MG512(r2, &p[2]);
    LOAD_MG512(r3, &p[3]);
    LOAD_MG512(r4, &p[4]);
    LOAD_MG512(r5, &p[5]);
    LOAD_MG512(r6, &p[6]);
    LOAD_MG512(r7, &p[7]);
    LOAD_MG512(r8, &p[8]);
    LOAD_MG512(r9, &p[9]);
    LOAD_MG512(ra, &p[10]);
    LOAD_MG512(rb, &p[11]);
    LOAD_MG512(rc, &p[12]);
    LOAD_MG512(rd, &p[13]);
    LOAD_MG512(re, &p[14]);
    LOAD_MG512(rf, &p[15]);
  }
}


void raw_load_movget_256(__m256i *src, size_t length)
{
  __m256i r0, r1, r2, r3, r4, r5, r6, r7;
  __m256i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 5;
  for (__m256i *p = src; p < src+length; p += 16) {
    LOAD_MG256(r0, &p[0]);
    LOAD_MG256(r1, &p[1]);
    LOAD_MG256(r2, &p[2]);
    LOAD_MG256(r3, &p[3]);
    LOAD_MG256(r4, &p[4]);
    LOAD_MG256(r5, &p[5]);
    LOAD_MG256(r6, &p[6]);
    LOAD_MG256(r7, &p[7]);
    LOAD_MG256(r8, &p[8]);
    LOAD_MG256(r9, &p[9]);
    LOAD_MG256(ra, &p[10]);
    LOAD_MG256(rb, &p[11]);
    LOAD_MG256(rc, &p[12]);
    LOAD_MG256(rd, &p[13]);
    LOAD_MG256(re, &p[14]);
    LOAD_MG256(rf, &p[15]);
  }
}

void raw_load_movget_128(__m128i *src, size_t length)
{
  __m128i r0, r1, r2, r3, r4, r5, r6, r7;
  __m128i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 4;
  for (__m128i *p = src; p < src+length; p += 16) {
    LOAD_MG128(r0, &p[0]);
    LOAD_MG128(r1, &p[1]);
    LOAD_MG128(r2, &p[2]);
    LOAD_MG128(r3, &p[3]);
    LOAD_MG128(r4, &p[4]);
    LOAD_MG128(r5, &p[5]);
    LOAD_MG128(r6, &p[6]);
    LOAD_MG128(r7, &p[7]);
    LOAD_MG128(r8, &p[8]);
    LOAD_MG128(r9, &p[9]);
    LOAD_MG128(ra, &p[10]);
    LOAD_MG128(rb, &p[11]);
    LOAD_MG128(rc, &p[12]);
    LOAD_MG128(rd, &p[13]);
    LOAD_MG128(re, &p[14]);
    LOAD_MG128(rf, &p[15]);
  }
}

void raw_load_movget_512A(__m512i *src, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOAD_MG512A(r0, p[0]);
    LOAD_MG512A(r1, p[1]);
    LOAD_MG512A(r2, p[2]);
    LOAD_MG512A(r3, p[3]);
    LOAD_MG512A(r4, p[4]);
    LOAD_MG512A(r5, p[5]);
    LOAD_MG512A(r6, p[6]);
    LOAD_MG512A(r7, p[7]);
    LOAD_MG512A(r8, p[8]);
    LOAD_MG512A(r9, p[9]);
    LOAD_MG512A(ra, p[10]);
    LOAD_MG512A(rb, p[11]);
    LOAD_MG512A(rc, p[12]);
    LOAD_MG512A(rd, p[13]);
    LOAD_MG512A(re, p[14]);
    LOAD_MG512A(rf, p[15]);
  }
}


void raw_load_movget_256A(__m256i *src, size_t length)
{
  __m256i r0, r1, r2, r3, r4, r5, r6, r7;
  __m256i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 5;
  for (__m256i *p = src; p < src+length; p += 16) {
    LOAD_MG256A(r0, p[0]);
    LOAD_MG256A(r1, p[1]);
    LOAD_MG256A(r2, p[2]);
    LOAD_MG256A(r3, p[3]);
    LOAD_MG256A(r4, p[4]);
    LOAD_MG256A(r5, p[5]);
    LOAD_MG256A(r6, p[6]);
    LOAD_MG256A(r7, p[7]);
    LOAD_MG256A(r8, p[8]);
    LOAD_MG256A(r9, p[9]);
    LOAD_MG256A(ra, p[10]);
    LOAD_MG256A(rb, p[11]);
    LOAD_MG256A(rc, p[12]);
    LOAD_MG256A(rd, p[13]);
    LOAD_MG256A(re, p[14]);
    LOAD_MG256A(rf, p[15]);
  }
}

void raw_load_movget_128A(__m128i *src, size_t length)
{
  __m128i r0, r1, r2, r3, r4, r5, r6, r7;
  __m128i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 4;
  for (__m128i *p = src; p < src+length; p += 16) {
    LOAD_MG128A(r0, p[0]);
    LOAD_MG128A(r1, p[1]);
    LOAD_MG128A(r2, p[2]);
    LOAD_MG128A(r3, p[3]);
    LOAD_MG128A(r4, p[4]);
    LOAD_MG128A(r5, p[5]);
    LOAD_MG128A(r6, p[6]);
    LOAD_MG128A(r7, p[7]);
    LOAD_MG128A(r8, p[8]);
    LOAD_MG128A(r9, p[9]);
    LOAD_MG128A(ra, p[10]);
    LOAD_MG128A(rb, p[11]);
    LOAD_MG128A(rc, p[12]);
    LOAD_MG128A(rd, p[13]);
    LOAD_MG128A(re, p[14]);
    LOAD_MG128A(rf, p[15]);
  }
}

/* in contrast, using _mm512_load_si512() or vmovaps does this: */


#define LOAD(reg,p) reg = _mm512_load_si512(p)

#define LOADA(reg,p) \
  __asm__ volatile ("vmovaps %1, %0\n": "=x"(reg): "m"(p))

void raw_load_mov_512(__m512i *src, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOAD(r0, &p[0]);
    LOAD(r1, &p[1]);
    LOAD(r2, &p[2]);
    LOAD(r3, &p[3]);
    LOAD(r4, &p[4]);
    LOAD(r5, &p[5]);
    LOAD(r6, &p[6]);
    LOAD(r7, &p[7]);
    LOAD(r8, &p[8]);
    LOAD(r9, &p[9]);
    LOAD(ra, &p[10]);
    LOAD(rb, &p[11]);
    LOAD(rc, &p[12]);
    LOAD(rd, &p[13]);
    LOAD(re, &p[14]);
    LOAD(rf, &p[15]);
  }
}

void raw_load_mov_512A(__m512i *src, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOADA(r0, p[0]);
    LOADA(r1, p[1]);
    LOADA(r2, p[2]);
    LOADA(r3, p[3]);
    LOADA(r4, p[4]);
    LOADA(r5, p[5]);
    LOADA(r6, p[6]);
    LOADA(r7, p[7]);
    LOADA(r8, p[8]);
    LOADA(r9, p[9]);
    LOADA(ra, p[10]);
    LOADA(rb, p[11]);
    LOADA(rc, p[12]);
    LOADA(rd, p[13]);
    LOADA(re, p[14]);
    LOADA(rf, p[15]);
  }
}
