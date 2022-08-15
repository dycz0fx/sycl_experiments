/* amfunctions.c
 * A Library of useful active message functions
 */

#include "amfunctions.h"
#include "misc.h"
#include <string.h>
#include <stdio.h>
#include "cpu_affinity.h"
#include "movget.h"

// stop complaints about unused values in the load functions

#pragma warning disable 593
#pragma warning disable 869

#define LOAD(reg,p) \
  __asm__ volatile ("vmovaps %1, %0\n": "=x"(reg): "m"(p));

#define LOADNT(reg,p) \
  __asm__ volatile ("vmovntdqa %1, %0\n": "=x"(reg): "m"(p));

#define LOAD8(reg, p) \
  reg = *((volatile uint64_t *) (p));

#define STORE(p,reg)						\
  __asm__ volatile ("vmovaps %0, %1\n": "=x"(reg): "m"(p));

#define STORENT(p,reg)						\
  __asm__ volatile ("vmovntdq %0, %1\n": "=x"(reg): "m"(p));


void raw_nopmov(__m512i *src, size_t length)
{
}

void raw_load1of8(uint64_t *src, size_t length)
{
  uint64_t r0, r1, r2, r3, r4, r5, r6, r7;
  uint64_t r8, r9, ra, rb, rc, rd, re, rf;
  __m512i *p = (__m512i *) src;
  __m512i *end = p + (length >> 6);
  //printf("raw_load cpu %d (%p, %ld)\n", get_cpu(), p, length);
  for (; p < end; p += 16) {
    LOAD8(r0, &p[0]);
    LOAD8(r1, &p[1]);
    LOAD8(r2, &p[2]);
    LOAD8(r3, &p[3]);
    LOAD8(r4, &p[4]);
    LOAD8(r5, &p[5]);
    LOAD8(r6, &p[6]);
    LOAD8(r7, &p[7]);
    LOAD8(r8, &p[8]);
    LOAD8(r9, &p[9]);
    LOAD8(ra, &p[10]);
    LOAD8(rb, &p[11]);
    LOAD8(rc, &p[12]);
    LOAD8(rd, &p[13]);
    LOAD8(re, &p[14]);
    LOAD8(rf, &p[15]);
  }
}

void raw_fill1of8(uint64_t *dest, size_t length)
{
  //printf("raw_fill cpu %d (%p, %ld)\n", get_cpu(), dest, length);
  length = length >> 6;
  for (__m512i *p = (__m512i *) dest; p < ((__m512i *) dest)+length; p += 16) {
    *((uint64_t *) (&p[0])) = 0;
    *((uint64_t *) (&p[1])) = 0;
    *((uint64_t *) (&p[2])) = 0;
    *((uint64_t *) (&p[3])) = 0;
    *((uint64_t *) (&p[4])) = 0;
    *((uint64_t *) (&p[5])) = 0;
    *((uint64_t *) (&p[6])) = 0;
    *((uint64_t *) (&p[7])) = 0;
    *((uint64_t *) (&p[8])) = 0;
    *((uint64_t *) (&p[9])) = 0;
    *((uint64_t *) (&p[10])) = 0;
    *((uint64_t *) (&p[11])) = 0;
    *((uint64_t *) (&p[12])) = 0;
    *((uint64_t *) (&p[13])) = 0;
    *((uint64_t *) (&p[14])) = 0;
    *((uint64_t *) (&p[15])) = 0;
  }
  //memset(cmd->dest, 0, cmd->length);
}


void raw_fillwithprefetch(__m512i *dest, size_t length)
{
  __m512i r0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
  int dist = 16;
  //printf("raw_fill cpu %d (%p, %ld)\n", get_cpu(), dest, length);
  length = length >> 6;
  for (__m512i *p = dest; p < dest+length; p += 16) {
    _mm_prefetch((const char *) (&p[dist + 0]), _MM_HINT_T0);
    _mm512_store_si512(&p[0], r0);
    _mm_prefetch((const char *) (&p[dist + 1]), _MM_HINT_T0);
    _mm512_store_si512(&p[1], r0);
    _mm_prefetch((const char *) (&p[dist + 1]), _MM_HINT_T0);
    _mm512_store_si512(&p[2], r0);
    _mm_prefetch((const char *) (&p[dist + 3]), _MM_HINT_T0);
    _mm512_store_si512(&p[3], r0);
    _mm_prefetch((const char *) (&p[dist + 4]), _MM_HINT_T0);
    _mm512_store_si512(&p[4], r0);
    _mm_prefetch((const char *) (&p[dist + 5]), _MM_HINT_T0);
    _mm512_store_si512(&p[5], r0);
    _mm_prefetch((const char *) (&p[dist + 6]), _MM_HINT_T0);
    _mm512_store_si512(&p[6], r0);
    _mm_prefetch((const char *) (&p[dist + 7]), _MM_HINT_T0);
    _mm512_store_si512(&p[7], r0);
    _mm_prefetch((const char *) (&p[dist + 8]), _MM_HINT_T0);
    _mm512_store_si512(&p[8], r0);
    _mm_prefetch((const char *) (&p[dist + 9]), _MM_HINT_T0);
    _mm512_store_si512(&p[9], r0);
    _mm_prefetch((const char *) (&p[dist + 10]), _MM_HINT_T0);
    _mm512_store_si512(&p[10], r0);
    _mm_prefetch((const char *) (&p[dist + 11]), _MM_HINT_T0);
    _mm512_store_si512(&p[11], r0);
    _mm_prefetch((const char *) (&p[dist + 12]), _MM_HINT_T0);
    _mm512_store_si512(&p[12], r0);
    _mm_prefetch((const char *) (&p[dist + 13]), _MM_HINT_T0);
    _mm512_store_si512(&p[13], r0);
    _mm_prefetch((const char *) (&p[dist + 14]), _MM_HINT_T0);
    _mm512_store_si512(&p[14], r0);
    _mm_prefetch((const char *) (&p[dist + 15]), _MM_HINT_T0);
    _mm512_store_si512(&p[15], r0);
  }
  //memset(cmd->dest, 0, cmd->length);
}

void raw_fillwithprefetchw(__m512i *dest, size_t length)
{
  __m512i r0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
  int dist = 16;
  //printf("raw_fill cpu %d (%p, %ld)\n", get_cpu(), dest, length);
  length = length >> 6;
  for (__m512i *p = dest; p < dest+length; p += 16) {
    _mm_prefetch((const char *) (&p[dist + 0]), _MM_HINT_ET0);
    _mm512_store_si512(&p[0], r0);
    _mm_prefetch((const char *) (&p[dist + 1]), _MM_HINT_ET0);
    _mm512_store_si512(&p[1], r0);
    _mm_prefetch((const char *) (&p[dist + 1]), _MM_HINT_ET0);
    _mm512_store_si512(&p[2], r0);
    _mm_prefetch((const char *) (&p[dist + 3]), _MM_HINT_ET0);
    _mm512_store_si512(&p[3], r0);
    _mm_prefetch((const char *) (&p[dist + 4]), _MM_HINT_ET0);
    _mm512_store_si512(&p[4], r0);
    _mm_prefetch((const char *) (&p[dist + 5]), _MM_HINT_ET0);
    _mm512_store_si512(&p[5], r0);
    _mm_prefetch((const char *) (&p[dist + 6]), _MM_HINT_ET0);
    _mm512_store_si512(&p[6], r0);
    _mm_prefetch((const char *) (&p[dist + 7]), _MM_HINT_ET0);
    _mm512_store_si512(&p[7], r0);
    _mm_prefetch((const char *) (&p[dist + 8]), _MM_HINT_ET0);
    _mm512_store_si512(&p[8], r0);
    _mm_prefetch((const char *) (&p[dist + 9]), _MM_HINT_ET0);
    _mm512_store_si512(&p[9], r0);
    _mm_prefetch((const char *) (&p[dist + 10]), _MM_HINT_ET0);
    _mm512_store_si512(&p[10], r0);
    _mm_prefetch((const char *) (&p[dist + 11]), _MM_HINT_ET0);
    _mm512_store_si512(&p[11], r0);
    _mm_prefetch((const char *) (&p[dist + 12]), _MM_HINT_ET0);
    _mm512_store_si512(&p[12], r0);
    _mm_prefetch((const char *) (&p[dist + 13]), _MM_HINT_ET0);
    _mm512_store_si512(&p[13], r0);
    _mm_prefetch((const char *) (&p[dist + 14]), _MM_HINT_ET0);
    _mm512_store_si512(&p[14], r0);
    _mm_prefetch((const char *) (&p[dist + 15]), _MM_HINT_ET0);
    _mm512_store_si512(&p[15], r0);
  }
  //memset(cmd->dest, 0, cmd->length);
}

void raw_fill(__m512i *dest, size_t length)
{
  __m512i r0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
  //printf("raw_fill cpu %d (%p, %ld)\n", get_cpu(), dest, length);
  length = length >> 6;
  for (__m512i *p = dest; p < dest+length; p += 16) {
    _mm512_store_si512(&p[0], r0);
    _mm512_store_si512(&p[1], r0);
    _mm512_store_si512(&p[2], r0);
    _mm512_store_si512(&p[3], r0);
    _mm512_store_si512(&p[4], r0);
    _mm512_store_si512(&p[5], r0);
    _mm512_store_si512(&p[6], r0);
    _mm512_store_si512(&p[7], r0);
    _mm512_store_si512(&p[8], r0);
    _mm512_store_si512(&p[9], r0);
    _mm512_store_si512(&p[10], r0);
    _mm512_store_si512(&p[11], r0);
    _mm512_store_si512(&p[12], r0);
    _mm512_store_si512(&p[13], r0);
    _mm512_store_si512(&p[14], r0);
    _mm512_store_si512(&p[15], r0);
  }
  //memset(cmd->dest, 0, cmd->length);
}

void raw_fillwithmovnt(__m512i *dest, size_t length)
{
  __m512i r0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
  //printf("raw_fill cpu %d (%p, %ld)\n", get_cpu(), dest, length);
  length = length >> 6;
  for (__m512i *p = dest; p < dest+length; p += 16) {
    _mm512_stream_si512(&p[0], r0);
    _mm512_stream_si512(&p[1], r0);
    _mm512_stream_si512(&p[2], r0);
    _mm512_stream_si512(&p[3], r0);
    _mm512_stream_si512(&p[4], r0);
    _mm512_stream_si512(&p[5], r0);
    _mm512_stream_si512(&p[6], r0);
    _mm512_stream_si512(&p[7], r0);
    _mm512_stream_si512(&p[8], r0);
    _mm512_stream_si512(&p[9], r0);
    _mm512_stream_si512(&p[10], r0);
    _mm512_stream_si512(&p[11], r0);
    _mm512_stream_si512(&p[12], r0);
    _mm512_stream_si512(&p[13], r0);
    _mm512_stream_si512(&p[14], r0);
    _mm512_stream_si512(&p[15], r0);
  }
  //memset(cmd->dest, 0, cmd->length);
}

void raw_fillwithmovnt1of8(__m512i *dest, size_t length)
{
  __m64 r0 = (__m64) 0L;
  //printf("raw_fill cpu %d (%p, %ld)\n", get_cpu(), dest, length);
  length = length >> 6;
  for (__m512i *p = dest; p < dest+length; p += 16) {
    _mm_stream_pi((__m64 *) &p[0], r0);
    _mm_stream_pi((__m64 *) &p[1], r0);
    _mm_stream_pi((__m64 *) &p[2], r0);
    _mm_stream_pi((__m64 *) &p[3], r0);
    _mm_stream_pi((__m64 *) &p[4], r0);
    _mm_stream_pi((__m64 *) &p[5], r0);
    _mm_stream_pi((__m64 *) &p[6], r0);
    _mm_stream_pi((__m64 *) &p[7], r0);
    _mm_stream_pi((__m64 *) &p[8], r0);
    _mm_stream_pi((__m64 *) &p[9], r0);
    _mm_stream_pi((__m64 *) &p[10], r0);
    _mm_stream_pi((__m64 *) &p[11], r0);
    _mm_stream_pi((__m64 *) &p[12], r0);
    _mm_stream_pi((__m64 *) &p[13], r0);
    _mm_stream_pi((__m64 *) &p[14], r0);
    _mm_stream_pi((__m64 *) &p[15], r0);
  }
  //memset(cmd->dest, 0, cmd->length);
}

void raw_fillwithmovnt8of8(__m64 *dest, size_t length)
{
  __m64 r0 = (__m64) 0L;
  //printf("raw_fillwithmovnt8of8 cpu %d (%p, %ld)\n", get_cpu(), dest, length);
  length = length >> 3;
  for (__m64 *p = dest; p < dest+length; p += 8) {
    _mm_stream_pi((__m64 *) &p[0], r0);
    _mm_stream_pi((__m64 *) &p[1], r0);
    _mm_stream_pi((__m64 *) &p[2], r0);
    _mm_stream_pi((__m64 *) &p[3], r0);
    _mm_stream_pi((__m64 *) &p[4], r0);
    _mm_stream_pi((__m64 *) &p[5], r0);
    _mm_stream_pi((__m64 *) &p[6], r0);
    _mm_stream_pi((__m64 *) &p[7], r0);
  }
}

void raw_fillwithmovdir64b(__m512i *dest, size_t length)
{
  __m512i r0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
  length = length >> 6;
  for (__m512i *p = dest; p < dest+length; p += 16) {
    _movdir64b(&p[0], &r0);
    _movdir64b(&p[1], &r0);
    _movdir64b(&p[2], &r0);
    _movdir64b(&p[3], &r0);
    _movdir64b(&p[4], &r0);
    _movdir64b(&p[5], &r0);
    _movdir64b(&p[6], &r0);
    _movdir64b(&p[7], &r0);
    _movdir64b(&p[8], &r0);
    _movdir64b(&p[9], &r0);
    _movdir64b(&p[10], &r0);
    _movdir64b(&p[11], &r0);
    _movdir64b(&p[12], &r0);
    _movdir64b(&p[13], &r0);
    _movdir64b(&p[14], &r0);
    _movdir64b(&p[15], &r0);
  }
}

void raw_repload(__m512i *src, size_t length)
{
  int iter = 100;
  while(iter--) raw_load(src, length);
}

void raw_load(__m512i *src, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  //printf("raw_load cpu %d (%p, %ld)\n", get_cpu(), src, length);
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOAD(r0, p[0]);
    LOAD(r1, p[1]);
    LOAD(r2, p[2]);
    LOAD(r3, p[3]);
    LOAD(r4, p[4]);
    LOAD(r5, p[5]);
    LOAD(r6, p[6]);
    LOAD(r7, p[7]);
    LOAD(r8, p[8]);
    LOAD(r9, p[9]);
    LOAD(ra, p[10]);
    LOAD(rb, p[11]);
    LOAD(rc, p[12]);
    LOAD(rd, p[13]);
    LOAD(re, p[14]);
    LOAD(rf, p[15]);
  }
}

void raw_loadnt(__m512i *src, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOADNT(r0, p[0]);
    LOADNT(r1, p[1]);
    LOADNT(r2, p[2]);
    LOADNT(r3, p[3]);
    LOADNT(r4, p[4]);
    LOADNT(r5, p[5]);
    LOADNT(r6, p[6]);
    LOADNT(r7, p[7]);
    LOADNT(r8, p[8]);
    LOADNT(r9, p[9]);
    LOADNT(ra, p[10]);
    LOADNT(rb, p[11]);
    LOADNT(rc, p[12]);
    LOADNT(rd, p[13]);
    LOADNT(re, p[14]);
    LOADNT(rf, p[15]);
  }
}

void raw_copy(__m512i *dest, __m512i *src, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOAD(r0, p[0]);
    LOAD(r1, p[1]);
    LOAD(r2, p[2]);
    LOAD(r3, p[3]);
    LOAD(r4, p[4]);
    LOAD(r5, p[5]);
    LOAD(r6, p[6]);
    LOAD(r7, p[7]);
    LOAD(r8, p[8]);
    LOAD(r9, p[9]);
    LOAD(ra, p[10]);
    LOAD(rb, p[11]);
    LOAD(rc, p[12]);
    LOAD(rd, p[13]);
    LOAD(re, p[14]);
    LOAD(rf, p[15]);
    _mm512_store_si512(&dest[0], r0);
    _mm512_store_si512(&dest[1], r1);
    _mm512_store_si512(&dest[2], r2);
    _mm512_store_si512(&dest[3], r3);
    _mm512_store_si512(&dest[4], r4);
    _mm512_store_si512(&dest[5], r5);
    _mm512_store_si512(&dest[6], r6);
    _mm512_store_si512(&dest[7], r7);
    _mm512_store_si512(&dest[8], r8);
    _mm512_store_si512(&dest[9], r9);
    _mm512_store_si512(&dest[10], ra);
    _mm512_store_si512(&dest[11], rb);
    _mm512_store_si512(&dest[12], rc);
    _mm512_store_si512(&dest[13], rd);
    _mm512_store_si512(&dest[14], re);
    _mm512_store_si512(&dest[15], rf);
    dest += 16;
  }
}

void raw_copy_256(__m256i *dest, __m256i *src, size_t length)
{
  __m256i r0, r1, r2, r3, r4, r5, r6, r7;
  __m256i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 5;
  for (__m256i *p = src; p < src+length; p += 16) {
    r0 = _mm256_load_si256(&p[0]);
    r1 = _mm256_load_si256(&p[1]);
    r2 = _mm256_load_si256(&p[2]);
    r3 = _mm256_load_si256(&p[3]);
    r4 = _mm256_load_si256(&p[4]);
    r5 = _mm256_load_si256(&p[5]);
    r6 = _mm256_load_si256(&p[6]);
    r7 = _mm256_load_si256(&p[7]);
    r8 = _mm256_load_si256(&p[8]);
    r9 = _mm256_load_si256(&p[9]);
    ra = _mm256_load_si256(&p[10]);
    rb = _mm256_load_si256(&p[11]);
    rc = _mm256_load_si256(&p[12]);
    rd = _mm256_load_si256(&p[13]);
    re = _mm256_load_si256(&p[14]);
    rf = _mm256_load_si256(&p[15]);
    _mm256_store_si256(&dest[0], r0);
    _mm256_store_si256(&dest[1], r1);
    _mm256_store_si256(&dest[2], r2);
    _mm256_store_si256(&dest[3], r3);
    _mm256_store_si256(&dest[4], r4);
    _mm256_store_si256(&dest[5], r5);
    _mm256_store_si256(&dest[6], r6);
    _mm256_store_si256(&dest[7], r7);
    _mm256_store_si256(&dest[8], r8);
    _mm256_store_si256(&dest[9], r9);
    _mm256_store_si256(&dest[10], ra);
    _mm256_store_si256(&dest[11], rb);
    _mm256_store_si256(&dest[12], rc);
    _mm256_store_si256(&dest[13], rd);
    _mm256_store_si256(&dest[14], re);
    _mm256_store_si256(&dest[15], rf);
    dest += 16;
  }
}

void raw_copy_stream(__m512i *dest, __m512i *src, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOAD(r0, p[0]);
    LOAD(r1, p[1]);
    LOAD(r2, p[2]);
    LOAD(r3, p[3]);
    LOAD(r4, p[4]);
    LOAD(r5, p[5]);
    LOAD(r6, p[6]);
    LOAD(r7, p[7]);
    LOAD(r8, p[8]);
    LOAD(r9, p[9]);
    LOAD(ra, p[10]);
    LOAD(rb, p[11]);
    LOAD(rc, p[12]);
    LOAD(rd, p[13]);
    LOAD(re, p[14]);
    LOAD(rf, p[15]);
    _mm512_stream_si512(&p[0], r0);
    _mm512_stream_si512(&p[1], r0);
    _mm512_stream_si512(&p[2], r0);
    _mm512_stream_si512(&p[3], r0);
    _mm512_stream_si512(&p[4], r0);
    _mm512_stream_si512(&p[5], r0);
    _mm512_stream_si512(&p[6], r0);
    _mm512_stream_si512(&p[7], r0);
    _mm512_stream_si512(&p[8], r0);
    _mm512_stream_si512(&p[9], r0);
    _mm512_stream_si512(&p[10], r0);
    _mm512_stream_si512(&p[11], r0);
    _mm512_stream_si512(&p[12], r0);
    _mm512_stream_si512(&p[13], r0);
    _mm512_stream_si512(&p[14], r0);
    _mm512_stream_si512(&p[15], r0);
    dest += 16;
  }
}

void raw_copy1(__m512i *dest, __m512i *src, size_t length)
{
  __m512i r0;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 1) {
    LOAD(r0, p[0]);
    _mm512_store_si512(&dest[0], r0);
    dest += 1;
  }

}

void raw_copynt(__m512i *dest, __m512i *src, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOADNT(r0, p[0]);
    LOADNT(r1, p[1]);
    LOADNT(r2, p[2]);
    LOADNT(r3, p[3]);
    LOADNT(r4, p[4]);
    LOADNT(r5, p[5]);
    LOADNT(r6, p[6]);
    LOADNT(r7, p[7]);
    LOADNT(r8, p[8]);
    LOADNT(r9, p[9]);
    LOADNT(ra, p[10]);
    LOADNT(rb, p[11]);
    LOADNT(rc, p[12]);
    LOADNT(rd, p[13]);
    LOADNT(re, p[14]);
    LOADNT(rf, p[15]);
    _mm512_store_si512(&dest[0], r0);
    _mm512_store_si512(&dest[1], r1);
    _mm512_store_si512(&dest[2], r2);
    _mm512_store_si512(&dest[3], r3);
    _mm512_store_si512(&dest[4], r4);
    _mm512_store_si512(&dest[5], r5);
    _mm512_store_si512(&dest[6], r6);
    _mm512_store_si512(&dest[7], r7);
    _mm512_store_si512(&dest[8], r8);
    _mm512_store_si512(&dest[9], r9);
    _mm512_store_si512(&dest[10], ra);
    _mm512_store_si512(&dest[11], rb);
    _mm512_store_si512(&dest[12], rc);
    _mm512_store_si512(&dest[13], rd);
    _mm512_store_si512(&dest[14], re);
    _mm512_store_si512(&dest[15], rf);
    dest += 16;
  }

}

/* transpose 8x8 double precision matrix subblock
 * rows are contiguous
 * columns are stride * 64 bytes apart
 * this copies 8 * 64 bytes or 512 bytes
 */
void transpose8_pd_subblock(__m512i *dest, __m512i *src, unsigned stride)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  LOAD(r0, src[(stride * 0)]); //  00 01 02 03 04 05 06 07
  LOAD(r1, src[(stride * 1)]); //  10 11 12 13 14 15 16 17
  LOAD(r2, src[(stride * 2)]); //  20 21 22 23 24 25 26 27
  LOAD(r3, src[(stride * 3)]); //  30 31 32 33 34 35 36 37
  LOAD(r4, src[(stride * 4)]); //  40 41 42 43 44 45 46 47
  LOAD(r5, src[(stride * 5)]); //  50 51 52 53 54 55 56 57
  LOAD(r6, src[(stride * 6)]); //  60 61 62 63 64 65 66 67
  LOAD(r7, src[(stride * 7)]); //  70 71 72 73 74 75 76 77
  __m512i t0 = _mm512_unpacklo_epi64(r0,r1); //  00 10 02 12 04 14 06 16 
  __m512i t1 = _mm512_unpackhi_epi64(r0,r1); //  01 11 03 13 05 15 07 17
  __m512i t2 = _mm512_unpacklo_epi64(r2,r3); //  20 30 22 32 24 34 26 36
  __m512i t3 = _mm512_unpackhi_epi64(r2,r3); //  21 31 23 33 25 35 27 37
  __m512i t4 = _mm512_unpacklo_epi64(r4,r5); //  40 50 42 52 44 54 46 56
  __m512i t5 = _mm512_unpackhi_epi64(r4,r5); //  41 51 43 53 45 55 47 57
  __m512i t6 = _mm512_unpacklo_epi64(r6,r7); //  60 70 62 72 64 74 66 76
  __m512i t7 = _mm512_unpackhi_epi64(r6,r7); //  61 71 63 73 65 75 67 77
#if 0
  _mm512_store_si512(dest + (stride * 0), t0); // 00 10 20 30 40 50 60 70
  _mm512_store_si512(dest + (stride * 1), t1); // 01 11 21 31 41 51 61 71
  _mm512_store_si512(dest + (stride * 2), t2); // 02 12 22 32 42 52 62 72
  _mm512_store_si512(dest + (stride * 3), t3); // 03 13 23 33 43 53 63 73
  _mm512_store_si512(dest + (stride * 4), t4); // 04 14 24 34 44 54 64 74
  _mm512_store_si512(dest + (stride * 5), t5); // 05 15 25 35 45 55 65 75
  _mm512_store_si512(dest + (stride * 6), t6); // 06 16 26 36 46 56 66 76
  _mm512_store_si512(dest + (stride * 7), t7); // 07 17 27 37 47 57 67 77
  print("step 1", (uint64_t *) dest);
#endif
  r0 = _mm512_shuffle_i64x2(t0, t2, 0x88); // 00 10 04 14 20 30 24 34
  r1 = _mm512_shuffle_i64x2(t0, t2, 0xdd); // 02 12 06 16 22 32 26 36
  r2 = _mm512_shuffle_i64x2(t1, t3, 0x88); // 01 11 05 15 21 31 25 35
  r3 = _mm512_shuffle_i64x2(t1, t3, 0xdd); // 03 13 07 17 23 33 27 37
  r4 = _mm512_shuffle_i64x2(t4, t6, 0x88); // 40 50 44 54 60 70 64 74
  r5 = _mm512_shuffle_i64x2(t4, t6, 0xdd); // 42 52 46 56 62 72 66 76
  r6 = _mm512_shuffle_i64x2(t5, t7, 0x88); // 41 51 45 55 61 71 65 75
  r7 = _mm512_shuffle_i64x2(t5, t7, 0xdd); // 43 53 47 57 63 73 67 77
#if 0
  _mm512_store_si512(dest + (stride * 0), r0);
  _mm512_store_si512(dest + (stride * 1), r1);
  _mm512_store_si512(dest + (stride * 2), r2);
  _mm512_store_si512(dest + (stride * 3), r3);
  _mm512_store_si512(dest + (stride * 4), r4);
  _mm512_store_si512(dest + (stride * 5), r5);
  _mm512_store_si512(dest + (stride * 6), r6);
  _mm512_store_si512(dest + (stride * 7), r7);
  print("step 2", (uint64_t *) dest);
#endif
  t0 = _mm512_shuffle_i64x2(r0, r4, 0x88); // 00 10 20 30 04 14 24 34
  t1 = _mm512_shuffle_i64x2(r2, r6, 0x88); // 01 11 21 31 05 15 25 35
  t2 = _mm512_shuffle_i64x2(r1, r5, 0x88); // 02 12 22 32 06 16 26 36
  t3 = _mm512_shuffle_i64x2(r3, r7, 0x88); // 03 13 23 33 07 17 27 37
  t4 = _mm512_shuffle_i64x2(r0, r4, 0xdd); // 40 50 60 70 44 54 64 74
  t5 = _mm512_shuffle_i64x2(r2, r6, 0xdd); // 41 51 61 71 45 55 65 75
  t6 = _mm512_shuffle_i64x2(r1, r5, 0xdd); // 42 52 62 72 46 56 66 76
  t7 = _mm512_shuffle_i64x2(r3, r7, 0xdd); // 43 53 63 73 47 57 67 77

  _mm512_store_si512(dest + (stride * 0), t0);
  _mm512_store_si512(dest + (stride * 1), t1);
  _mm512_store_si512(dest + (stride * 2), t2);
  _mm512_store_si512(dest + (stride * 3), t3);
  _mm512_store_si512(dest + (stride * 4), t4);
  _mm512_store_si512(dest + (stride * 5), t5);
  _mm512_store_si512(dest + (stride * 6), t6);
  _mm512_store_si512(dest + (stride * 7), t7);
#if 0
  print("step 3", (uint64_t *) dest);
#endif
}

void transpose8_pd_subblock_nt(__m512i *dest, __m512i *src, unsigned stride)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  LOADNT(r0, src[(stride * 0)]); //  00 01 02 03 04 05 06 07
  LOADNT(r1, src[(stride * 1)]); //  10 11 12 13 14 15 16 17
  LOADNT(r2, src[(stride * 2)]); //  20 21 22 23 24 25 26 27
  LOADNT(r3, src[(stride * 3)]); //  30 31 32 33 34 35 36 37
  LOADNT(r4, src[(stride * 4)]); //  40 41 42 43 44 45 46 47
  LOADNT(r5, src[(stride * 5)]); //  50 51 52 53 54 55 56 57
  LOADNT(r6, src[(stride * 6)]); //  60 61 62 63 64 65 66 67
  LOADNT(r7, src[(stride * 7)]); //  70 71 72 73 74 75 76 77
  __m512i t0 = _mm512_unpacklo_epi64(r0,r1); //  00 10 02 12 04 14 06 16 
  __m512i t1 = _mm512_unpackhi_epi64(r0,r1); //  01 11 03 13 05 15 07 17
  __m512i t2 = _mm512_unpacklo_epi64(r2,r3); //  20 30 22 32 24 34 26 36
  __m512i t3 = _mm512_unpackhi_epi64(r2,r3); //  21 31 23 33 25 35 27 37
  __m512i t4 = _mm512_unpacklo_epi64(r4,r5); //  40 50 42 52 44 54 46 56
  __m512i t5 = _mm512_unpackhi_epi64(r4,r5); //  41 51 43 53 45 55 47 57
  __m512i t6 = _mm512_unpacklo_epi64(r6,r7); //  60 70 62 72 64 74 66 76
  __m512i t7 = _mm512_unpackhi_epi64(r6,r7); //  61 71 63 73 65 75 67 77
#if 0
  _mm512_store_si512(dest + (stride * 0), t0); // 00 10 20 30 40 50 60 70
  _mm512_store_si512(dest + (stride * 1), t1); // 01 11 21 31 41 51 61 71
  _mm512_store_si512(dest + (stride * 2), t2); // 02 12 22 32 42 52 62 72
  _mm512_store_si512(dest + (stride * 3), t3); // 03 13 23 33 43 53 63 73
  _mm512_store_si512(dest + (stride * 4), t4); // 04 14 24 34 44 54 64 74
  _mm512_store_si512(dest + (stride * 5), t5); // 05 15 25 35 45 55 65 75
  _mm512_store_si512(dest + (stride * 6), t6); // 06 16 26 36 46 56 66 76
  _mm512_store_si512(dest + (stride * 7), t7); // 07 17 27 37 47 57 67 77
  print("step 1", (uint64_t *) dest);
#endif
  r0 = _mm512_shuffle_i64x2(t0, t2, 0x88); // 00 10 04 14 20 30 24 34
  r1 = _mm512_shuffle_i64x2(t0, t2, 0xdd); // 02 12 06 16 22 32 26 36
  r2 = _mm512_shuffle_i64x2(t1, t3, 0x88); // 01 11 05 15 21 31 25 35
  r3 = _mm512_shuffle_i64x2(t1, t3, 0xdd); // 03 13 07 17 23 33 27 37
  r4 = _mm512_shuffle_i64x2(t4, t6, 0x88); // 40 50 44 54 60 70 64 74
  r5 = _mm512_shuffle_i64x2(t4, t6, 0xdd); // 42 52 46 56 62 72 66 76
  r6 = _mm512_shuffle_i64x2(t5, t7, 0x88); // 41 51 45 55 61 71 65 75
  r7 = _mm512_shuffle_i64x2(t5, t7, 0xdd); // 43 53 47 57 63 73 67 77
#if 0
  _mm512_store_si512(dest + (stride * 0), r0);
  _mm512_store_si512(dest + (stride * 1), r1);
  _mm512_store_si512(dest + (stride * 2), r2);
  _mm512_store_si512(dest + (stride * 3), r3);
  _mm512_store_si512(dest + (stride * 4), r4);
  _mm512_store_si512(dest + (stride * 5), r5);
  _mm512_store_si512(dest + (stride * 6), r6);
  _mm512_store_si512(dest + (stride * 7), r7);
  print("step 2", (uint64_t *) dest);
#endif
  t0 = _mm512_shuffle_i64x2(r0, r4, 0x88); // 00 10 20 30 04 14 24 34
  t1 = _mm512_shuffle_i64x2(r2, r6, 0x88); // 01 11 21 31 05 15 25 35
  t2 = _mm512_shuffle_i64x2(r1, r5, 0x88); // 02 12 22 32 06 16 26 36
  t3 = _mm512_shuffle_i64x2(r3, r7, 0x88); // 03 13 23 33 07 17 27 37
  t4 = _mm512_shuffle_i64x2(r0, r4, 0xdd); // 40 50 60 70 44 54 64 74
  t5 = _mm512_shuffle_i64x2(r2, r6, 0xdd); // 41 51 61 71 45 55 65 75
  t6 = _mm512_shuffle_i64x2(r1, r5, 0xdd); // 42 52 62 72 46 56 66 76
  t7 = _mm512_shuffle_i64x2(r3, r7, 0xdd); // 43 53 63 73 47 57 67 77

  _mm512_store_si512(dest + (stride * 0), t0);
  _mm512_store_si512(dest + (stride * 1), t1);
  _mm512_store_si512(dest + (stride * 2), t2);
  _mm512_store_si512(dest + (stride * 3), t3);
  _mm512_store_si512(dest + (stride * 4), t4);
  _mm512_store_si512(dest + (stride * 5), t5);
  _mm512_store_si512(dest + (stride * 6), t6);
  _mm512_store_si512(dest + (stride * 7), t7);
#if 0
  print("step 3", (uint64_t *) dest);
#endif
}


/* This transposes a 16 x 16 array of 32 bit words
 * The rows are contiguous
 * The columns are stride bytes apart, which must be a multiple of 64
 */
void transpose16_ps_subblock(__m512i *dest, __m512i *src, unsigned stride)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;

  LOAD(r0, src[(stride * 0)]);
  LOAD(r1, src[(stride * 1)]);
  LOAD(r2, src[(stride * 2)]);
  LOAD(r3, src[(stride * 3)]);
  LOAD(r4, src[(stride * 4)]);
  LOAD(r5, src[(stride * 5)]);
  LOAD(r6, src[(stride * 6)]);
  LOAD(r7, src[(stride * 7)]);
  LOAD(r8, src[(stride * 8)]);
  LOAD(r9, src[(stride * 9)]);
  LOAD(ra, src[(stride * 10)]);
  LOAD(rb, src[(stride * 11)]);
  LOAD(rc, src[(stride * 12)]);
  LOAD(rd, src[(stride * 13)]);
  LOAD(re, src[(stride * 14)]);
  LOAD(rf, src[(stride * 15)]);
  __m512i t0 = _mm512_unpacklo_epi32(r0,r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
  __m512i t1 = _mm512_unpackhi_epi32(r0,r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
  __m512i t2 = _mm512_unpacklo_epi32(r2,r3); //  32  48  33  49 ...
  __m512i t3 = _mm512_unpackhi_epi32(r2,r3); //  34  50  35  51 ...
  __m512i t4 = _mm512_unpacklo_epi32(r4,r5); //  64  80  65  81 ...  
  __m512i t5 = _mm512_unpackhi_epi32(r4,r5); //  66  82  67  83 ...
  __m512i t6 = _mm512_unpacklo_epi32(r6,r7); //  96 112  97 113 ...
  __m512i t7 = _mm512_unpackhi_epi32(r6,r7); //  98 114  99 115 ...
  __m512i t8 = _mm512_unpacklo_epi32(r8,r9); // 128 ...
  __m512i t9 = _mm512_unpackhi_epi32(r8,r9); // 130 ...
  __m512i ta = _mm512_unpacklo_epi32(ra,rb); // 160 ...
  __m512i tb = _mm512_unpackhi_epi32(ra,rb); // 162 ...
  __m512i tc = _mm512_unpacklo_epi32(rc,rd); // 196 ...
  __m512i td = _mm512_unpackhi_epi32(rc,rd); // 198 ...
  __m512i te = _mm512_unpacklo_epi32(re,rf); // 228 ...
  __m512i tf = _mm512_unpackhi_epi32(re,rf); // 230 ...
  r0 = _mm512_unpacklo_epi64(t0,t2); //   0  16  32  48 ...
  r1 = _mm512_unpackhi_epi64(t0,t2); //   1  17  33  49 ...
  r2 = _mm512_unpacklo_epi64(t1,t3); //   2  18  34  49 ...
  r3 = _mm512_unpackhi_epi64(t1,t3); //   3  19  35  51 ...
  r4 = _mm512_unpacklo_epi64(t4,t6); //  64  80  96 112 ...  
  r5 = _mm512_unpackhi_epi64(t4,t6); //  65  81  97 114 ...
  r6 = _mm512_unpacklo_epi64(t5,t7); //  66  82  98 113 ...
  r7 = _mm512_unpackhi_epi64(t5,t7); //  67  83  99 115 ...
  r8 = _mm512_unpacklo_epi64(t8,ta); // 128 144 160 176 ...  
  r9 = _mm512_unpackhi_epi64(t8,ta); // 129 145 161 178 ...
  ra = _mm512_unpacklo_epi64(t9,tb); // 130 146 162 177 ... 
  rb = _mm512_unpackhi_epi64(t9,tb); // 131 147 163 179 ...
  rc = _mm512_unpacklo_epi64(tc,te); // 192 208 228 240 ... 
  rd = _mm512_unpackhi_epi64(tc,te); // 193 209 229 241 ...
  re = _mm512_unpacklo_epi64(td,tf); // 194 210 230 242 ...
  rf = _mm512_unpackhi_epi64(td,tf); // 195 211 231 243 ...
  
  t0 = _mm512_shuffle_i32x4(r0, r4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
  t1 = _mm512_shuffle_i32x4(r1, r5, 0x88); //   1  17  33  49 ...
  t2 = _mm512_shuffle_i32x4(r2, r6, 0x88); //   2  18  34  50 ...
  t3 = _mm512_shuffle_i32x4(r3, r7, 0x88); //   3  19  35  51 ...
  t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd); //   4  20  36  52 ...
  t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd); //   5  21  37  53 ...
  t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd); //   6  22  38  54 ...
  t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd); //   7  23  39  55 ...
  t8 = _mm512_shuffle_i32x4(r8, rc, 0x88); // 128 144 160 176 ...
  t9 = _mm512_shuffle_i32x4(r9, rd, 0x88); // 129 145 161 177 ...
  ta = _mm512_shuffle_i32x4(ra, re, 0x88); // 130 146 162 178 ...
  tb = _mm512_shuffle_i32x4(rb, rf, 0x88); // 131 147 163 179 ...
  tc = _mm512_shuffle_i32x4(r8, rc, 0xdd); // 132 148 164 180 ...
  td = _mm512_shuffle_i32x4(r9, rd, 0xdd); // 133 149 165 181 ...
  te = _mm512_shuffle_i32x4(ra, re, 0xdd); // 134 150 166 182 ...
  tf = _mm512_shuffle_i32x4(rb, rf, 0xdd); // 135 151 167 183 ...
  
  r0 = _mm512_shuffle_i32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
  r1 = _mm512_shuffle_i32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
  r2 = _mm512_shuffle_i32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
  r3 = _mm512_shuffle_i32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
  r4 = _mm512_shuffle_i32x4(t4, tc, 0x88); //   4 ...
  r5 = _mm512_shuffle_i32x4(t5, td, 0x88); //   5 ...
  r6 = _mm512_shuffle_i32x4(t6, te, 0x88); //   6 ...
  r7 = _mm512_shuffle_i32x4(t7, tf, 0x88); //   7 ...
  r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd); //   8 ...
  r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd); //   9 ...
  ra = _mm512_shuffle_i32x4(t2, ta, 0xdd); //  10 ...
  rb = _mm512_shuffle_i32x4(t3, tb, 0xdd); //  11 ...
  rc = _mm512_shuffle_i32x4(t4, tc, 0xdd); //  12 ...
  rd = _mm512_shuffle_i32x4(t5, td, 0xdd); //  13 ...
  re = _mm512_shuffle_i32x4(t6, te, 0xdd); //  14 ...
  rf = _mm512_shuffle_i32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255
  // store everything back
  _mm512_store_si512(dest + (stride * 0), r0);
  _mm512_store_si512(dest + (stride * 1), r1);
  _mm512_store_si512(dest + (stride * 2), r2);
  _mm512_store_si512(dest + (stride * 3), r3);
  _mm512_store_si512(dest + (stride * 4), r4);
  _mm512_store_si512(dest + (stride * 5), r5);
  _mm512_store_si512(dest + (stride * 6), r6);
  _mm512_store_si512(dest + (stride * 7), r7);
  _mm512_store_si512(dest + (stride * 8), r8);
  _mm512_store_si512(dest + (stride * 9), r9);
  _mm512_store_si512(dest + (stride * 10), ra);
  _mm512_store_si512(dest + (stride * 11), rb);
  _mm512_store_si512(dest + (stride * 12), rc);
  _mm512_store_si512(dest + (stride * 13), rd);
  _mm512_store_si512(dest + (stride * 14), re);
  _mm512_store_si512(dest + (stride * 15), rf);
}

/* This transposes a 16 x 16 array of 32 bit words
 * The rows are contiguous
 * The columns are stride bytes apart, which must be a multiple of 64
 */
void transpose16_ps_subblocknt(__m512i *dest, __m512i *src, unsigned stride)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;

  LOADNT(r0, src[(stride * 0)]);
  LOADNT(r1, src[(stride * 1)]);
  LOADNT(r2, src[(stride * 2)]);
  LOADNT(r3, src[(stride * 3)]);
  LOADNT(r4, src[(stride * 4)]);
  LOADNT(r5, src[(stride * 5)]);
  LOADNT(r6, src[(stride * 6)]);
  LOADNT(r7, src[(stride * 7)]);
  LOADNT(r8, src[(stride * 8)]);
  LOADNT(r9, src[(stride * 9)]);
  LOADNT(ra, src[(stride * 10)]);
  LOADNT(rb, src[(stride * 11)]);
  LOADNT(rc, src[(stride * 12)]);
  LOADNT(rd, src[(stride * 13)]);
  LOADNT(re, src[(stride * 14)]);
  LOADNT(rf, src[(stride * 15)]);
  __m512i t0 = _mm512_unpacklo_epi32(r0,r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
  __m512i t1 = _mm512_unpackhi_epi32(r0,r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
  __m512i t2 = _mm512_unpacklo_epi32(r2,r3); //  32  48  33  49 ...
  __m512i t3 = _mm512_unpackhi_epi32(r2,r3); //  34  50  35  51 ...
  __m512i t4 = _mm512_unpacklo_epi32(r4,r5); //  64  80  65  81 ...  
  __m512i t5 = _mm512_unpackhi_epi32(r4,r5); //  66  82  67  83 ...
  __m512i t6 = _mm512_unpacklo_epi32(r6,r7); //  96 112  97 113 ...
  __m512i t7 = _mm512_unpackhi_epi32(r6,r7); //  98 114  99 115 ...
  __m512i t8 = _mm512_unpacklo_epi32(r8,r9); // 128 ...
  __m512i t9 = _mm512_unpackhi_epi32(r8,r9); // 130 ...
  __m512i ta = _mm512_unpacklo_epi32(ra,rb); // 160 ...
  __m512i tb = _mm512_unpackhi_epi32(ra,rb); // 162 ...
  __m512i tc = _mm512_unpacklo_epi32(rc,rd); // 196 ...
  __m512i td = _mm512_unpackhi_epi32(rc,rd); // 198 ...
  __m512i te = _mm512_unpacklo_epi32(re,rf); // 228 ...
  __m512i tf = _mm512_unpackhi_epi32(re,rf); // 230 ...
  r0 = _mm512_unpacklo_epi64(t0,t2); //   0  16  32  48 ...
  r1 = _mm512_unpackhi_epi64(t0,t2); //   1  17  33  49 ...
  r2 = _mm512_unpacklo_epi64(t1,t3); //   2  18  34  49 ...
  r3 = _mm512_unpackhi_epi64(t1,t3); //   3  19  35  51 ...
  r4 = _mm512_unpacklo_epi64(t4,t6); //  64  80  96 112 ...  
  r5 = _mm512_unpackhi_epi64(t4,t6); //  65  81  97 114 ...
  r6 = _mm512_unpacklo_epi64(t5,t7); //  66  82  98 113 ...
  r7 = _mm512_unpackhi_epi64(t5,t7); //  67  83  99 115 ...
  r8 = _mm512_unpacklo_epi64(t8,ta); // 128 144 160 176 ...  
  r9 = _mm512_unpackhi_epi64(t8,ta); // 129 145 161 178 ...
  ra = _mm512_unpacklo_epi64(t9,tb); // 130 146 162 177 ... 
  rb = _mm512_unpackhi_epi64(t9,tb); // 131 147 163 179 ...
  rc = _mm512_unpacklo_epi64(tc,te); // 192 208 228 240 ... 
  rd = _mm512_unpackhi_epi64(tc,te); // 193 209 229 241 ...
  re = _mm512_unpacklo_epi64(td,tf); // 194 210 230 242 ...
  rf = _mm512_unpackhi_epi64(td,tf); // 195 211 231 243 ...
  
  t0 = _mm512_shuffle_i32x4(r0, r4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
  t1 = _mm512_shuffle_i32x4(r1, r5, 0x88); //   1  17  33  49 ...
  t2 = _mm512_shuffle_i32x4(r2, r6, 0x88); //   2  18  34  50 ...
  t3 = _mm512_shuffle_i32x4(r3, r7, 0x88); //   3  19  35  51 ...
  t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd); //   4  20  36  52 ...
  t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd); //   5  21  37  53 ...
  t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd); //   6  22  38  54 ...
  t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd); //   7  23  39  55 ...
  t8 = _mm512_shuffle_i32x4(r8, rc, 0x88); // 128 144 160 176 ...
  t9 = _mm512_shuffle_i32x4(r9, rd, 0x88); // 129 145 161 177 ...
  ta = _mm512_shuffle_i32x4(ra, re, 0x88); // 130 146 162 178 ...
  tb = _mm512_shuffle_i32x4(rb, rf, 0x88); // 131 147 163 179 ...
  tc = _mm512_shuffle_i32x4(r8, rc, 0xdd); // 132 148 164 180 ...
  td = _mm512_shuffle_i32x4(r9, rd, 0xdd); // 133 149 165 181 ...
  te = _mm512_shuffle_i32x4(ra, re, 0xdd); // 134 150 166 182 ...
  tf = _mm512_shuffle_i32x4(rb, rf, 0xdd); // 135 151 167 183 ...
  
  r0 = _mm512_shuffle_i32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
  r1 = _mm512_shuffle_i32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
  r2 = _mm512_shuffle_i32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
  r3 = _mm512_shuffle_i32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
  r4 = _mm512_shuffle_i32x4(t4, tc, 0x88); //   4 ...
  r5 = _mm512_shuffle_i32x4(t5, td, 0x88); //   5 ...
  r6 = _mm512_shuffle_i32x4(t6, te, 0x88); //   6 ...
  r7 = _mm512_shuffle_i32x4(t7, tf, 0x88); //   7 ...
  r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd); //   8 ...
  r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd); //   9 ...
  ra = _mm512_shuffle_i32x4(t2, ta, 0xdd); //  10 ...
  rb = _mm512_shuffle_i32x4(t3, tb, 0xdd); //  11 ...
  rc = _mm512_shuffle_i32x4(t4, tc, 0xdd); //  12 ...
  rd = _mm512_shuffle_i32x4(t5, td, 0xdd); //  13 ...
  re = _mm512_shuffle_i32x4(t6, te, 0xdd); //  14 ...
  rf = _mm512_shuffle_i32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255
  // store everything back
  _mm512_store_si512(dest + (stride * 0), r0);
  _mm512_store_si512(dest + (stride * 1), r1);
  _mm512_store_si512(dest + (stride * 2), r2);
  _mm512_store_si512(dest + (stride * 3), r3);
  _mm512_store_si512(dest + (stride * 4), r4);
  _mm512_store_si512(dest + (stride * 5), r5);
  _mm512_store_si512(dest + (stride * 6), r6);
  _mm512_store_si512(dest + (stride * 7), r7);
  _mm512_store_si512(dest + (stride * 8), r8);
  _mm512_store_si512(dest + (stride * 9), r9);
  _mm512_store_si512(dest + (stride * 10), ra);
  _mm512_store_si512(dest + (stride * 11), rb);
  _mm512_store_si512(dest + (stride * 12), rc);
  _mm512_store_si512(dest + (stride * 13), rd);
  _mm512_store_si512(dest + (stride * 14), re);
  _mm512_store_si512(dest + (stride * 15), rf);
}

void raw_transpose(__m512i *dest, __m512i *src, size_t length)
{
  length = length >> 6;
  for (unsigned i = 0; i < length; i += 16) {
    transpose16_ps_subblock(dest + i, src + i, 1);
  }
}

void raw_transposent(__m512i *dest, __m512i *src, size_t length)
{
  length = length >> 6;
  for (unsigned i = 0; i < length; i += 16) {
    transpose16_ps_subblocknt(dest + i, src + i, 1);
  }
}
void raw_transpose_pd(__m512i *dest, __m512i *src, size_t length)
{
  length = length >> 6;
  for (unsigned i = 0; i < length; i += 16) {
    transpose8_pd_subblock(dest + i, src + i, 1);
    transpose8_pd_subblock(dest + 8 + i, src + 8 + i, 1);
  }
}

void raw_transpose_pd_nt(__m512i *dest, __m512i *src, size_t length)
{
  length = length >> 6;
  for (unsigned i = 0; i < length; i += 16) {
    transpose8_pd_subblock_nt(dest + i, src + i, 1);
    transpose8_pd_subblock_nt(dest + 8 + i, src + 8 + i, 1);
  }
}
 
void raw_prefetch_T0(__m512i *src, size_t length)
{
  __m512i *end = src + (length >> 6);
  for (; src < end; src += 16) {  
    _mm_prefetch((const char *) (src + 0), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 1), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 2), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 3), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 4), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 5), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 6), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 7), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 8), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 9), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 10), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 11), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 12), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 13), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 14), _MM_HINT_T0);
    _mm_prefetch((const char *) (src + 15), _MM_HINT_T0);
  }
}


void raw_prefetch_T1(__m512i *src, size_t length)
{
  __m512i *end = src + (length >> 6);
  for (; src < end; src += 16) {  
    _mm_prefetch((const char *) (src + 0), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 1), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 2), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 3), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 4), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 5), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 6), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 7), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 8), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 9), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 10), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 11), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 12), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 13), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 14), _MM_HINT_T1);
    _mm_prefetch((const char *) (src + 15), _MM_HINT_T1);
  }
}


void raw_prefetch_T2(__m512i *src, size_t length)
{
  __m512i *end = src + (length >> 6);
  for (; src < end; src += 16) {  
    _mm_prefetch((const char *) (src + 0), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 1), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 2), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 3), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 4), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 5), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 6), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 7), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 8), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 9), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 10), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 11), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 12), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 13), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 14), _MM_HINT_T2);
    _mm_prefetch((const char *) (src + 15), _MM_HINT_T2);
  }
}


void raw_prefetch_NTA(__m512i *src, size_t length)
{
  __m512i *end = src + (length >> 6);
  for (; src < end; src += 16) {  
    _mm_prefetch((const char *) (src + 0), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 1), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 2), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 3), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 4), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 5), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 6), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 7), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 8), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 9), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 10), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 11), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 12), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 13), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 14), _MM_HINT_NTA);
    _mm_prefetch((const char *) (src + 15), _MM_HINT_NTA);
  }
}

void raw_prefetch_ET0(__m512i *src, size_t length)
{
  __m512i *end = src + (length >> 6);
  for (; src < end; src += 16) {  
    _mm_prefetch((const char *) (src + 0), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 1), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 2), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 3), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 4), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 5), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 6), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 7), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 8), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 9), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 10), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 11), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 12), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 13), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 14), _MM_HINT_ET0);
    _mm_prefetch((const char *) (src + 15), _MM_HINT_ET0);
  }
}


void raw_prefetch_ET1(__m512i *src, size_t length)
{
#if 0
  __m512i *end = src + (length >> 6);
  for (; src < end; src += 16) {  
    _mm_prefetch((const char *) (src + 0), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 1), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 2), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 3), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 4), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 5), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 6), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 7), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 8), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 9), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 10), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 11), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 12), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 13), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 14), _MM_HINT_ET1);
    _mm_prefetch((const char *) (src + 15), _MM_HINT_ET1);
  }
#endif
}


void raw_prefetch_ET2(__m512i *src, size_t length)
{
#if 0
  __m512i *end = src + (length >> 6);
  for (; src < end; src += 16) {  
    _mm_prefetch((const char *) (src + 0), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 1), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 2), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 3), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 4), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 5), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 6), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 7), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 8), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 9), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 10), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 11), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 12), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 13), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 14), _MM_HINT_ET2);
    _mm_prefetch((const char *) (src + 15), _MM_HINT_ET2);
  }
#endif
}


void raw_prefetch_ENTA(__m512i *src, size_t length)
{
#if 0
  __m512i *end = src + (length >> 6);
  for (; src < end; src += 16) {  
    _mm_prefetch((const char *) (src + 0), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 1), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 2), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 3), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 4), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 5), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 6), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 7), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 8), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 9), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 10), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 11), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 12), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 13), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 14), _MM_HINT_ENTA);
    _mm_prefetch((const char *) (src + 15), _MM_HINT_ENTA);
  }
#endif
}


void raw_loadandprefetch_T0(__m512i *src, __m512i *pf, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOAD(r0, p[0]);
    _mm_prefetch((const char *) (pf + 0), _MM_HINT_T0);
    LOAD(r1, p[1]);
    _mm_prefetch((const char *) (pf + 1), _MM_HINT_T0);
    LOAD(r2, p[2]);
    _mm_prefetch((const char *) (pf + 2), _MM_HINT_T0);
    LOAD(r3, p[3]);
    _mm_prefetch((const char *) (pf + 3), _MM_HINT_T0);
    LOAD(r4, p[4]);
    _mm_prefetch((const char *) (pf + 4), _MM_HINT_T0);
    LOAD(r5, p[5]);
    _mm_prefetch((const char *) (pf + 5), _MM_HINT_T0);
    LOAD(r6, p[6]);
    _mm_prefetch((const char *) (pf + 6), _MM_HINT_T0);
    LOAD(r7, p[7]);
    _mm_prefetch((const char *) (pf + 7), _MM_HINT_T0);
    LOAD(r8, p[8]);
    _mm_prefetch((const char *) (pf + 8), _MM_HINT_T0);
    LOAD(r9, p[9]);
    _mm_prefetch((const char *) (pf + 9), _MM_HINT_T0);
    LOAD(ra, p[10]);
    _mm_prefetch((const char *) (pf + 10), _MM_HINT_T0);
    LOAD(rb, p[11]);
    _mm_prefetch((const char *) (pf + 11), _MM_HINT_T0);
    LOAD(rc, p[12]);
    _mm_prefetch((const char *) (pf + 12), _MM_HINT_T0);
    LOAD(rd, p[13]);
    _mm_prefetch((const char *) (pf + 13), _MM_HINT_T0);
    LOAD(re, p[14]);
    _mm_prefetch((const char *) (pf + 14), _MM_HINT_T0);
    LOAD(rf, p[15]);
    _mm_prefetch((const char *) (pf + 15), _MM_HINT_T0);
  }
}

void raw_loadandprefetch_T1(__m512i *src, __m512i *pf, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOAD(r0, p[0]);
    _mm_prefetch((const char *) (pf + 0), _MM_HINT_T1);
    LOAD(r1, p[1]);
    _mm_prefetch((const char *) (pf + 1), _MM_HINT_T1);
    LOAD(r2, p[2]);
    _mm_prefetch((const char *) (pf + 2), _MM_HINT_T1);
    LOAD(r3, p[3]);
    _mm_prefetch((const char *) (pf + 3), _MM_HINT_T1);
    LOAD(r4, p[4]);
    _mm_prefetch((const char *) (pf + 4), _MM_HINT_T1);
    LOAD(r5, p[5]);
    _mm_prefetch((const char *) (pf + 5), _MM_HINT_T1);
    LOAD(r6, p[6]);
    _mm_prefetch((const char *) (pf + 6), _MM_HINT_T1);
    LOAD(r7, p[7]);
    _mm_prefetch((const char *) (pf + 7), _MM_HINT_T1);
    LOAD(r8, p[8]);
    _mm_prefetch((const char *) (pf + 8), _MM_HINT_T1);
    LOAD(r9, p[9]);
    _mm_prefetch((const char *) (pf + 9), _MM_HINT_T1);
    LOAD(ra, p[10]);
    _mm_prefetch((const char *) (pf + 10), _MM_HINT_T1);
    LOAD(rb, p[11]);
    _mm_prefetch((const char *) (pf + 11), _MM_HINT_T1);
    LOAD(rc, p[12]);
    _mm_prefetch((const char *) (pf + 12), _MM_HINT_T1);
    LOAD(rd, p[13]);
    _mm_prefetch((const char *) (pf + 13), _MM_HINT_T1);
    LOAD(re, p[14]);
    _mm_prefetch((const char *) (pf + 14), _MM_HINT_T1);
    LOAD(rf, p[15]);
    _mm_prefetch((const char *) (pf + 15), _MM_HINT_T1);
  }
}

void raw_fillandprefetch_ET0(__m512i *dest, __m512i *pf, size_t length)
{
  __m512i r0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
  length = length >> 6;
  for (__m512i *p = dest; p < dest+length; p += 16) {
    _mm_prefetch((const char *) (&pf[0]), _MM_HINT_ET0);
    _mm512_store_si512(&p[0], r0);
    _mm_prefetch((const char *) (&pf[1]), _MM_HINT_ET0);
    _mm512_store_si512(&p[1], r0);
    _mm_prefetch((const char *) (&pf[1]), _MM_HINT_ET0);
    _mm512_store_si512(&p[2], r0);
    _mm_prefetch((const char *) (&pf[3]), _MM_HINT_ET0);
    _mm512_store_si512(&p[3], r0);
    _mm_prefetch((const char *) (&pf[4]), _MM_HINT_ET0);
    _mm512_store_si512(&p[4], r0);
    _mm_prefetch((const char *) (&pf[5]), _MM_HINT_ET0);
    _mm512_store_si512(&p[5], r0);
    _mm_prefetch((const char *) (&pf[6]), _MM_HINT_ET0);
    _mm512_store_si512(&p[6], r0);
    _mm_prefetch((const char *) (&pf[7]), _MM_HINT_ET0);
    _mm512_store_si512(&p[7], r0);
    _mm_prefetch((const char *) (&pf[8]), _MM_HINT_ET0);
    _mm512_store_si512(&p[8], r0);
    _mm_prefetch((const char *) (&pf[9]), _MM_HINT_ET0);
    _mm512_store_si512(&p[9], r0);
    _mm_prefetch((const char *) (&pf[10]), _MM_HINT_ET0);
    _mm512_store_si512(&p[10], r0);
    _mm_prefetch((const char *) (&pf[11]), _MM_HINT_ET0);
    _mm512_store_si512(&p[11], r0);
    _mm_prefetch((const char *) (&pf[12]), _MM_HINT_ET0);
    _mm512_store_si512(&p[12], r0);
    _mm_prefetch((const char *) (&pf[13]), _MM_HINT_ET0);
    _mm512_store_si512(&p[13], r0);
    _mm_prefetch((const char *) (&pf[14]), _MM_HINT_ET0);
    _mm512_store_si512(&p[14], r0);
    _mm_prefetch((const char *) (&pf[15]), _MM_HINT_ET0);
    _mm512_store_si512(&p[15], r0);
    pf += 16;
  }
  //memset(cmd->dest, 0, cmd->length);
}

#if 0
void raw_fillandprefetch_ET1(__m512i *dest, __m512i *pf, size_t length)
{
  __m512i r0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
  length = length >> 6;
  for (__m512i *p = dest; p < dest+length; p += 16) {
    _mm_prefetch((const char *) (&pf[0]), _MM_HINT_ET1);
    _mm512_store_si512(&p[0], r0);
    _mm_prefetch((const char *) (&pf[1]), _MM_HINT_ET1);
    _mm512_store_si512(&p[1], r0);
    _mm_prefetch((const char *) (&pf[1]), _MM_HINT_ET1);
    _mm512_store_si512(&p[2], r0);
    _mm_prefetch((const char *) (&pf[3]), _MM_HINT_ET1);
    _mm512_store_si512(&p[3], r0);
    _mm_prefetch((const char *) (&pf[4]), _MM_HINT_ET1);
    _mm512_store_si512(&p[4], r0);
    _mm_prefetch((const char *) (&pf[5]), _MM_HINT_ET1);
    _mm512_store_si512(&p[5], r0);
    _mm_prefetch((const char *) (&pf[6]), _MM_HINT_ET1);
    _mm512_store_si512(&p[6], r0);
    _mm_prefetch((const char *) (&pf[7]), _MM_HINT_ET1);
    _mm512_store_si512(&p[7], r0);
    _mm_prefetch((const char *) (&pf[8]), _MM_HINT_ET1);
    _mm512_store_si512(&p[8], r0);
    _mm_prefetch((const char *) (&pf[9]), _MM_HINT_ET1);
    _mm512_store_si512(&p[9], r0);
    _mm_prefetch((const char *) (&pf[10]), _MM_HINT_ET1);
    _mm512_store_si512(&p[10], r0);
    _mm_prefetch((const char *) (&pf[11]), _MM_HINT_ET1);
    _mm512_store_si512(&p[11], r0);
    _mm_prefetch((const char *) (&pf[12]), _MM_HINT_ET1);
    _mm512_store_si512(&p[12], r0);
    _mm_prefetch((const char *) (&pf[13]), _MM_HINT_ET1);
    _mm512_store_si512(&p[13], r0);
    _mm_prefetch((const char *) (&pf[14]), _MM_HINT_ET1);
    _mm512_store_si512(&p[14], r0);
    _mm_prefetch((const char *) (&pf[15]), _MM_HINT_ET1);
    _mm512_store_si512(&p[15], r0);
    pf += 16;
  }
  //memset(cmd->dest, 0, cmd->length);
}
#endif

void raw_flush(__m512i *src, size_t length)
{
  length = length >> 6;
  for (unsigned i = 0; i < length; i += 16) {
    _mm_clflush((const char *) (src + i + 0));
    _mm_clflush((const char *) (src + i + 1));
    _mm_clflush((const char *) (src + i + 2));
    _mm_clflush((const char *) (src + i + 3));
    _mm_clflush((const char *) (src + i + 4));
    _mm_clflush((const char *) (src + i + 5));
    _mm_clflush((const char *) (src + i + 6));
    _mm_clflush((const char *) (src + i + 7));
    _mm_clflush((const char *) (src + i + 8));
    _mm_clflush((const char *) (src + i + 9));
    _mm_clflush((const char *) (src + i + 10));
    _mm_clflush((const char *) (src + i + 11));
    _mm_clflush((const char *) (src + i + 12));
    _mm_clflush((const char *) (src + i + 13));
    _mm_clflush((const char *) (src + i + 14));
    _mm_clflush((const char *) (src + i + 15));
  }
}

void raw_cldemote(__m512i *src, size_t length)
{
  length = length >> 6;
  for (unsigned i = 0; i < length; i += 16) {
    _mm_cldemote((const char *) (src + i + 0));
    _mm_cldemote((const char *) (src + i + 1));
    _mm_cldemote((const char *) (src + i + 2));
    _mm_cldemote((const char *) (src + i + 3));
    _mm_cldemote((const char *) (src + i + 4));
    _mm_cldemote((const char *) (src + i + 5));
    _mm_cldemote((const char *) (src + i + 6));
    _mm_cldemote((const char *) (src + i + 7));
    _mm_cldemote((const char *) (src + i + 8));
    _mm_cldemote((const char *) (src + i + 9));
    _mm_cldemote((const char *) (src + i + 10));
    _mm_cldemote((const char *) (src + i + 11));
    _mm_cldemote((const char *) (src + i + 12));
    _mm_cldemote((const char *) (src + i + 13));
    _mm_cldemote((const char *) (src + i + 14));
    _mm_cldemote((const char *) (src + i + 15));
  }
}

void raw_fillandcldemote(__m512i *dest, size_t length)
{
  __m512i r0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
  length = length >> 6;
  for (__m512i *p = dest; p < dest+length; p += 16) {
    _mm512_store_si512(&p[0], r0);
    _mm_cldemote(&p[0]);
    _mm512_store_si512(&p[1], r0);
    _mm_cldemote(&p[1]);
    _mm512_store_si512(&p[2], r0);
    _mm_cldemote(&p[2]);
    _mm512_store_si512(&p[3], r0);
    _mm_cldemote(&p[3]);
    _mm512_store_si512(&p[4], r0);
    _mm_cldemote(&p[4]);
    _mm512_store_si512(&p[5], r0);
    _mm_cldemote(&p[5]);
    _mm512_store_si512(&p[6], r0);
    _mm_cldemote(&p[6]);
    _mm512_store_si512(&p[7], r0);
    _mm_cldemote(&p[7]);
    _mm512_store_si512(&p[8], r0);
    _mm_cldemote(&p[8]);
    _mm512_store_si512(&p[9], r0);
    _mm_cldemote(&p[9]);
    _mm512_store_si512(&p[10], r0);
    _mm_cldemote(&p[10]);
    _mm512_store_si512(&p[11], r0);
    _mm_cldemote(&p[11]);
    _mm512_store_si512(&p[12], r0);
    _mm_cldemote(&p[12]);
    _mm512_store_si512(&p[13], r0);
    _mm_cldemote(&p[13]);
    _mm512_store_si512(&p[14], r0);
    _mm_cldemote(&p[14]);
    _mm512_store_si512(&p[15], r0);
    _mm_cldemote(&p[15]);
  }
}


void raw_fillandmovnt(__m512i *dest, size_t length)
{
  __m512i r0 = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);
  length = length >> 6;
  for (volatile __m512i *p = dest; p < dest+length; p += 16) {
    STORE((volatile void *) &p[0], r0);
    STORENT((volatile void *) &p[0], r0);
    STORE((volatile void *) &p[1], r0);
    STORENT((volatile void *) &p[1], r0);
    STORE((volatile void *) &p[2], r0);
    STORENT((volatile void *) &p[2], r0);
    STORE((volatile void *) &p[3], r0);
    STORENT((volatile void *) &p[3], r0);
    STORE((volatile void *) &p[4], r0);
    STORENT((volatile void *) &p[4], r0);
    STORE((volatile void *) &p[5], r0);
    STORENT((volatile void *) &p[5], r0);
    STORE((volatile void *) &p[6], r0);
    STORENT((volatile void *) &p[6], r0);
    STORE((volatile void *) &p[7], r0);
    STORENT((volatile void *) &p[7], r0);
    STORE((volatile void *) &p[8], r0);
    STORENT((volatile void *) &p[8], r0);
    STORE((volatile void *) &p[9], r0);
    STORENT((volatile void *) &p[9], r0);
    STORE((volatile void *) &p[10], r0);
    STORENT((volatile void *) &p[10], r0);
    STORE((volatile void *) &p[11], r0);
    STORENT((volatile void *) &p[11], r0);
    STORE((volatile void *) &p[12], r0);
    STORENT((volatile void *) &p[12], r0);
    STORE((volatile void *) &p[13], r0);
    STORENT((volatile void *) &p[13], r0);
    STORE((volatile void *) &p[14], r0);
    STORENT((volatile void *) &p[14], r0);
    STORE((volatile void *) &p[15], r0);
    STORENT((volatile void *) &p[15], r0);
  }
}



void raw_loadfill(__m512i *dest, size_t length)
{
  raw_load(dest, length);
  raw_fill(dest, length);
}

void raw_fillfill(__m512i *dest, size_t length)
{
  raw_fill(dest, length);
  raw_fill(dest, length);
}

void raw_load64(uint64_t *src, size_t length)
{
  uint64_t r0 = 0;
  length = length >> 3;
  while(length--) r0 += *src++;
}

void raw_copy64(uint64_t *dest, uint64_t *src, size_t length)
{
  length = length >> 3;
  while(length--) *dest++ = *src++;
}

/* These are the active message wrappers for the above
 * raw functions
 */
void do_nothing(AMMsg *p __attribute__ ((unused)))
{

}

void do_nopmov(AMMsg *p __attribute__ ((unused)))
{
  raw_nopmov((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_fill(AMMsg *p)
{
  raw_fill((__m512i *) p->arg[0], (size_t) p->arg[1]);
}
void do_fillwithmovnt(AMMsg *p)
{
  raw_fillwithmovnt((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_fillwithmovnt1of8(AMMsg *p)
{
  raw_fillwithmovnt1of8((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_fillwithmovnt8of8(AMMsg *p)
{
  raw_fillwithmovnt8of8((__m64 *) p->arg[0], (size_t) p->arg[1]);
}

void do_fillwithprefetch(AMMsg *p)
{
  raw_fillwithprefetch((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_fillwithprefetchw(AMMsg *p)
{
  raw_fillwithprefetchw((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_fill1of8(AMMsg *p)
{
  raw_fill1of8((uint64_t *) p->arg[0], (size_t) p->arg[1]);
}

void do_fillwithmovdir64b(AMMsg *p)
{
  raw_fillwithmovdir64b((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_load(AMMsg *p)
{
  raw_load((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_repload(AMMsg *p)
{
  raw_repload((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_load1of8(AMMsg *p)
{
  raw_load1of8((uint64_t *) p->arg[0], (size_t) p->arg[1]);
}

void do_loadnt(AMMsg *p)
{
  raw_loadnt((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_copy(AMMsg *p)
{
  raw_copy((__m512i *) p->arg[0], (__m512i *) p->arg[1], (size_t) p->arg[2]);
}

void do_copy_256(AMMsg *p)
{
  raw_copy_256((__m256i *) p->arg[0], (__m256i *) p->arg[1], (size_t) p->arg[2]);
}

void do_copy_stream(AMMsg *p)
{
  raw_copy_stream((__m512i *) p->arg[0], (__m512i *) p->arg[1], (size_t) p->arg[2]);
}

void do_copy1(AMMsg *p)
{
  raw_copy1((__m512i *) p->arg[0], (__m512i *) p->arg[1], (size_t) p->arg[2]);
}

void do_copynt(AMMsg *p)
{
  raw_copynt((__m512i *) p->arg[0], (__m512i *) p->arg[1], (size_t) p->arg[2]);
}

void do_transpose(AMMsg *p)
{
  raw_transpose((__m512i *) p->arg[0], (__m512i *) p->arg[1], 
		(size_t) p->arg[2]);
}

void do_transposent(AMMsg *p)
{
  raw_transposent((__m512i *) p->arg[0], (__m512i *) p->arg[1], 
		(size_t) p->arg[2]);
}
void do_transpose_pd(AMMsg *p)
{
  raw_transpose_pd((__m512i *) p->arg[0], (__m512i *) p->arg[1], 
		(size_t) p->arg[2]);
}

void do_transpose_pd_nt(AMMsg *p)
{
  raw_transpose_pd_nt((__m512i *) p->arg[0], (__m512i *) p->arg[1], 
		(size_t) p->arg[2]);
}

void do_prefetch_T0(AMMsg *p)
{
  raw_prefetch_T0((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_prefetch_T1(AMMsg *p)
{
  raw_prefetch_T1((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_prefetch_T2(AMMsg *p)
{
  raw_prefetch_T2((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_prefetch_NTA(AMMsg *p)
{
  raw_prefetch_NTA((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_prefetch_ET0(AMMsg *p)
{
  raw_prefetch_ET0((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_prefetch_ET1(AMMsg *p)
{
  raw_prefetch_ET1((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_prefetch_ET2(AMMsg *p)
{
  raw_prefetch_ET2((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_prefetch_ENTA(AMMsg *p)
{
  raw_prefetch_ENTA((__m512i *) p->arg[0], (size_t) p->arg[1]);
}


void do_flush(AMMsg *p)
{
  raw_flush((__m512i *) p->arg[0], (size_t) p->arg[1]);
}
void do_cldemote(AMMsg *p)
{
  raw_cldemote((__m512i *) p->arg[0], (size_t) p->arg[1]);
}
void do_fillandcldemote(AMMsg *p)
{
  raw_fillandcldemote((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_fillandmovnt(AMMsg *p)
{
  raw_fillandmovnt((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_memcpy(AMMsg *p)
{
  memcpy((void *) p->arg[0], (void *) p->arg[1], (size_t) p->arg[2]);
}


void do_memset(AMMsg *p)
{
  memset((void *) p->arg[0], (int) p->arg[1], (size_t) p->arg[2]);
}

void do_loadfill(AMMsg *p)
{
  raw_loadfill((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_fillfill(AMMsg *p)
{
  raw_fillfill((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_load64(AMMsg *p)
{
  raw_load64((uint64_t *) p->arg[0], (size_t) p->arg[1]);
}

void do_copy64(AMMsg *p)
{
  raw_copy64((uint64_t *) p->arg[0], 
	     (uint64_t *) p->arg[1], 
	     (size_t) p->arg[2]);
}



void am_do_nopmov(AMSet *ams, int cpu, void *dest, size_t length)
{
  AM_Send(&ams->am[cpu], do_nopmov, NULL, 2, dest, length);
}

void am_do_fill(AMSet *ams, int cpu, void *dest, size_t length)
{
  AM_Send(&ams->am[cpu], do_fill, NULL, 2, dest, length);
}

void am_do_fillwithmovnt(AMSet *ams, int cpu, void *dest, size_t length)
{
  AM_Send(&ams->am[cpu], do_fillwithmovnt, NULL, 2, dest, length);
}

void am_do_fillwithprefetch(AMSet *ams, int cpu, void *dest, size_t length)
{
  AM_Send(&ams->am[cpu], do_fillwithprefetch, NULL, 2, dest, length);
}

void am_do_fillwithprefetchw(AMSet *ams, int cpu, void *dest, size_t length)
{
  AM_Send(&ams->am[cpu], do_fillwithprefetchw, NULL, 2, dest, length);
}

void am_do_fill1of8(AMSet *ams, int cpu, void *dest, size_t length)
{
  AM_Send(&ams->am[cpu], do_fill1of8, NULL, 2, dest, length);
}

void am_do_load(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_load, NULL, 2, src, length);
}

void am_do_load1of8(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_load1of8, NULL, 2, src, length);
}

void am_do_loadnt(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_loadnt, NULL, 2, src, length);
}

void am_do_copy(AMSet *ams, int cpu, void *dest, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_copy, NULL, 3, dest, src, length);
}

void am_do_copy_256(AMSet *ams, int cpu, void *dest, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_copy_256, NULL, 3, dest, src, length);
}

void am_do_copy_stream(AMSet *ams, int cpu, void *dest, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_copy_stream, NULL, 3, dest, src, length);
}

void am_do_copy1(AMSet *ams, int cpu, void *dest, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_copy1, NULL, 3, dest, src, length);
}

void am_do_copynt(AMSet *ams, int cpu, void *dest, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_copynt, NULL, 3, dest, src, length);
}

void am_do_transpose(AMSet *ams, int cpu, void *dest, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_transpose, NULL, 3, dest, src, length);
}
void am_do_transposent(AMSet *ams, int cpu, void *dest, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_transposent, NULL, 3, dest, src, length);
}

void am_do_transpose_pd(AMSet *ams, int cpu, void *dest, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_transpose_pd, NULL, 3, dest, src, length);
}
void am_do_transpose_pd_nt(AMSet *ams, int cpu, void *dest, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_transpose_pd_nt, NULL, 3, dest, src, length);
}

void am_do_prefetch_T0(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_prefetch_T0, NULL, 2, src, length);
}

void am_do_prefetch_T1(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_prefetch_T1, NULL, 2, src, length);
}

void am_do_prefetch_T2(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_prefetch_T2, NULL, 2, src, length);
}

void am_do_prefetch_NTA(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_prefetch_NTA, NULL, 2, src, length);
}

void am_do_prefetch_ET0(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_prefetch_ET0, NULL, 2, src, length);
}

void am_do_prefetch_ET1(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_prefetch_ET1, NULL, 2, src, length);
}

void am_do_prefetch_ET2(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_prefetch_ET2, NULL, 2, src, length);
}

void am_do_prefetch_ENTA(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_prefetch_ENTA, NULL, 2, src, length);
}



void am_do_memcpy(AMSet *ams, int cpu, void *dest, void *src, size_t n)
{
  AM_Send(&ams->am[cpu], do_memcpy, NULL, 3, dest, src, n);
}

void am_do_memset(AMSet *ams, int cpu, void *buf, int c, size_t n)
{
  AM_Send(&ams->am[cpu], do_memset, NULL, 3, buf, c, n);
}

void am_do_loadfill(AMSet *ams, int cpu, void *dest, size_t length)
{
  AM_Send(&ams->am[cpu], do_loadfill, NULL, 2, dest, length);
}

void am_do_fillfill(AMSet *ams, int cpu, void *dest, size_t length)
{
  AM_Send(&ams->am[cpu], do_fillfill, NULL, 2, dest, length);
}

void am_do_load64(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_load64, NULL, 2, src, length);
}

void am_do_copy64(AMSet *ams, int cpu, void *dest, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_load64, NULL, 3, dest, src, length);
}



void do_movget_512(AMMsg *p)
{
  raw_load_movget_512((__m512i *) p->arg[0], (size_t) p->arg[1]);
}

void do_movget_256(AMMsg *p)
{
  raw_load_movget_256((__m256i *) p->arg[0], (size_t) p->arg[1]);
}

void do_movget_128(AMMsg *p)
{
  raw_load_movget_128((__m128i *) p->arg[0], (size_t) p->arg[1]);
}

void do_movget_64(AMMsg *p)
{
  raw_load_movget_64((uint64_t *) p->arg[0], (size_t) p->arg[1]);
}

void do_movget_32(AMMsg *p)
{
  raw_load_movget_32((uint32_t *) p->arg[0], (size_t) p->arg[1]);
}

void do_movget_16(AMMsg *p)
{
  raw_load_movget_16((uint16_t *) p->arg[0], (size_t) p->arg[1]);
}

void am_do_movget_512(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_movget_512, NULL, 2, src, length);
}

void am_do_movget_256(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_movget_256, NULL, 2, src, length);
}

void am_do_movget_128(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_movget_128, NULL, 2, src, length);
}

void am_do_movget_64(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_movget_64, NULL, 2, src, length);
}

void am_do_movget_32(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_movget_32, NULL, 2, src, length);
}

void am_do_movget_16(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_movget_16, NULL, 2, src, length);
}

void am_do_fillandcldemote(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_fillandcldemote, NULL, 2, src, length);
}

void am_do_fillandmovnt(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_fillandmovnt, NULL, 2, src, length);
}

void am_do_cldemote(AMSet *ams, int cpu, void *src, size_t length)
{
  AM_Send(&ams->am[cpu], do_cldemote, NULL, 2, src, length);
}

void do_ssc_mark_1(AMMsg *p)
{
  ssc_mark_1();
}

void do_ssc_mark_2(AMMsg *p)
{
  ssc_mark_2();
}

void do_start_trace(AMMsg *p)
{
  start_trace();
}
