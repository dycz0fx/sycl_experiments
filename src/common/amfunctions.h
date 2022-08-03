/* amfunctions.c
 * A Library of useful active message functions
 */

#ifndef AMFUNCTIONS_H
#define AMFUNCTIONS_H

#include "ActiveMessage.h"
#include <stddef.h>
#include <immintrin.h>

/* These can be used directly
 * They have 16 way unrolled loops
 * src and dest must be cachline aligned
 * length must be a multiple of 512
 */
void raw_nopmov(__m512i *dest, size_t length);
void raw_fill(__m512i *dest, size_t length);
void raw_fillwithmovnt(__m512i *dest, size_t length);
void raw_fillwithmovnt1of8(__m512i *dest, size_t length);
void raw_fillwithmovnt8of8(__m64 *dest, size_t length);
void raw_fillwithprefetch(__m512i *dest, size_t length);
void raw_fillwithprefetchw(__m512i *dest, size_t length);
void raw_fillwithmovdir64b(__m512i *dest, size_t length);
void raw_repload(__m512i *src, size_t length);
void raw_load(__m512i *src, size_t length);
void raw_loadnt(__m512i *src, size_t length);
void raw_copy(__m512i *dest, __m512i *src, size_t length);
void raw_copy_256(__m256i *dest, __m256i *src, size_t length);
void raw_copy_stream(__m512i *dest, __m512i *src, size_t length);
void raw_copy1(__m512i *dest, __m512i *src, size_t length);
void raw_copynt(__m512i *dest, __m512i *src, size_t length);
void transpose16_ps_subblock(__m512i *dest, __m512i *src, unsigned stride);
void transpose16_ps_subblocknt(__m512i *dest, __m512i *src, unsigned stride);
void raw_transpose(__m512i *dest, __m512i *src, size_t length);
void raw_transposent(__m512i *dest, __m512i *src, size_t length);
void transpose8_pd_subblock(__m512i *dest, __m512i *src, unsigned stride);
void transpose8_pd_subblock_nt(__m512i *dest, __m512i *src, unsigned stride);
void raw_transpose_pd(__m512i *dest, __m512i *src, size_t length);
void raw_transpose_pd_nt(__m512i *dest, __m512i *src, size_t length);
void raw_flush(__m512i *src, size_t length);

void raw_fillandcldemote(__m512i *src, size_t length);
void raw_fillandmovnt(__m512i *dest, size_t length); // proxy cldemote
void raw_cldemote(__m512i *src, size_t length);

void raw_prefetch_T0(__m512i *src, size_t length);
void raw_prefetch_T1(__m512i *src, size_t length);
void raw_prefetch_T2(__m512i *src, size_t length);
void raw_prefetch_NTA(__m512i *src, size_t length);
void raw_prefetch_ET0(__m512i *src, size_t length);
void raw_prefetch_ET1(__m512i *src, size_t length);
void raw_prefetch_ET2(__m512i *src, size_t length);
void raw_prefetch_ENTA(__m512i *src, size_t length);

void raw_fillandprefetch_ET0(__m512i *dest, __m512i *of, size_t length);
void raw_fillandprefetch_ET1(__m512i *dest, __m512i *of, size_t length);
void raw_loadandprefetch_T0(__m512i *src, __m512i *of, size_t length);
void raw_loadandprefetch_T1(__m512i *src, __m512i *of, size_t length);

/* useful for models without avx512 */

/* the 1 of 8 functions read or write only the first 8 bytes of a line */
void raw_load1of8(uint64_t *src, size_t length);
void raw_fill1of8(uint64_t *src, size_t length);
void raw_load64(uint64_t *src, size_t length);
void raw_copy64(uint64_t *dest, uint64_t *src, size_t length);

/* composed operations */

void raw_loadfill(__m512i *src, size_t length);
void raw_fillfill(__m512i *src, size_t length);


/* active message handler versions */
void do_nothing(AMMsg *p);
void do_fill(AMMsg *p);
void do_fillwithmovnt(AMMsg *p);
void do_fillwithmovnt1of8(AMMsg *p);
void do_fillwithmovnt8of8(AMMsg *p);
void do_nopmov(AMMsg *p);
void do_fillwithprefetch(AMMsg *p);
void do_fillwithprefetchw(AMMsg *p);
void do_load(AMMsg *p);
void do_repload(AMMsg *p);
void do_fill1of8(AMMsg *p);
void do_fillwithmovdir64b(AMMsg *p);
void do_load1of8(AMMsg *p);
void do_loadnt(AMMsg *p);
void do_copy(AMMsg *p);
void do_copy_256(AMMsg *p);
void do_copy_stream(AMMsg *p);
void do_copy1(AMMsg *p);

void do_copynt(AMMsg *p);
void do_transpose(AMMsg *p);
void do_transposent(AMMsg *p);
void do_transpose_pd(AMMsg *p);
void do_transpose_pd_nt(AMMsg *p);
void do_flush(AMMsg *p);
void do_cldemote(AMMsg *p);
void do_fillandcldemote(AMMsg *p);
void do_fillandmovnt(AMMsg *p);
void do_memcpy(AMMsg *p);
void do_memset(AMMsg *p);

void do_prefetch_T0(AMMsg *p);
void do_prefetch_T1(AMMsg *p);
void do_prefetch_T2(AMMsg *p);
void do_prefetch_NTA(AMMsg *p);
void do_prefetch_ET0(AMMsg *p);
void do_prefetch_ET1(AMMsg *p);
void do_prefetch_ET2(AMMsg *p);
void do_prefetch_ENTA(AMMsg *p);


void do_loadfill(AMMsg *p);
void do_fillfill(AMMsg *p);

void do_load64(AMMsg *p);
void do_copy64(AMMsg *p);

/* wrappers for active messages 
 * These can be used directly
 * They have 16 way unrolled loops
 * src and dest must be cachline aligned
 * length must be a multiple of 512
 */
void am_do_nopmov(AMSet *ams, int cpu, void *dest, size_t length);
void am_do_fill(AMSet *ams, int cpu, void *dest, size_t length);
void am_do_fillwithmovnt(AMSet *ams, int cpu, void *dest, size_t length);
void am_do_load(AMSet *ams, int cpu, void *src, size_t length);
void am_do_fill1of8(AMSet *ams, int cpu, void *dest, size_t length);
void am_do_load1of8(AMSet *ams, int cpu, void *src, size_t length);
void am_do_loadnt(AMSet *ams, int cpu, void *src, size_t length);
void am_do_copy(AMSet *ams, int cpu, void *dest, void *src, size_t length);
void am_do_copy_256(AMSet *ams, int cpu, void *dest, void *src, size_t length);
void am_do_copy_stream(AMSet *ams, int cpu, void *dest, void *src, size_t length);
void am_do_copy1(AMSet *ams, int cpu, void *dest, void *src, size_t length);
void am_do_copynt(AMSet *ams, int cpu, void *dest, void *src, size_t length);
void am_do_transpose(AMSet *ams, int cpu, void *dest, void *src, size_t length);
void am_do_transposent(AMSet *ams, int cpu, void *dest, void *src, size_t length);
void am_do_transpose_pd(AMSet *ams, int cpu, void *dest, void *src, size_t length);
void am_do_transpose_pd_nt(AMSet *ams, int cpu, void *dest, void *src, size_t length);
void am_do_prefetch_T0(AMSet *ams, int cpu, void *src, size_t length);
void am_do_prefetch_T1(AMSet *ams, int cpu, void *src, size_t length);
void am_do_prefetch_T2(AMSet *ams, int cpu, void *src, size_t length);
void am_do_prefetch_NTA(AMSet *ams, int cpu, void *src, size_t length);
void am_do_prefetch_ET0(AMSet *ams, int cpu, void *src, size_t length);
void am_do_prefetch_ET1(AMSet *ams, int cpu, void *src, size_t length);
void am_do_prefetch_ET2(AMSet *ams, int cpu, void *src, size_t length);
void am_do_prefetch_ENTA(AMSet *ams, int cpu, void *src, size_t length);

void am_do_flush(AMSet *ams, int cpu, void *src, size_t length);
void am_do_fillandcldemote(AMSet *ams, int cpu, void *src, size_t length);
void am_do_fillandmovnt(AMSet *ams, int cpu, void *src, size_t length);
void am_do_cldemote(AMSet *ams, int cpu, void *src, size_t length);
void am_do_memcpy(AMSet *ams, int cpu, void *dest, void *src, size_t n);
void am_do_memset(AMSet *ams, int cpu, void *buf, int c, size_t n);
void am_do_loadfill(AMSet *ams, int cpu, void *dest, size_t length);
void am_do_fillfill(AMSet *ams, int cpu, void *dest, size_t length);
void am_do_load64(AMSet *ams, int cpu, void *src, size_t length);
void am_do_copy64(AMSet *ams, int cpu, void *dst, void *src, size_t length);



// movget with correct encodings
// amfunctions.h equivalents
void do_movget_512(AMMsg *p);
void do_movget_256(AMMsg *p);
void do_movget_128(AMMsg *p);
void do_movget_64(AMMsg *p);
void do_movget_32(AMMsg *p);
void do_movget_16(AMMsg *p);
void am_do_movget_512(AMSet *ams, int cpu, void *src, size_t length);
void am_do_movget_256(AMSet *ams, int cpu, void *src, size_t length);
void am_do_movget_128(AMSet *ams, int cpu, void *src, size_t length);
void am_do_movget_64(AMSet *ams, int cpu, void *src, size_t length);
void am_do_movget_32(AMSet *ams, int cpu, void *src, size_t length);
void am_do_movget_16(AMSet *ams, int cpu, void *src, size_t length);


void do_ssc_mark_1(AMMsg *p);
void do_ssc_mark_2(AMMsg *p);
void do_start_trace(AMMsg *p);

#endif
