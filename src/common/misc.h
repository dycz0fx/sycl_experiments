#ifndef MISC__H
#define MISC__H

#ifndef __UPC__

#include "sim_help.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>    // ick, but printf is mentioned below

typedef union {
    __m512i mm;
    __m512  mf;
    __m512d md;
    uint32_t u32[16];
    uint64_t u64[8];
} variant512_t;

#define EXTRACT(_x_,_i_) ({ variant512_t t = {mm:_x_}; t.u32[_i_];})
#define EXTRACT_U64(_x_,_i_) ({ variant512_t t = {mm:_x_}; t.u64[_i_];})
void print__m512i(__m512i x);

#endif

#ifndef XSTR
#define XSTR(s) STRMACRO(s)
#define STRMACRO(s) #s
#endif

void ssc_mark_1();
void ssc_mark_2();
//void movget_memcpy(void* dst, void *src, int nb);  what was this for?

// Signal to start a sniper trace
void start_trace();


// Debug tools

#define LOC() printf("%s:%d (%s)\n", __FILE__, __LINE__, __FUNCTION__); \
  fflush(NULL);

//#define LOC() 

#endif
