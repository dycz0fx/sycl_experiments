/* movget.c
 * block load functions using different movget instructions
 */

#include "movget.h"

// stop complaints about unused values in the load functions

#pragma warning disable 593
#pragma warning disable 869

#if 0

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
  volatile __m512i r0, r1, r2, r3, r4, r5, r6, r7;
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
  for (__m256i *p = src; p < src+length;) {
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
    p += 16;
  }
}

void raw_load_movget_128(__m128i *src, size_t length)
{
  __m128i r0, r1, r2, r3, r4, r5, r6, r7;
  __m128i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 4;
  for (__m128i *p = src; p < src+length;) {
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
    p += 16;
  }
}

void raw_load_movget_512A(__m512i *src, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    LOAD_MG512A(r0, &p[0]);
    LOAD_MG512A(r1, &p[1]);
    LOAD_MG512A(r2, &p[2]);
    LOAD_MG512A(r3, &p[3]);
    LOAD_MG512A(r4, &p[4]);
    LOAD_MG512A(r5, &p[5]);
    LOAD_MG512A(r6, &p[6]);
    LOAD_MG512A(r7, &p[7]);
    LOAD_MG512A(r8, &p[8]);
    LOAD_MG512A(r9, &p[9]);
    LOAD_MG512A(ra, &p[10]);
    LOAD_MG512A(rb, &p[11]);
    LOAD_MG512A(rc, &p[12]);
    LOAD_MG512A(rd, &p[13]);
    LOAD_MG512A(re, &p[14]);
    LOAD_MG512A(rf, &p[15]);
  }
}


void raw_load_movget_256A(__m256i *src, size_t length)
{
  __m256i r0, r1, r2, r3, r4, r5, r6, r7;
  __m256i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 5;
  for (__m256i *p = src; p < src+length;) {
    LOAD_MG256A(r0, &p[0]);
    LOAD_MG256A(r1, &p[1]);
    LOAD_MG256A(r2, &p[2]);
    LOAD_MG256A(r3, &p[3]);
    LOAD_MG256A(r4, &p[4]);
    LOAD_MG256A(r5, &p[5]);
    LOAD_MG256A(r6, &p[6]);
    LOAD_MG256A(r7, &p[7]);
    LOAD_MG256A(r8, &p[8]);
    LOAD_MG256A(r9, &p[9]);
    LOAD_MG256A(ra, &p[10]);
    LOAD_MG256A(rb, &p[11]);
    LOAD_MG256A(rc, &p[12]);
    LOAD_MG256A(rd, &p[13]);
    LOAD_MG256A(re, &p[14]);
    LOAD_MG256A(rf, &p[15]);
    p += 16;
  }
}

void raw_load_movget_128A(__m128i *src, size_t length)
{
  __m128i r0, r1, r2, r3, r4, r5, r6, r7;
  __m128i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 4;
  for (__m128i *p = src; p < src+length;) {
    LOAD_MG128A(r0, &p[0]);
    LOAD_MG128A(r1, &p[1]);
    LOAD_MG128A(r2, &p[2]);
    LOAD_MG128A(r3, &p[3]);
    LOAD_MG128A(r4, &p[4]);
    LOAD_MG128A(r5, &p[5]);
    LOAD_MG128A(r6, &p[6]);
    LOAD_MG128A(r7, &p[7]);
    LOAD_MG128A(r8, &p[8]);
    LOAD_MG128A(r9, &p[9]);
    LOAD_MG128A(ra, &p[10]);
    LOAD_MG128A(rb, &p[11]);
    LOAD_MG128A(rc, &p[12]);
    LOAD_MG128A(rd, &p[13]);
    LOAD_MG128A(re, &p[14]);
    LOAD_MG128A(rf, &p[15]);
    p += 16;
  }
}

#endif


// not needed now we have compiler intrinsics

/*
 ./xed64  -64 -v 5 -e vmovget zmm0 mem64:rdi
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: VMOVGET MEM0:zmmword ptr [RDI], MEM_WIDTH:64, MODE:2, REG0:ZMM0, SMODE:2, VL:2
OPERAND ORDER: REG0 MEM0 
Encodable! 62F27E48C507
.byte 0x62,0xf2,0x7e,0x48,0xc5,0x07

 ./xed64  -64 -v 5 -e vmovget zmm1 mem64:rdi,-,-,01
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: VMOVGET DISP_WIDTH:8, MEM0:zmmword ptr [RDI+0x1], MEM_WIDTH:64, MODE:2, REG0:ZMM1, SMODE:2, VL:2
OPERAND ORDER: REG0 MEM0 
Encodable! 62F27E48C54F01
.byte 0x62,0xf2,0x7e,0x48,0xc5,0x4f,0x01

etc.
 */
void raw_load_movget_512(__m512i *src, size_t length)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;
  __m512i r8, r9, ra, rb, rc, rd, re, rf;
  length = length >> 6;
  for (__m512i *p = src; p < src+length; p += 16) {
    __asm__ volatile (" .byte 0x62,0xf2,0x7e,0x48,0xc5,0x07\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0xf2,0x7e,0x48,0xc5,0x4f,0x01\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0xf2,0x7e,0x48,0xc5,0x57,0x02\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0xf2,0x7e,0x48,0xc5,0x5f,0x03\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0xf2,0x7e,0x48,0xc5,0x67,0x04\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0xf2,0x7e,0x48,0xc5,0x6f,0x05\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0xf2,0x7e,0x48,0xc5,0x77,0x06\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0xf2,0x7e,0x48,0xc5,0x7f,0x07\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0x72,0x7e,0x48,0xc5,0x47,0x08\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0x72,0x7e,0x48,0xc5,0x4f,0x09\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0x72,0x7e,0x48,0xc5,0x57,0x0A\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0x72,0x7e,0x48,0xc5,0x5f,0x0B\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0x72,0x7e,0x48,0xc5,0x67,0x0C\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0x72,0x7e,0x48,0xc5,0x6f,0x0D\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0x72,0x7e,0x48,0xc5,0x77,0x0E\n"::"D"(p));
    __asm__ volatile (" .byte 0x62,0x72,0x7e,0x48,0xc5,0x7f,0x0F\n"::"D"(p));
  }
}


/*
 ./xed64  -64 -v 5 -e vmovget ymm0 mem32:rdi
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: VMOVGET MEM0:ymmword ptr [RDI], MEM_WIDTH:32, MODE:2, REG0:YMM0, SMODE:2, VL:1
OPERAND ORDER: REG0 MEM0 
Encodable! C4E27EC507
.byte 0xc4,0xe2,0x7e,0xc5,0x07

  ./xed64  -64 -v 5 -e vmovget ymm1 mem32:rdi,-,-,20
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: VMOVGET DISP_WIDTH:8, MEM0:ymmword ptr [RDI+0x20], MEM_WIDTH:32, MODE:2, REG0:YMM1, SMODE:2, VL:1
OPERAND ORDER: REG0 MEM0 
Encodable! C4E27EC54F20
.byte 0xc4,0xe2,0x7e,0xc5,0x4f,0x20

etc.
 */

void raw_load_movget_256(__m256i *src, size_t length)
{
  __m256i r0, r1, r2, r3;
  length = length >> 5;
  for (__m256i *p = src; p < src+length;) {
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x07\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x4f,0x20\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x57,0x40\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x5f,0x60\n"::"D"(p));
    p += 4;
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x07\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x4f,0x20\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x57,0x40\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x5f,0x60\n"::"D"(p));
    p += 4;
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x07\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x4f,0x20\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x57,0x40\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x5f,0x60\n"::"D"(p));
    p += 4;
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x07\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x4f,0x20\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x57,0x40\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x5f,0x60\n"::"D"(p));
    p += 4;
  }
}

/*
 ./xed64  -64 -v 5 -e vmovget xmm0 mem16:rdi
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: VMOVGET MEM0:xmmword ptr [RDI], MEM_WIDTH:16, MODE:2, REG0:XMM0, SMODE:2
OPERAND ORDER: REG0 MEM0 
Encodable! C4E27AC507
.byte 0xc4,0xe2,0x7a,0xc5,0x07

  ./xed64  -64 -v 5 -e vmovget xmm1 mem16:rdi,-,-,10
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: VMOVGET DISP_WIDTH:8, MEM0:xmmword ptr [RDI+0x10], MEM_WIDTH:16, MODE:2, REG0:XMM1, SMODE:2
OPERAND ORDER: REG0 MEM0 
Encodable! C4E27AC54F10
.byte 0xc4,0xe2,0x7a,0xc5,0x4f,0x10

etc.
 */
void raw_load_movget_128(__m128i *src, size_t length)
{
  __m128i r0, r1, r2, r3, r4, r5, r6, r7;
  length = length >> 4;
  for (__m128i *p = src; p < src+length;) {
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x07\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x4f,0x10\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x57,0x20\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x5f,0x30\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x67,0x40\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x6f,0x50\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x77,0x60\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x7f,0x70\n"::"D"(p));
    p += 8;
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x07\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x4f,0x10\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x57,0x20\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x5f,0x30\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x67,0x40\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x6f,0x50\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x77,0x60\n"::"D"(p));
    __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x7f,0x70\n"::"D"(p));
    p += 8;
  }
}

/* The following are not unrolled */

/*
 ./xed64  -64 -v 5 -e movget/64 rax mem8:rdi
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: MOVGET EOSZ:3, MEM0:qword ptr [RDI], MEM_WIDTH:8, MODE:2, REG0:RAX, SMODE:2
OPERAND ORDER: REG0 MEM0 
Encodable! F3480F38FA07
.byte 0xf3,0x48,0x0f,0x38,0xfa,0x07

 */

void raw_load_movget_64(uint64_t *src, size_t length)
{
  uint64_t r0;
  length = length >> 3;
  for (uint64_t *p = src; p < src + length; p += 1) {
    __asm__ volatile (" .byte 0xf3,0x48,0x0f,0x38,0xfa,0x07"::"D"(p):"rax");
  }
}

/*
 ./xed64  -64 -v 5 -e movget/32 eax mem4:rdi
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: MOVGET EOSZ:2, MEM0:dword ptr [RDI], MEM_WIDTH:4, MODE:2, REG0:EAX, SMODE:2
OPERAND ORDER: REG0 MEM0 
Encodable! F30F38FA07
.byte 0xf3,0x0f,0x38,0xfa,0x07
*/
void raw_load_movget_32(uint32_t *src, size_t length)
{
  uint32_t r0;
  length = length >> 2;
  for (uint32_t *p = src; p < src + length; p += 1) {
    __asm__ volatile (" .byte 0xf3,0x0f,0x38,0xfa,0x07"::"D"(p):"eax");
  }
}

/*
 ./xed64  -64 -v 5 -e movget/16 ax mem2:rdi
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: MOVGET EOSZ:1, MEM0:word ptr [RDI], MEM_WIDTH:2, MODE:2, REG0:AX, SMODE:2
OPERAND ORDER: REG0 MEM0 
Encodable! F3660F38FA07
.byte 0xf3,0x66,0x0f,0x38,0xfa,0x07
*/
void raw_load_movget_16(uint16_t *src, size_t length)
{
  uint16_t r0;
  length = length >> 1;
  for (uint16_t *p = src; p < src + length; p += 1) {
    __asm__ volatile (" .byte 0xf3,0x66,0x0f,0x38,0xfa,0x07"::"D"(p):"ax");
  }
}


// test functions for the inlines



uint16_t test_movget_16(uint16_t *p)
{
  return(movget_16(p));
}

uint32_t test_movget_32(uint32_t *p)
{
  return(movget_32(p));
}

uint64_t test_movget_64(uint64_t *p)
{
  return(movget_64(p));
}


__m128i test_movget_128(__m128i *p)
{
  return(movget_128(p));
}

__m256i test_movget_256(__m256i *p)
{
  return(movget_256(p));
}

__m512i test_movget_512(__m512i *p)
{
  return(movget_512(p));
}


void test_movget64b(__m512i *dest, __m512i *src)
{
  movget64b(dest, src);
}

/* This is an illustration that these are not, in fact, intrinsics, they
   dont't track registers properly
*/

__m512i test_add(__m512i *a, __m512i *b)
{
  __m512i aa, bb;
  aa = movget_512(a);
  bb = movget_512(a);
  return (_mm512_add_epi64(aa, bb));
}


