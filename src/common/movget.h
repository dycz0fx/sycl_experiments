/* movget.h
 * A Library of useful movget functions
 */

#include <immintrin.h>
#include <stdint.h>

// stop complaints about unused values in the load functions


void raw_load_movget_512(__m512i *src, size_t length);
void raw_load_movget_256(__m256i *src, size_t length);
void raw_load_movget_128(__m128i *src, size_t length);
void raw_load_movget_64(uint64_t *src, size_t length);
void raw_load_movget_32(uint32_t *src, size_t length);
void raw_load_movget_16(uint16_t *src, size_t length);

// now that compilers support movget we don't need these

// generic inlineable loads


/* WARNING.  The vector loads are not true intrinsics. they may 
   do the wrong thing.  Check the generated code
*/

/*
 ./xed64 -64 -v 5 -e vmovget xmm0 mem16:rdi
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: VMOVGET MEM0:xmmword ptr [RDI], MEM_WIDTH:16, MODE:2, REG0:XMM0, SMODE:2
OPERAND ORDER: REG0 MEM0 
Encodable! C4E27AC507
.byte 0xc4,0xe2,0x7a,0xc5,0x07
 */
static inline __m128i movget_128(__m128i *p)
{
  __m128i reg;
  __asm__ volatile (" .byte 0xc4,0xe2,0x7a,0xc5,0x07":"=x"(reg):"D"(p));
  return(reg);
}



/*
 ./xed64 -64 -v 5 -e vmovget ymm0 mem32:rdi
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: VMOVGET MEM0:ymmword ptr [RDI], MEM_WIDTH:32, MODE:2, REG0:YMM0, SMODE:2, VL:1
OPERAND ORDER: REG0 MEM0 
Encodable! C4E27EC507
.byte 0xc4,0xe2,0x7e,0xc5,0x07

 */

static inline __m256i movget_256(__m256i *p)
{
  __m256i reg;
  __asm__ volatile (" .byte 0xc4,0xe2,0x7e,0xc5,0x7":"=x"(reg):"D"(p));
  return(reg);
}

/*
 ./xed64 -64 -v 5 -e vmovget zmm0 mem64:rdi
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: VMOVGET MEM0:zmmword ptr [RDI], MEM_WIDTH:64, MODE:2, REG0:ZMM0, SMODE:2, VL:2
OPERAND ORDER: REG0 MEM0 
Encodable! 62F27E48C507
.byte 0x62,0xf2,0x7e,0x48,0xc5,0x07

*/

static inline __m512i movget_512(__m512i *p)
{
  __m512i reg;
  __asm__ volatile (" .byte 0x62,0xf2,0x7e,0x48,0xc5,0x07" :"=x"(reg):"D"(p));
  return(reg);
}

/*
  ./xed64 -64 -v 5 -e movget/64 rax mem8:rdi    
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: MOVGET EOSZ:3, MEM0:qword ptr [RDI], MEM_WIDTH:8, MODE:2, REG0:RAX, SMODE:2
OPERAND ORDER: REG0 MEM0 
Encodable! F3480F38FA07
.byte 0xf3,0x48,0x0f,0x38,0xfa,0x07

 */
static inline uint64_t movget_64(uint64_t *p)
{
  uint64_t reg;
  __asm__ volatile (" .byte 0xf3,0x48,0x0f,0x38,0xfa,0x07":"=a"(reg):"D"(p));
  return(reg);
}

/*
 ./xed64 -64 -v 5 -e movget/32 eax mem4:rdi    
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: MOVGET EOSZ:2, MEM0:dword ptr [RDI], MEM_WIDTH:4, MODE:2, REG0:EAX, SMODE:2
OPERAND ORDER: REG0 MEM0 
Encodable! F30F38FA07
.byte 0xf3,0x0f,0x38,0xfa,0x07

 */
static inline uint32_t movget_32(uint32_t *p)
{
  uint32_t reg;
  __asm__ volatile (" .byte 0xf3,0x0f,0x38,0xfa,0x07":"=a"(reg):"D"(p));
  return(reg);
}

/*
 ./xed64 -64 -v 5 -e movget/16 ax mem2:rdi    
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: MOVGET EOSZ:1, MEM0:word ptr [RDI], MEM_WIDTH:2, MODE:2, REG0:AX, SMODE:2
OPERAND ORDER: REG0 MEM0 
Encodable! F3660F38FA07
.byte 0xf3,0x66,0x0f,0x38,0xfa,0x07

 */
static inline uint16_t movget_16(uint16_t *p)
{
  uint16_t reg;
  __asm__ volatile (" .byte 0xf3,0x66,0x0f,0x38,0xfa,0x07":"=a"(reg):"D"(p));
  return(reg);
}

/*
 ./xed64 -64 -v 5 -e movget64b rdi mem64:rsi
Initializing XED tables...
Done initialing XED tables.
#XED version: [8.23.0-4-g01333f9]
Request: MOVGET64B MEM0:zmmword ptr [RSI], MEM_WIDTH:64, MODE:2, REG0:RDI, SMODE:2
OPERAND ORDER: REG0 MEM0 
Encodable! 480F38FA3E
.byte 0x48,0x0f,0x38,0xfa,0x3e

*/

static inline void movget64b(__m512i *dest, __m512i *src)
{
  __asm__ volatile (" .byte 0x48,0x0f,0x38,0xfa,0x3e"::"D"(dest),"S"(src):"memory");
}


