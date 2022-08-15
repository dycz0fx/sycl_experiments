/*
  This Software is part of Wind River Simics. The rights to copy, distribute,
  modify, or otherwise make use of this Software may be licensed only
  pursuant to the terms of an applicable Wind River license agreement.
  
  Copyright 2010 Intel Corporation
*/

#ifndef SIMICS_MAGIC_INSTRUCTION_H
#define SIMICS_MAGIC_INSTRUCTION_H

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * This file contains the magic instructions for different target
 * architectures, as understood by different compilers.
 *
 *   arch   instr           limit               compilers       additional requirements
 *   ----------------------------------------------------------------------------------
 *   h8300  brn n           -128 <= n <= 127    gcc
 *   mips   li $zero,n      0 <= n <= 0xffff    gcc
 *   ppc    rlwimi x,x,0,y,z 0 <= n <= 0x1fff   gcc
 *                           x = n[12:8]
 *                           y = n[7:4]
 *                           z = n[3:0] | 16
 *   ppc    rlwimi x,x,0,y,z 0 <= n <= 0x7fff   gcc             old rlwimi
 *                           x = n[14:10]
 *                           y = n[9:5]
 *                           z = n[4:0]
 *   ppc    mr n,n          0 <= n < 32         gcc             compat magic
 *   ppc64  fmr n,n         0 <= n < 32         gcc             compat magic
 *   sh     mov rn,rn       0 <= rn < 16        gcc
 *   sparc  sethi n,%g0     0 <  n < (1 << 22)  gcc, WS C[++]
 *   x86    cpuid           0 <= n < 0x10000    gcc             (eax & 0xffff0000) == 0x4711
 * 
 */

/*
  <add id="magic instruction figure">
  <figure label="magic_instruction_figure">
  <center>
  <table border="cross">
    <tr>
       <td><b>Target</b></td>
       <td><b>Magic instruction</b></td>
       <td><b>Conditions on <arg>n</arg></b></td>
       <td/>
    </tr>
    <tr>
       <td>ARM</td><td><tt>orreq rn, rn, rn</tt></td>
                   <td><math>0 &lt;= n &lt; 15</math></td>
                   <td/>
    </tr>
    <tr>
       <td>H8300</td><td><tt>brn n</tt></td>
                     <td><math>-128 &lt;= n &lt;= 127</math></td>
                     <td/>
    </tr>
    <tr>
       <td>MIPS</td><td><tt>li %zero, n</tt></td>
                    <td><math>0 &lt;= n &lt; 0x10000</math></td>
                    <td/>
    </tr>
    <tr>
      <td>PowerPC</td><td><tt>rlwimi x,x,0,y,z</tt></td>
      <td><math>0 &lt;= n &lt; 8192</math></td>
      <td>new encoding</td>
    </tr>
    <tr>
      <td/>
      <td><math>x=n[12:8], y=n[7:4], z=n[3:0]|16</math></td>
      <td/>
      <td/>
    </tr>
    <tr>
      <td>PowerPC</td><td><tt>rlwimi x,x,0,y,z</tt></td>
      <td><math>0 &lt;= n &lt; 32768</math></td>
      <td>old encoding</td>
    </tr>
    <tr>
      <td/>
      <td><math>x=n[14:10], y=n[9:5], z=n[4:0]</math></td>
      <td/>
      <td/>
    </tr>
    <tr>
       <td>SH</td><td><tt>mov rn, rn</tt></td>
                        <td><math>0 &lt;= n &lt; 16</math></td>
                        <td/>
    </tr>
    <tr>
       <td>SPARC</td><td><tt>sethi n, %g0</tt></td>
               <td><math>1 &lt;= n &lt; 0x400000</math></td>
               <td/>
    </tr>
    <tr>
      <td>x86</td>
      <td><tt>cpuid</tt></td>
      <td><math>0 &lt;= n &lt; 0x10000</math></td>
      <td/>
    </tr>
    <tr>
      <td/>
      <td>with <tt>eax</tt> = <math>0x47110000 | n</math></td>
      <td/>
      <td/>
    </tr>
    <tr>
       <td>Cell SPU</td><td><tt>or n, n, n</tt></td>
                        <td><math>0 &lt;= n &lt; 128</math></td>
                        <td/>
    </tr>
  </table>
  </center>
  <caption>Magic instructions for different Simics Targets</caption>
  </figure>
  </add>
*/

/* BEGIN EXPORT simics/magic-instruction.h */

#define __MAGIC_CASSERT(p) do {                                 \
        typedef int __check_magic_argument[(p) ? 1 : -1];       \
} while (0)

#if defined(__GNUC__) || defined(__INTEL_COMPILER)

#if defined(__sparc) || defined(__sparc__)

#define MAGIC(n) do {                                   \
	__MAGIC_CASSERT((n) > 0 && (n) < (1U << 22));   \
        __asm__ __volatile__ ("sethi " #n ", %g0");     \
} while (0)

#define MAGIC_BREAKPOINT MAGIC(0x40000)

#elif defined(__i386) || defined(__x86_64__) || defined(__INTEL_COMPILER)

#define MAGIC(n) do {                                                       \
        int dummy_eax;                                                      \
	__MAGIC_CASSERT((unsigned)(n) < 0x10000);     \
        __asm__ __volatile__ ("cpuid"                                       \
                              : "=a" (dummy_eax)                            \
                              : "a" (0x4711 | ((unsigned)(n) << 16))        \
                              : "ecx", "edx", "ebx");                       \
} while (0)

#define MAGIC_BREAKPOINT MAGIC(0)

#elif defined(__powerpc__)

#if defined __powerpc64__ || defined SIM_NEW_RLWIMI_MAGIC
#define MAGIC(n) do {                                           \
       __MAGIC_CASSERT((n) >= 0 && (n) < (1 << 13));            \
       __asm__ __volatile__ ("rlwimi %0,%0,0,%1,%2"             \
                             :: "i" (((n) >> 8) & 0x1f),        \
                                "i" (((n) >> 4) & 0xf),         \
                                "i" ((((n) >> 0) & 0xf) | 16)); \
} while (0)
#else

#define MAGIC(n) do {                                           \
       __MAGIC_CASSERT((n) >= 0 && (n) < (1 << 15));            \
       __asm__ __volatile__ ("rlwimi %0,%0,0,%1,%2"             \
                             :: "i" (((n) >> 10) & 0x1f),       \
                                "i" (((n) >>  5) & 0x1f),       \
                                "i" (((n) >>  0) & 0x1f));      \
} while (0)
#endif

#define MAGIC_BREAKPOINT MAGIC(0)

#elif defined(__arm__)
#if defined(__thumb__)
#define MAGIC(n) do {                                   \
        __MAGIC_CASSERT((n) == 0);                      \
        __asm__ __volatile__ ("orr r0, r0");            \
} while (0)
#else
#define MAGIC(n) do {                                   \
        __MAGIC_CASSERT((n) == 0);                      \
        __asm__ __volatile__ ("orreq r0, r0, r0");      \
} while (0)
#endif // __thumb__
#define MAGIC_BREAKPOINT MAGIC(0)

#elif defined(__mips__)

#define MAGIC(n) do {                                   \
	__MAGIC_CASSERT((n) >= 0 && (n) <= 0xffff);     \
        /* GAS refuses to do 'li $zero,n' */            \
        __asm__ __volatile__ (".word 0x24000000+" #n);	\
} while (0)

#define MAGIC_BREAKPOINT MAGIC(0)

#elif defined __H8300__ || defined __H8300S__

#define MAGIC(n) do {                                   \
        __MAGIC_CASSERT((n) >= 0 && (n) <= 0xff);       \
        __asm__ __volatile__(".byte 0x41\n" /* brn */   \
                             ".byte " #n);  /* disp */  \
} while (0)

#define MAGIC_BREAKPOINT MAGIC(0)

#elif defined __SPU__

#define MAGIC(n) do {                                     \
	__MAGIC_CASSERT((n) >= 0 && (n) < 128);           \
        __asm__ __volatile__ ("or " #n ", " #n ", " #n);  \
} while (0)

#define MAGIC_BREAKPOINT MAGIC(0)

#elif defined(_SH1) || defined(_SH4) || defined(_SH4A)

#define MAGIC(n) do {                                \
        __MAGIC_CASSERT((n) >= 0 && (n) < 16);       \
        __asm__ __volatile__ ("mov r" #n ", r" #n);  \
} while (0)

#define MAGIC_BREAKPOINT MAGIC(0)

#else  /* !__sparc && !__i386 && !__powerpc__ */
#error "Unsupported architecture"
#endif /* !__sparc && !__i386 && !__powerpc__ */

#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)

#if defined(__sparc)

#define MAGIC(n) do {                                   \
	__MAGIC_CASSERT((n) > 0 && (n) < (1U << 22));   \
        asm ("sethi " #n ", %g0");                      \
} while (0)

#define MAGIC_BREAKPOINT MAGIC(0x40000)

#else  /* !__sparc */
#error "Unsupported architecture"
#endif /* !__sparc */

#else  /* !__GNUC__ && !__SUNPRO_C && !__SUNPRO_CC */

#ifdef _MSC_VER
#define MAGIC(n)
#define MAGIC_BREAKPOINT
#pragma message("MAGIC() macro needs attention!")
#else
#error "Unsupported compiler"
#endif

#endif /* !__GNUC__ && !__SUNPRO_C && !__SUNPRO_CC */

#if defined(__cplusplus)
}
#endif

#endif
