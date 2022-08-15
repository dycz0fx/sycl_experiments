#ifndef SIM_HELP_H
#define SIM_HELP_H

#define SSC_MARK(x) __SSC_MARK(x)
#ifndef __SSC_MARK
#define __SSC_MARK(tag)                                           \
  __asm__ __volatile__ ("movl %0, %%ebx; .byte 0x64, 0x67, 0x90 " ::"i"(tag):"%ebx")
#endif

#endif

