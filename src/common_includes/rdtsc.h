#ifndef MY_RDTSC__H
#define MY_RDTSC__H

#include <stdio.h>
/* so printf will be defined */

static inline unsigned long rdtsc(){
    unsigned int hi, lo;

    __asm volatile (
            "xorl %%eax, %%eax\n"
            "cpuid            \n"
            "rdtsc            \n"
            :"=a"(lo), "=d"(hi)
            :
            :"%ebx", "%ecx"
            );

    return ((unsigned long)hi << 32) | lo;
}

static inline int tsc_overhead() {
    unsigned long t0, t1;
    t0 = rdtsc();
    t1 = rdtsc();
    return (int)(t1 - t0);
}

static inline void show_tsc_overhead(){
    printf("tsc_overhead=%d\n", tsc_overhead());
}

/* See https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/ia-32-ia-64-benchmark-code-execution-paper.pdf */

static inline unsigned long rdtscp() {
    unsigned int hi, lo;

    __asm volatile (
            "rdtscp           \n"
	    "mov %%edx,%1    \n"
	    "mov %%eax,%0    \n"
            "cpuid            \n"
            :"=r"(lo), "=r"(hi)
            :
            :"%rax", "%rbx", "%rcx", "%rdx"
            );

    return ((unsigned long)hi << 32) | lo;
}


#endif
