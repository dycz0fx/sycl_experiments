#pragma once

#include <stdint.h>
#include <xmmintrin.h>

#if defined(USE_MOVNT_AS_RAO)

#ifndef __ICC
#define __mm_stream_ps __builtin_ia32_movntps
#define __mm_stream_pd __builtin_ia32_movntpd
#endif

static inline void mm_aadd_ss(float *dst, __m128 val) {
    _mm_stream_ps(dst, val);
}

static inline void mm_aadd_sd(double *dst, __m128d val) {
    _mm_stream_pd(dst, val);
}

#elif !defined(__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES)

/* CPUID RAO_INT */
//void _aadd32(int *dst, int val);
static inline void aadd64(int64_t *dst, int64_t val) {
    // aadd %%rax, (%%rdx)
    __asm__ __volatile__ (
            ".byte 0x48, 0x0f, 0x38, 0xd0, 0x02 \n\t"
            :
            : "d"(dst), "a"(val)
            : "memory"
        );
}

static inline void aadd32(int32_t *dst, int32_t val)
{
    __asm__  (
            ".byte 0x0f,0x38,0xd0,0x07 \n\t"
            :
            : "D"(dst), "a"(val)
            : "memory"
        );

}

static inline void aand64(int64_t *dst, int64_t val) {
    __asm__ __volatile__ (
            ".byte 0xf3, 0x48, 0x0f, 0x38, 0xd0, 0x02 \n\t"
            :
            : "d"(dst), "a"(val)
            : "memory"
        );
}

//void _axor32(int *dst, int val);
static inline void axor64(int64_t *dst, int64_t val) {
    __asm__ __volatile__ (
            ".byte 0xf3, 0x48, 0x0f, 0x38, 0xd1, 0x02 \n\t"
            :
            : "d"(dst), "a"(val)
            : "memory"
        );
}

//void _aor32(int *dst, int val);
static inline void aor64(int64_t *dst, int64_t val) {
    __asm__ __volatile__ (
            ".byte 0x48, 0x0f, 0x38, 0xd1, 0x02 \n\t"
            :
            : "d"(dst), "a"(val)
            : "memory"
        );
}

/* CPUID RAO_INT_XCHG */
//int _axchg_i32(int *dst, int val);
static inline int64_t axchg_i64(int64_t *dst, int64_t val) {
    int64_t ret;
    // axchg %%rax, (%%rdx)
    __asm__ __volatile__ (".byte 0x48, 0x0f, 0x38, 0xd2, 0x02 \n\t"
            : "=a"(ret)
            : "d"(dst), "a"(val)
            : "memory");
    return ret;
}

/* Intrinsics for instruction: AXCHGADD */
//int _axchgadd_i32(int *dst, int val);
static inline int64_t axchgadd_i64(int64_t *dst, int64_t val) {
    int64_t ret;
    // axchgadd %%rax, (%%rdx)
    __asm__ __volatile__ (".byte 0xf3, 0x48, 0x0f, 0x38, 0xd2, 0x02 \n\t"
            : "=a"(ret)
            : "d"(dst), "a"(val)
            : "memory");
    return ret;
}

/* Intrinsics for instruction: AXCHGDEC */
//int _axchgdec_i32(int *dst);
static inline int64_t axchgdec_i64(int64_t *dst) {
    int64_t ret;
    __asm__ __volatile__ (".byte 0xf3, 0x48, 0x0f, 0x38, 0xd3, 0x02 \n\t"
            : "=a"(ret)
            : "d"(dst)
            : "memory");
    return ret;
}

/* Intrinsics for instruction: AXCHGINC */
//int _axchginc_i32(int *dst);
static inline int64_t axchginc_i64(int64_t *dst) {
    int64_t ret;
    __asm__ __volatile__ (".byte 0x48, 0x0f, 0x38, 0xd3, 0x02 \n\t"
            : "=a"(ret)
            : "d"(dst)
            : "memory");
    return ret;
}

/* CPUID RAO_AVX_FP/RAO_AVX512_FP */
/* Intrinsic for instruction: VAADDSD */
void mm_aadd_sd(double* dst, __m128d val) {
    // "vaaddsd %%xmm0, (%%rdx) \n\t"
    __asm__ __volatile__ ("movupd %1, %%xmm0 \n\t"
            ".byte 0xc4, 0xe2, 0xfb, 0x84, 0x02 \n\t"
            :
            : "d"(dst), "rm"(val)
            : "memory", "%xmm0");
}

/* Intrinsic for instruction: VAADDSS */
void mm_aadd_ss(float* dst, __m128 val) {
    // "vaaddss %%xmm0, (%%rdx) \n\t"
    __asm__ __volatile__ ("movups %1, %%xmm0 \n\t"
            ".byte 0xc4, 0xe2, 0x7a, 0x84, 0x02 \n\t"
            :
            : "d"(dst), "rm"(val)
            : "memory", "%xmm0");
}

/* Intrinsics for instruction: VAADDPD */
//void _mm_aadd_pd(double*, __m128d);
//void _mm256_aadd_pd(double*, __m256d);
//void _mm512_aadd_pd(double*, __m512d);

/* Intrinsics for instruction: VAADDPS */
//void _mm_aadd_ps(float*, __m128);
//void _mm256_aadd_ps(float*, __m256);
//void _mm512_aadd_ps(float*, __m512);

#else

#include <x86intrin.h>
#include <zmmintrin_internal.h>

static inline void aadd64(int64_t *dst, int64_t val) {
    _aadd64(dst, val);
}

static inline void aand64(int64_t *dst, int64_t val) {
    _aand64(dst, val);
}

static inline void axor64(int64_t *dst, int64_t val) {
    _axor64(dst, val);
}

static inline void aor64(int64_t *dst, int64_t val) {
    _aor64(dst, val);
}

static inline int64_t axchg_i64(int64_t *dst, int64_t val) {
    return _axchg_i64(dst, val);
}

static inline int64_t axchgadd_i64(int64_t *dst, int64_t val) {
    return _axchgadd_i64(dst, val);
}

static inline int64_t axchgdec_i64(int64_t *dst) {
    return _axchgdec_i64(dst);
}

static inline int64_t axchginc_i64(int64_t *dst) {
    return _axchginc_i64(dst);
}

static inline void mm_aadd_ss(float *dst, __m128 val) {
    //_mm_aadd_ss(dst, val);
    __asm__ __volatile__ ("movups %1, %%xmm0 \n\t"
            "vaaddss %%xmm0, (%%rdx) \n\t"
            :
            : "d"(dst), "rm"(val)
            : "memory", "%xmm0");
}

static inline void mm_aadd_sd(double *dst, __m128d val) {
    //_mm_aadd_sd(dst, val);
    __asm__ __volatile__ ("movupd %1, %%xmm0 \n\t"
            "vaaddsd %%xmm0, (%%rdx) \n\t"
            :
            : "d"(dst), "rm"(val)
            : "memory", "%xmm0");
}

#endif

#ifdef TEST_RAO_INTRINSIC

#include <stdio.h>
#include <assert.h>

int main() {
    volatile int64_t b = 0;
    int64_t *p = (int64_t*)&b;

    aadd64(p, 1);
    aadd64(p, 1);
    printf("%ld\n", b);

    aand64(p, 2);
    aand64(p, 2);
    printf("%ld\n", b);

    axor64(p, 3);
    axor64(p, 3);
    printf("%ld\n", b);

    aor64(p, 4);
    aor64(p, 4);
    printf("%ld\n", b);

    axchg_i64(p, 5);
    axchg_i64(p, 5);
    printf("%ld\n", b);

    axchgadd_i64(p, 6);
    axchgadd_i64(p, 6);
    printf("%ld\n", b);

    axchgdec_i64(p);
    axchgdec_i64(p);
    printf("%ld\n", b);

    axchginc_i64(p);
    axchginc_i64(p);
    printf("%ld\n", b);

    assert(b==17);

    float f = 0.0;
    __m128 fval = {1.0};
    mm_aadd_ss(&f, fval);
    mm_aadd_ss(&f, fval);
    printf("%f\n", f);

    double d = 0.0;
    __m128d val = {1.5};
    mm_aadd_sd(&d, val);
    mm_aadd_sd(&d, val);
    printf("%lf\n", d);
}

#endif
