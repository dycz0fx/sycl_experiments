#include "stddef.h"

void movget_memcpy(void *dst, void *src, size_t count);
void avx_memcpy(void *dst, void *src, size_t count);
void movget_memcpy_noavx(void *dst, void *src, size_t count);

