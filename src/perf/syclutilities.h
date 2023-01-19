#ifndef SYCLUTILITIES_H
#define SYCLUTILITIES_H
#include <stdarg.h>
#include <stdio.h>

#define cpu_relax() asm volatile("rep; nop")

void dbprintf(int line, const char *format, ...)
{
  va_list arglist;
  printf("line %d: ", line);
  va_start(arglist, format);
  vprintf(format, arglist);
  va_end(arglist);
}

#define DP(...) if (DEBUG) dbprintf(__LINE__, __VA_ARGS__)

#define HERE()
#define CHERE(name) if (DEBUG) std::cout << name << " " <<  __FUNCTION__ << ": " << __LINE__ << std::endl; 

constexpr double NSEC_IN_SEC = 1000000000.0;


/* clock stuff */

#ifdef __SYCL_DEVICE_ONLY__
extern "C" {
  SYCL_EXTERNAL ulong intel_get_cycle_counter() __attribute__((overloadable));
}

ulong get_cycle_counter() {
  return intel_get_cycle_counter();
}
#else
ulong get_cycle_counter() {
  return 0xDEADBEEF;
}
#endif // __SYCL_DEVICE_ONLY__

/* obtain a host mapping for a device memory buffer */

template<typename T>
T *get_mmap_address(T * device_ptr, size_t size, sycl::queue Q);

// print runtime of a kernel
void printduration(const char* name, sycl::event e);

ulong kernelcycles(sycl::event e);

ulong max_frequency(sycl::queue q);

#endif // ifndef SYCLUTILITIES_H
