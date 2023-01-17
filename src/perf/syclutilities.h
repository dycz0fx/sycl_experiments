#ifndef SYCLUTILITIES_H
#define SYCLUTILITIES_H


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

#endif // ifndef SYCLUTILITIES_H
