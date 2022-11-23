#include <CL/sycl.hpp>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <getopt.h>
#include "level_zero/ze_api.h"
#include <sys/mman.h>
#include"../common_includes/rdtsc.h"
#include <time.h>
#include <sys/stat.h>
#include <immintrin.h>
#include <chrono>
#include "uncached.cpp"

constexpr size_t BUFSIZE = (1L<<20);
constexpr double NSEC_IN_SEC = 1000000000.0;

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

template<typename T>
T *get_mmap_address(T * device_ptr, size_t size, sycl::queue Q) {
    sycl::context ctx = Q.get_context();
    ze_ipc_mem_handle_t ze_ipc_handle;
    ze_result_t ret = zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx), device_ptr, &ze_ipc_handle);
    std::cout<<"zeMemGetIpcHandle return : " << ret << std::endl;
    assert(ret == ZE_RESULT_SUCCESS);
    int fd;
    memcpy(&fd, &ze_ipc_handle, sizeof(fd));
    std::cout << " fd " << fd << std::endl;
    struct stat statbuf;
    fstat(fd, &statbuf);
    std::cout << "requested size " << size << std::endl;
    std::cout << "fd size " << statbuf.st_size << std::endl;

    void *base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == (void *) -1) {
      std::cout << "mmap returned -1" << std::endl;
      std::cout << strerror(errno) << std::endl;  
    }
    assert(base != (void *) -1);
    return (T*)base;
}

/* command line mode arguments:
 *
 * storeflush iter setflagloc pollflagloc tocpumode fromcpumode
 *   tocpuloc 0 = device, 1 = host
 *   togpuloc 0 = device, 1 = host
 *   mode  0 = pointer, 1 = atomic, 2 = uncached
 */

void printduration(const char* name, sycl::event e)
{
  uint64_t start =
    e.get_profiling_info<sycl::info::event_profiling::command_start>();
  uint64_t end =
    e.get_profiling_info<sycl::info::event_profiling::command_end>();
  double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
  std::cout << name << " execution time: " << duration << " sec" << std::endl;
}

void check_ulong(const char *test, ulong *d, size_t count)
{
  for (int idx = 0; idx < count; idx += 1) {
    ulong val;
    val = idx;
    if (d[idx] != val) {
      printf("ulong %s error idx[%d] == %ld should be %ld\n", test, idx, d[idx], val);
    }
  }
  printf("ulong %s\n", test);
}

void check_ulong2(const char *test, ulong2 *d, size_t count)
{
  for (int idx = 0; idx < count; idx += 1) {
    ulong2 val;
    for (int i = 0; i < 1; i += 1) {
      val[i] = idx + (i*1000000);
      if (d[idx][i] != val[i]) {
	printf("ulong2 %s error idx[%d][%d] == %ld should be %ld\n", test, idx, i, d[idx][i], val[i]);
      }
    }
  }
  printf("ulong2 %s\n", test);
}

void check_ulong4(const char *test, ulong4 *d, size_t count)
{
  for (int idx = 0; idx < count; idx += 1) {
    ulong4 val;
    for (int i = 0; i < 1; i += 1) {
      val[i] = idx + (i*1000000);
      if (d[idx][i] != val[i]) {
	printf("ulong4 %s error idx[%d][%d] == %ld should be %ld\n", test, idx, i, d[idx][i], val[i]);
      }
    }
  }
  printf("ulong4 %s\n", test);
}

void check_ulong8(const char *test, ulong8 *d, size_t count)
{
  for (int idx = 0; idx < count; idx += 1) {
    ulong8 val;
    for (int i = 0; i < 1; i += 1) {
      val[i] = idx + (i*1000000);
      if (d[idx][i] != val[i]) {
	printf("ulong8 %s error idx[%d][%d] == %ld should be %ld\n", test, idx, i, d[idx][i], val[i]);
      }
    }
  }
  printf("ulong8 %s\n", test);
}

void kernel_fill_ulong(sycl::queue q, ulong *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong val = idx;
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}


void kernel_fill_ulong2(sycl::queue q, ulong2 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong2 val;
	  for (int i = 0; i < 2; i += 1) val[i] = idx + (i*1000000);
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}


void kernel_fill_ulong4(sycl::queue q, ulong4 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong4 val;
	  for (int i = 0; i < 4; i += 1) val[i] = idx + (i*1000000);
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}

void kernel_fill_ulong8(sycl::queue q, ulong8 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong8 val;
	  for (int i = 0; i < 8; i += 1) val[i] = idx + (i*1000000);
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}


void kernel_store_ulong(sycl::queue q, ulong *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong val = idx;
	  ucs_ulong(&d[idx], val);
	});
    });
  e.wait_and_throw();
}

void kernel_store_ulong2(sycl::queue q, ulong2 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong2 val;
	  for (int i = 0; i < 2; i += 1) val[i] = idx + (i*1000000);
	  ucs_ulong2(&d[idx],val);
	});
    });
  e.wait_and_throw();
}

void kernel_store_ulong4(sycl::queue q, ulong4 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong4 val;
	  for (int i = 0; i < 4; i += 1) val[i] = idx + (i*1000000);
	  ucs_ulong4(&d[idx],val);
	});
    });
  e.wait_and_throw();
}

void kernel_store_ulong8(sycl::queue q, ulong8 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong8 val;
	  for (int i = 0; i < 8; i += 1) val[i] = idx + (i*1000000);
	  ucs_ulong8(&d[idx],val);
	});
    });
  e.wait_and_throw();
}


void kernel_load_ulong(sycl::queue q, ulong *d, ulong *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = ucl_ulong(&s[idx]);
	});
    });
  e.wait_and_throw();
}


void kernel_load_ulong2(sycl::queue q, ulong2 *d, ulong2 *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = ucl_ulong2(&s[idx]);
	});
    });
  e.wait_and_throw();
}

void kernel_load_ulong4(sycl::queue q, ulong4 *d, ulong4 *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = ucl_ulong4(&s[idx]);
	});
    });
  e.wait_and_throw();
}

void kernel_load_ulong8(sycl::queue q, ulong8 *d, ulong8 *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = ucl_ulong8(&s[idx]);
	});
    });
  e.wait_and_throw();
}


// ############################

int main(int argc, char *argv[]) {
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
  sycl::queue Q(sycl::gpu_selector{}, prop_list);
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  // the *2 is for an old bug that is probably fixed
  uint64_t * host_mem = (uint64_t *) sycl::aligned_alloc_host(4096, BUFSIZE*2, Q);
  uint64_t * check_mem = (uint64_t *) sycl::aligned_alloc_host(4096, BUFSIZE*2, Q);
  uint64_t * device_mem = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE*2, Q);
  uint64_t * host_view_device_mem = get_mmap_address(device_mem, BUFSIZE, Q);
  std::cout << " host_mem " << host_mem << std::endl;
  std::cout << " device_mem " << device_mem << std::endl;
  std::cout << " host_view_device_mem " << host_view_device_mem << std::endl;
  int size = 0;
  int mode = 0;
  int mem = 0;
  if (argc < 4) exit(1);
  size = atoi(argv[1]);
  mode = atoi(argv[2]);
  mem = atoi(argv[3]);
  
  std::cout << " size " << size << std::endl;
  std::cout << " mode " << mode << std::endl;
  std::cout << " mem " << mem << std::endl;
  assert(size <= BUFSIZE);
  
  sycl::context ctx = Q.get_context();
  void * test_mem = NULL;
  if (mem == 1) {
    test_mem = device_mem;
  } else if (mem == 2) {
    test_mem = host_mem;
  }
 

  if (mode == 1) {
    size_t count = size / sizeof(ulong);
    ulong *test_area = (ulong *) test_mem;
    ulong *check_area = (ulong *) check_mem;
    Q.memset(test_area, 0xff, size);
    Q.wait_and_throw();
    kernel_fill_ulong(Q, test_area, count);
    Q.memcpy(check_area, test_area, size);
    Q.wait_and_throw();
    check_ulong("fill", check_area, count);
    Q.memset(test_area, 0xff, size);
    Q.wait_and_throw();
    kernel_store_ulong(Q, test_area, size);
    Q.memcpy(check_area, test_area, size);
    Q.wait_and_throw();
    check_ulong("store", check_area, count);
    memset(check_area, 0xff, size);
    kernel_fill_ulong(Q, test_area, count);
    kernel_load_ulong(Q, check_area, test_area, count);
    check_ulong("load", check_area, count);
  } else if (mode == 2) {
    size_t count = size / sizeof(ulong2);
    ulong2 *test_area = (ulong2 *) test_mem;
    ulong2 *check_area = (ulong2 *) check_mem;
    Q.memset(test_area, 0xff, size);
    Q.wait_and_throw();
    kernel_fill_ulong2(Q, test_area, count);
    Q.memcpy(check_area, test_area, size);
    Q.wait_and_throw();
    check_ulong2("fill", check_area, count);
    Q.memset(test_area, 0xff, size);
    Q.wait_and_throw();
    kernel_store_ulong2(Q, test_area, size);
    Q.memcpy(check_area, test_area, size);
    Q.wait_and_throw();
    check_ulong2("store", check_area, count);
    memset(check_area, 0xff, size);
    kernel_fill_ulong2(Q, test_area, count);
    kernel_load_ulong2(Q, check_area, test_area, count);
    check_ulong2("load", check_area, count);
  } else if (mode == 4) {
    size_t count = size / sizeof(ulong4);
    ulong4 *test_area = (ulong4 *) test_mem;
    ulong4 *check_area = (ulong4 *) check_mem;
    Q.memset(test_area, 0xff, size);
    Q.wait_and_throw();
    kernel_fill_ulong4(Q, test_area, count);
    Q.memcpy(check_area, test_area, size);
    Q.wait_and_throw();
    check_ulong4("fill", check_area, count);
    Q.memset(test_area, 0xff, size);
    Q.wait_and_throw();
    kernel_store_ulong4(Q, test_area, size);
    Q.memcpy(check_area, test_area, size);
    Q.wait_and_throw();
    check_ulong4("store", check_area, count);
    memset(check_area, 0xff, size);
    kernel_fill_ulong4(Q, test_area, count);
    kernel_load_ulong4(Q, check_area, test_area, count);
    check_ulong4("load", check_area, count);
  } else if (mode == 8) {
    size_t count = size / sizeof(ulong8);
    ulong8 *test_area = (ulong8 *) test_mem;
    ulong8 *check_area = (ulong8 *) check_mem;
    Q.memset(test_area, 0xff, size);
    Q.wait_and_throw();
    kernel_fill_ulong8(Q, test_area, count);
    Q.memcpy(check_area, test_area, size);
    Q.wait_and_throw();
    check_ulong8("fill", check_area, count);
    Q.memset(test_area, 0xff, size);
    Q.wait_and_throw();
    kernel_store_ulong8(Q, test_area, size);
    Q.memcpy(check_area, test_area, size);
    Q.wait_and_throw();
    check_ulong8("store", check_area, count);
    memset(check_area, 0xff, size);
    kernel_fill_ulong8(Q, test_area, count);
    kernel_load_ulong8(Q, check_area, test_area, count);
    check_ulong8("load", check_area, count);
  }
  
  return 0;
}

