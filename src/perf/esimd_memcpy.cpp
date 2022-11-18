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

constexpr size_t BUFSIZE = (1L << 32);   // 4 GB
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

/* command line mode arguments:
 *
 * storeflush iter setflagloc pollflagloc tocpumode fromcpumode
 *   tocpuloc 0 = device, 1 = host
 *   togpuloc 0 = device, 1 = host
 *   mode  0 = pointer, 1 = atomic, 2 = uncached
 */



void printduration(const char* name, sycl::event e)
  {
    ulong start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
    ulong end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
    std::cout << name << " execution time: " << duration << " sec" << std::endl;
  }

namespace lsc_esimd = sycl::ext::intel::esimd;

template <typename T, int VEC_LEN, lsc_esimd::lsc_data_size DS, lsc_esimd::cache_hint LD_L1H, lsc_esimd::cache_hint LD_L3H, lsc_esimd::cache_hint ST_L1H, lsc_esimd::cache_hint ST_L3H>
class StreamCopyESIMDLscKernel {
  int n;
  T* array_src;
  T* array_dest;
 
  StreamCopyESIMDLscKernel(int n, T* array_src, T* array_dest)
    : n(n), array_src(array_src), array_dest(array_dest) {}
 
  void operator()(sycl::id<1> ii) const SYCL_ESIMD_KERNEL {
    int i = ii.get(0);
    esimd::simd<T, VEC_LEN> vec_a;
 
    vec_a = lsc_esimd::lsc_block_load<T, VEC_LEN, DS, LD_L1H, LD_L3H>(array_src + i * VEC_LEN);
 
    lsc_esimd::lsc_block_store<T, VEC_LEN, DS, ST_L1H, ST_L3H>(array_dest + i * VEC_LEN, vec_a);
  }
};
 
lsc_esimd::cache_hint (sycl::ext::intel::esimd namespace):
 
enum class cache_hint : uint8_t {
  none = 0,
  uncached = 1,
  cached = 2,
  write_back = 3,
  write_through = 4,
  streaming = 5,
  read_invalidate = 6
};
 
// ############################

int main(int argc, char *argv[]) {
  unsigned long start_host_time, end_host_time;
  unsigned long start_device_time, end_device_time;
  struct timespec ts_start, ts_end;
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

  // modelled after Jeff Hammond code in PRK repo
  sycl::queue q;
  
  uint64_t *dev_src;
  uint64_t *dev_dest;
  uint64_t *host_src;
  uint64_t *host_dest;

  std::cout<<"selected device : "<<q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<q.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  uint device_frequency = (uint)q.get_device().get_info<sycl::info::device::max_clock_frequency>();
  std::cout << "device frequency " << device_frequency << std::endl;

  dev_src = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, q);
  dev_dest = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, q);
  
  host_src = (ulong *) sycl::aligned_alloc_host(4096, BUFSIZE, q);
  host_dest = (ulong *) sycl::aligned_alloc_host(4096, BUFSIZE, q);
  std::cout << " dev_src " << dev_src << std::endl;
  std::cout << " dev_dest " << dev_dest << std::endl;
  std::cout << " host_src " << host_src << std::endl;
  std::cout << " host_dest " << host_dest << std::endl;

  int iter = 0;
  for (int i = 0; i < BUFSIZE/sizeof(ulong); i += 1) host_src[i] = i;
  memset(host_dest, 0xff, BUFSIZE);
  // initialize device_mem
  q.memcpy(dev_src, host_src, BUFSIZE);
  q.wait_and_throw();
  q.memcpy(dev_dest, host_dest, BUFSIZE);
  q.wait_and_throw();

  std::cout << " memory initialized " << std::endl;

  // Measure time for an empty kernel (with size 1)
  printf("csv,mode,size,sgsize,wgsize,count,duration,bandwidth\n");
    // size is in bytes
  for (int mode = 0; mode < 5; mode += 1) {
    ulong *s, *d;
    s = dev_src;
    d = dev_dest;

    for (size_t size = 32; size < BUFSIZE; size <<= 1) {
      size_t iterations = size / sizeof(ulong);
      int max_wg_size = 1024;
      
      for (int sg_size = 16; sg_size <= 32; sg_size <<= 1) {
	for (int wg_size = sg_size; wg_size < max_wg_size; wg_size <<= 1) {
	  double duration;
	  int count;
	  size_t loc_loop = iterations / (sg_size * wg_size);

	  printf("node size %ld sg_size %d wg_size %d\n", size, sg_size, wg_size);
	  fflush(stdout);
	  // run for more and more counts until it takes more than 0.1 sec
	  for (count = 1; count < (1 << 20); count <<= 1) {
	    clock_gettime(CLOCK_REALTIME, &ts_start);
	    for (int iter = 0; iter < count; iter += 1) {
	      auto e = q.submit([&](sycl::handler &h) {
		  h.parallel_for_work_group(sycl::range(1), sycl::range(wg_size), [=](sycl::group<1> grp) {
		      grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
			  int j = it.get_global_id()[0];
			  for (int k = j * loc_loop; k < (j+1) * loc_loop; k += 1)
			    d[k] = s[k];
			});
		    });
		});
	      e.wait_and_throw();
	    }
	    // sequential
	    clock_gettime(CLOCK_REALTIME, &ts_end);
	    duration = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	      ((double) (ts_end.tv_nsec - ts_start.tv_nsec));
	    
	    duration /= 1000000000.0;
	    if (duration > 0.1) {
	      break;
	    }
	  }
	  double per_iter = duration / count;
	  double bw = size / (per_iter);
	  double bw_mb = bw / 1000000.0;
	  printf("csv,%d,%ld,%d,%d,%d,%f,%f\n", mode, size, sg_size, wg_size, count, duration, bw_mb);
	}
      }
    }
  }
  // check destination buffer
      
  std::cout<<"kernel returned" << std::endl;
  return 0;
}

