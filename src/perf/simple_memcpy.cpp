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


// ############################

int main(int argc, char *argv[]) {
  unsigned long start_host_time, end_host_time;
  unsigned long start_device_time, end_device_time;
  struct timespec ts_start, ts_end;
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

  // modelled after Jeff Hammond code in PRK repo
  sycl::queue q;
  
  sycl::device dev = q.get_device();

  std::cout << "**max wg: " << dev.get_info<cl::sycl::info::device::max_work_group_size>()  << std::endl;


  uint64_t * dev_src;
  uint64_t * dev_dest;

  std::cout<<"selected device : "<< dev.get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<< dev.get_info<sycl::info::device::vendor>() << std::endl;
  uint device_frequency = (uint) dev.get_info<sycl::info::device::max_clock_frequency>();
  std::cout << "device frequency " << device_frequency << std::endl;

  dev_src = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, q);
  dev_dest = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, q);
  
  std::cout << " dev_src " << dev_src << std::endl;
  std::cout << " dev_dest" << dev_dest << std::endl;
  std::cout << " memory initialized " << std::endl;

  // Measure time for an empty kernel (with size 1)
  printf("csv,mode,size,sgsize,wgsize,count,duration,bandwidth\n");
    // size is in bytes
  int mode  = 0;
  ulong *s, *d;
  s = dev_src;
  d = dev_dest;
  for (size_t size = 32*sizeof(ulong); size < BUFSIZE; size <<= 1) {
    size_t iterations = size / sizeof(ulong);
    int max_wg_size = dev.get_info<cl::sycl::info::device::max_work_group_size>();
    printf ("max wg_size %d\n", max_wg_size);
    for (int sg_size = 16; sg_size <= 32; sg_size <<= 1) {
      int this_max_wg_size = max_wg_size;
      if (this_max_wg_size > iterations) this_max_wg_size = iterations;
      for (int wg_size = sg_size; wg_size <= this_max_wg_size; wg_size <<= 1) {
	double duration;
	int count;
	printf("node size %ld iterations %ld sg_size %d wg_size %d\n", size, iterations, sg_size, wg_size);
	fflush(stdout);
	// run for more and more counts until it takes more than 0.1 sec
	for (count = 1; count < (1 << 20); count <<= 1) {
	  clock_gettime(CLOCK_REALTIME, &ts_start);
	  for (int iter = 0; iter < count; iter += 1) {
	    auto e = q.submit([&](sycl::handler &h) {
		h.parallel_for(sycl::nd_range<1>(iterations, wg_size), [=](sycl::id<1> idx) [[intel::reqd_sub_group_size(32)]]{
		    d[idx] = s[idx];
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

  // check destination buffer
      
  std::cout<<"kernel returned" << std::endl;
  return 0;
}

