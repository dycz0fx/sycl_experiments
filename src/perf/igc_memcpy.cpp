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

constexpr size_t BUFSIZE = (1<<26);   // 64 MB
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


typedef sycl::vec<ulong,2> ulong2;
typedef sycl::vec<ulong,4> ulong4;
typedef sycl::vec<ulong,8> ulong8;
// ############################
// copied from https://github.com/intel/intel-graphics-compiler/blob/master/IGC/BiFModule/Implementation/IGCBiF_Intrinsics_Lsc.cl#L90
// Load message caching control
enum LSC_LDCC {
    LSC_LDCC_DEFAULT      = 0,
    LSC_LDCC_L1UC_L3UC    = 1,   // Override to L1 uncached and L3 uncached
    LSC_LDCC_L1UC_L3C     = 2,   // Override to L1 uncached and L3 cached
    LSC_LDCC_L1C_L3UC     = 3,   // Override to L1 cached and L3 uncached
    LSC_LDCC_L1C_L3C      = 4,   // Override to L1 cached and L3 cached
    LSC_LDCC_L1S_L3UC     = 5,   // Override to L1 streaming load and L3 uncached
    LSC_LDCC_L1S_L3C      = 6,   // Override to L1 streaming load and L3 cached
    LSC_LDCC_L1IAR_L3C    = 7,   // Override to L1 invalidate-after-read, and L3 cached
};
//#include <IGCBiF_Intrinsics_Lsc.cl>
//
enum LSC_STCC {
    LSC_STCC_DEFAULT      = 0,
    LSC_STCC_L1UC_L3UC    = 1,   // Override to L1 uncached and L3 uncached
    LSC_STCC_L1UC_L3WB    = 2,   // Override to L1 uncached and L3 written back
    LSC_STCC_L1WT_L3UC    = 3,   // Override to L1 written through and L3 uncached
    LSC_STCC_L1WT_L3WB    = 4,   // Override to L1 written through and L3 written back
    LSC_STCC_L1S_L3UC     = 5,   // Override to L1 streaming and L3 uncached
    LSC_STCC_L1S_L3WB     = 6,   // Override to L1 streaming and L3 written back
    LSC_STCC_L1WB_L3WB    = 7,   // Override to L1 written through and L3 written back
};
#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_ulong (ulong  *base, int immElemOff, ulong  val, enum LSC_STCC cacheOpt); //D64V1

SYCL_EXTERNAL extern "C" ulong   __builtin_IB_lsc_load_global_ulong (ulong  *base, int immElemOff, enum LSC_LDCC cacheOpt); //D64V1

// vector 2
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_ulong2 (ulong2  *base, int immElemOff, ulong2  val, enum LSC_STCC cacheOpt); //D64V1

SYCL_EXTERNAL extern "C" ulong2   __builtin_IB_lsc_load_global_ulong2 (ulong2  *base, int immElemOff, enum LSC_LDCC cacheOpt); //D64V1

// vector 4
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_ulong4 (ulong4  *base, int immElemOff, ulong4  val, enum LSC_STCC cacheOpt); //D64V1

SYCL_EXTERNAL extern "C" ulong4   __builtin_IB_lsc_load_global_ulong4 (ulong4  *base, int immElemOff, enum LSC_LDCC cacheOpt); //D64V1

// vector 8
SYCL_EXTERNAL extern "C" void  __builtin_IB_lsc_store_global_ulong8 (ulong8  *base, int immElemOff, ulong8  val, enum LSC_STCC cacheOpt); //D64V1

SYCL_EXTERNAL extern "C" ulong8   __builtin_IB_lsc_load_global_ulong8 (ulong8  *base, int immElemOff, enum LSC_LDCC cacheOpt); //D64V1

#endif

static inline void block_copy(ulong  *d, ulong *s)
{
  ulong v;
#ifdef __SYCL_DEVICE_ONLY__
  v = __builtin_IB_lsc_load_global_ulong (s, 0, LSC_LDCC_L1UC_L3UC);
  __builtin_IB_lsc_store_global_ulong (d, 0, v, LSC_STCC_L1UC_L3UC);
#else
  v = *s;
  *d = v;
#endif
}

// vector 2

static inline void block_copy2(ulong2  *d, ulong2 *s)
{
  ulong2 v;
#ifdef __SYCL_DEVICE_ONLY__
  v = __builtin_IB_lsc_load_global_ulong2 (s, 0, LSC_LDCC_L1UC_L3UC);
  __builtin_IB_lsc_store_global_ulong2 (d, 0, v, LSC_STCC_L1UC_L3UC);
#else
  v = *s;
  *d = v;
#endif
}

// vector 4
static inline void block_copy4(ulong4  *d, ulong4 *s)
{
  ulong4 v;
#ifdef __SYCL_DEVICE_ONLY__
  v = __builtin_IB_lsc_load_global_ulong4 (s, 0, LSC_LDCC_L1UC_L3UC);
  __builtin_IB_lsc_store_global_ulong4 (d, 0, v, LSC_STCC_L1UC_L3UC);
#else
  v = *s;
  *d = v;
#endif
}

// vector 8

static inline void block_copy8(ulong8  *d, ulong8 *s)
{
  ulong8 v;
#ifdef __SYCL_DEVICE_ONLY__
  v = __builtin_IB_lsc_load_global_ulong8 (s, 0, LSC_LDCC_L1UC_L3UC);
  __builtin_IB_lsc_store_global_ulong8 (d, 0, v, LSC_STCC_L1UC_L3UC);
#else
  v = *s;
  *d = v;
#endif
}


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
  sycl::queue Q(sycl::gpu_selector_v, prop_list);
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  uint device_frequency = (uint)Q.get_device().get_info<sycl::info::device::max_clock_frequency>();
  std::cout << "device frequency " << device_frequency << std::endl;

  ulong * host_src = (ulong *) sycl::aligned_alloc_host(4096, BUFSIZE, Q);
  ulong * host_dest = (ulong *) sycl::aligned_alloc_host(4096, BUFSIZE, Q);

  ulong * dev_src = (ulong *) sycl::aligned_alloc_device(4096, BUFSIZE, Q);
  ulong * dev_dest = (ulong *) sycl::aligned_alloc_device(4096, BUFSIZE, Q);
 
  std::cout << " host_src" << host_src << std::endl;
  std::cout << " host_dest" << host_dest << std::endl;
  std::cout << " dev_src " << dev_src << std::endl;
  std::cout << " dev_dest " << dev_dest << std::endl;

  int iter = 0;
  for (int i = 0; i < BUFSIZE/sizeof(ulong); i += 1) host_src[i] = i;
  memset(host_dest, 0xff, BUFSIZE);
  // initialize device_mem
  Q.memcpy(dev_src, host_src, BUFSIZE);
  Q.wait_and_throw();
  Q.memcpy(dev_dest, host_dest, BUFSIZE);
  Q.wait_and_throw();

  std::cout << " memory initialized " << std::endl;


  // Measure time for an empty kernel (with size 1)
  printf("csv, vec, size, threads,  count, duration, bandwidth MB/s\n");
  int vectorlength = 1;
    // size is in bytes
  for (size_t size = 1; size < BUFSIZE; size <<= 1) {
    int iterations = size / (sizeof(ulong) * vectorlength);
    int maxthreads = iterations;
    if (maxthreads > 1024) maxthreads = 1024;
    for (int threads = 1; threads <= maxthreads; threads <<= 1) {
      // locloop is in ulong
      ulong loc_loop = iterations / threads;
      double duration;
      int count;
      // run for more and more counts until it takes more than 0.1 sec
      for (count = 1; count < (1 << 20); count <<= 1) {
	clock_gettime(CLOCK_REALTIME, &ts_start);
	for (int iter = 0; iter < count; iter += 1) {
	  auto e = Q.submit([&](sycl::handler &h) {
	      h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
		  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		      int j = it.get_global_id()[0];
		      for (int k = j * loc_loop; k < (j+1) * loc_loop; k += 1) {
			dev_dest[k] = dev_src[k];
		      }
		      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
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
      printf("csv, %d, %ld, %d, %d, %f, %f\n", 0, size, threads, count, duration, bw_mb);
    }
  }

  for(int vectorlength = 1; vectorlength <= 8; vectorlength <<= 1) {  
    // size is in bytes
    for (size_t size = 1; size < BUFSIZE; size <<= 1) {
      int iterations = size / (sizeof(ulong) * vectorlength);
      int maxthreads = iterations;
      if (maxthreads > 1024) maxthreads = 1024;
      for (int threads = 1; threads <= maxthreads; threads <<= 1) {
	// locloop is in ulong
	ulong loc_loop = iterations / threads;
	double duration;
	int count;
	// run for more and more counts until it takes more than 0.1 sec
	for (count = 1; count < (1 << 20); count <<= 1) {
	  clock_gettime(CLOCK_REALTIME, &ts_start);
	  if (vectorlength == 1) {
	    for (int iter = 0; iter < count; iter += 1) {
	      auto e = Q.submit([&](sycl::handler &h) {
		  h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
		      grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
			  int j = it.get_global_id()[0];
			  for (int k = j * loc_loop; k < (j+1) * loc_loop; k += 1) {
			    block_copy(&dev_dest[k], &dev_src[k]);
			  }
			  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
			});
		    });
		});
	      e.wait_and_throw();
	    }
	  } else if (vectorlength == 2) {
	    for (int iter = 0; iter < count; iter += 1) {
	      auto e = Q.submit([&](sycl::handler &h) {
		  h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
		      grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
			  int j = it.get_global_id()[0];
			  for (int k = j * loc_loop; k < (j+1) * loc_loop; k += 1) {
			    block_copy2((ulong2 *) &dev_dest[k], (ulong2 *) &dev_src[k]);
			  }
			  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
			});
		    });
		});
	      e.wait_and_throw();
	    }
	  } else if (vectorlength == 4) {
	    for (int iter = 0; iter < count; iter += 1) {
	      auto e = Q.submit([&](sycl::handler &h) {
		  h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
		      grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
			  int j = it.get_global_id()[0];
			  for (int k = j * loc_loop; k < (j+1) * loc_loop; k += 1) {
			    block_copy4((ulong4 *) &dev_dest[k], (ulong4 *) &dev_src[k]);
			  }
			  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
			});
		    });
		});
	      e.wait_and_throw();
	    }
	  } else if (vectorlength == 8) {
	    for (int iter = 0; iter < count; iter += 1) {
	      auto e = Q.submit([&](sycl::handler &h) {
		  h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
		      grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
			  int j = it.get_global_id()[0];
			  for (int k = j * loc_loop; k < (j+1) * loc_loop; k += 1) {
			    block_copy8((ulong8 *) &dev_dest[k], (ulong8 *) &dev_src[k]);
			  }
			  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
			});
		    });
		});
	      e.wait_and_throw();
	    }
	  } else {
	    assert(0);
	  }
	  // vector 1

	  
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
	printf("csv, %d, %ld, %d, %d, %f, %f\n", vectorlength, size, threads, count, duration, bw_mb);
      }
    }
  }

  // check destination buffer
  
  std::cout<<"kernel returned" << std::endl;
  return 0;
}

