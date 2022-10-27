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

constexpr size_t CTLSIZE = (4096);
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



#endif
static inline void block_store(ulong  *base, int immElemOff, ulong  val)
{
#ifdef __SYCL_DEVICE_ONLY__
#if 1
  __builtin_IB_lsc_store_global_ulong (base, immElemOff, val, LSC_STCC_L1UC_L3UC);
  //__builtin_IB_lsc_load_global_ulong (base, immElemOff, LSC_LDCC_L1UC_L3UC);
#else
  *base = val;
  val = *((volatile ulong *) base);
#endif
#else
  *base = val;
#endif
}

static inline ulong block_load(ulong  *base, int immElemOff)
{
  ulong v;
#ifdef __SYCL_DEVICE_ONLY__
#if 1
  v = __builtin_IB_lsc_load_global_ulong (base, immElemOff, LSC_LDCC_L1UC_L3UC);
#else
  v = *((volatile ulong *) base);
#endif
#else
  v = *base;
#endif
  return(v);
}

void printduration(const char* name, sycl::event e)
  {
    uint64_t start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end =
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
  sycl::queue Q(sycl::gpu_selector{}, prop_list);
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  uint device_frequency = (uint)Q.get_device().get_info<sycl::info::device::max_clock_frequency>();
  std::cout << "device frequency " << device_frequency << std::endl;
  // the *2 is for an old bug that is probably fixed
  uint64_t * host_mem = (uint64_t *) sycl::aligned_alloc_host(4096, CTLSIZE*2, Q);
  uint64_t * device_mem = (uint64_t *) sycl::aligned_alloc_device(4096, CTLSIZE*2, Q);
  uint64_t * host_view_device_mem = get_mmap_address(device_mem, CTLSIZE, Q);
  std::cout << " host_mem " << host_mem << std::endl;
  std::cout << " device_mem " << device_mem << std::endl;
  std::cout << " host_view_device_mem " << host_view_device_mem << std::endl;
  int tocpuloc = 0;
  int togpuloc = 0;
  int tocpumode = 0;
  int togpumode = 0;
  int iter = 0;
  if (argc < 6) exit(1);
  iter = atoi(argv[1]);
  tocpuloc = atoi(argv[2]);
  togpuloc = atoi(argv[3]);
  tocpumode = atoi(argv[4]);
  togpumode = atoi(argv[5]);
  std::cout << " iter " << iter << std::endl;
  std::cout << " tocpuloc " << tocpuloc << std::endl;
  std::cout << " togpuloc " << togpuloc << std::endl;
  std::cout << " tocpumode " << tocpumode << std::endl;
  std::cout << " togpumode " << togpumode << std::endl;
  
  sycl::context ctx = Q.get_context();

  // initialize host_mem
  memset(host_mem, 0xff, CTLSIZE);
  // initialize device_mem
  auto e = Q.submit([&](sycl::handler &h) {
      h.memcpy(device_mem, host_mem, CTLSIZE);
    });
  e.wait_and_throw();
  std::cout << " memory initialized " << std::endl;
  // direction device to host
  uint64_t *device_set_flag = NULL;
  volatile uint64_t *host_poll_flag = NULL;
  // direction host to device
  volatile uint64_t *host_set_flag = NULL;
  uint64_t *device_poll_flag = NULL;
  
  if (tocpuloc == 0) {
    device_set_flag = &host_mem[0];
    host_poll_flag = &host_mem[0];
  } else if (tocpuloc == 1) {
    device_set_flag = &device_mem[0];
    host_poll_flag = &host_view_device_mem[0];
  } else {
    assert(0);
  }
  if (togpuloc == 0) {
    host_set_flag = &host_mem[8];
    device_poll_flag = &host_mem[8];
  } else if (togpuloc == 1) {
    host_set_flag = &host_view_device_mem[8];
    device_poll_flag = &device_mem[8];
  } else {
    assert(0);
  }
  
  
  e = Q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(1), [=](sycl::group<1> grp) {
	  sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(device_set_flag[0]);
	  sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> gpu_to_cpu(device_poll_flag[0]);
	  unsigned long lstart_device_time, lend_device_time;
	  lstart_device_time = get_cycle_counter();
	  for (int i = 0; i < iter; i += 1) {
	    if (tocpumode == 0) *device_set_flag = i;
	    else if (tocpumode == 1) gpu_to_cpu.store(i);
	    else if (tocpumode == 2) block_store(device_set_flag, 0, i);
	    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
	    int timeout = 0;
	    for (;;) {
	      uint64_t val;
	      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
	      if (togpumode == 0) {
		val = *device_poll_flag;
	      } else if (togpumode == 1) {
		val = cpu_to_gpu.load();
	      } else if (togpumode == 2) {
		val = block_load(device_poll_flag, 0);
	      }
	      //	      if (timeout > 1000000) break;
	      if (timeout < 1000000) timeout += 1;
	      if (val == i) break;
	    }
	  }
	  lend_device_time = get_cycle_counter();
	  host_mem[16] = lstart_device_time;
	  host_mem[24] = lend_device_time;
	});
      });
  std::cout<<"kernel launched" << std::endl;
  // cpu part
  clock_gettime(CLOCK_REALTIME, &ts_start);
  start_host_time = rdtsc();
  for (int i = 0; i < iter; i += 1) {
    while (*host_poll_flag != i);
    //    std::cout << "got " << i << std::endl;
    *host_set_flag = i;
    _mm_sfence();
  }
  end_host_time = rdtsc();
  clock_gettime(CLOCK_REALTIME, &ts_end);
  double elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * NSEC_IN_SEC
    + 
	((double) (ts_end.tv_nsec - ts_start.tv_nsec));
  std::cout << "iter " << iter << " nsed each " << elapsed / iter << std::endl;
  e.wait_and_throw();
  start_device_time = host_mem[16];
  end_device_time = host_mem[24];
  printduration("gpu kernel ", e);
  std::cout << "rdtsc per iteration :" << (end_host_time - start_host_time) / iter << std::endl;
  std::cout << "device clock per iteration :" << (end_device_time - start_device_time) / iter << std::endl;
    /* common cleanup */
  std::cout<<"kernel returned" << std::endl;
  return 0;
}

