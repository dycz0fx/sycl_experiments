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

constexpr size_t CTLSIZE = (4096);
constexpr double NSEC_IN_SEC = 1000000000.0;

#ifndef SYCL_EXT_ONEAPI_DEVICE_GLOBAL
#error "no global"
#endif

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
  struct timespec ts_start, ts_end;
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
  sycl::queue Q(sycl::gpu_selector{}, prop_list);
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;

  sycl::ext::oneapi::experimental::device_global int gint1;
  sycl::ext::oneapi::experimental::device_global int gint2;
  
  e = Q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(1), [=](sycl::group<1> grp) {
	  int x = gint1;
	  gint2 = x;
	});
      });
  std::cout<<"kernel launched" << std::endl;
  // cpu part
  clock_gettime(CLOCK_REALTIME, &ts_start);
  // host code here
  clock_gettime(CLOCK_REALTIME, &ts_end);
  double elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * NSEC_IN_SEC
    + 
	((double) (ts_end.tv_nsec - ts_start.tv_nsec));
  std::cout << "iter " << iter << " nsed each " << elapsed / iter << std::endl;
  e.wait_and_throw();
  printduration("gpu kernel ", e);
    /* common cleanup */
  std::cout<<"kernel returned" << std::endl;
  return 0;
}

