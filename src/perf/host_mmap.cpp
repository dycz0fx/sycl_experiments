#include<CL/sycl.hpp>
#include<stdlib.h>
#include<unistd.h>
#include<thread>
#include<iostream>
#include"level_zero/ze_api.h"
#include<CL/sycl/backend/level_zero.hpp>
#include<sys/mman.h>
#include"../common_includes/rdtsc.h"
#include <time.h>

constexpr size_t BUFSIZE = 32;
constexpr uint64_t cpu_to_gpu_index = 0;
constexpr uint64_t gpu_to_cpu_index = 8;
#define cpu_relax() asm volatile("rep; nop")

template<typename T>
T *get_mmap_address(T * device_ptr, size_t size, sycl::queue Q) {
    sycl::context ctx = Q.get_context();
    ze_ipc_mem_handle_t ze_ipc_handle;
    ze_result_t ret = zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx), device_ptr, &ze_ipc_handle);
    std::cout<<"zeMemGetIpcHandle return : "<<ret<<"\n";
    assert(ret == ZE_RESULT_SUCCESS);

    int fd;
    memcpy(&fd, &ze_ipc_handle, sizeof(fd));
    void *base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(base != (void *) -1);

    return (T*)base;
}

int main(int argc, char *argv[]) {
  uint64_t count;
  if (argc < 2) exit(1);
  count = atol(argv[1]);
  sycl::queue Q;
  std::cout<<"count : "<< count << std::endl;
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() <<"\n";
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() <<"\n";

  uint64_t * device_mem = sycl::malloc_device<uint64_t>(BUFSIZE, Q);
  std::array<uint64_t,BUFSIZE> host_array;
  memset(&host_array[0], 0, sizeof(uint64_t) * BUFSIZE);
  
  //create mmap mapping of usm device memory on host
  sycl::context ctx = Q.get_context();
  
  uint64_t *host_map = get_mmap_address((uint64_t *) device_mem, sizeof(uint64_t) * BUFSIZE, Q);
  
  std::cout << " device_mem " << device_mem << "\n";
  std::cout << " host_map " << host_map << "\n";
  
  auto e = Q.submit([&](sycl::handler &h) {
      h.memcpy(device_mem, &host_array[0], sizeof(uint64_t) * BUFSIZE);
    });
  e.wait_and_throw();
    
  std::cout<<"kernel going to launch\n";

  e = Q.submit([&](sycl::handler &h) {
      //   sycl::stream os(1024, 128, h);
      //h.single_task([=]() {
      h.parallel_for(sycl::nd_range<1>{{1}, {1}}, [=](sycl::nd_item<1> idx) {
	  //  os<<"kernel start\n";
	  uint64_t prev = (uint64_t) -1L;
	  sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(device_mem[cpu_to_gpu_index]);
	  sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> gpu_to_cpu(device_mem[gpu_to_cpu_index]);
	  do {
	    uint64_t temp;
	    for (;;) {
	      temp = cpu_to_gpu.load();
	      if (temp != prev) break;
	    }
	    prev = temp;
	    gpu_to_cpu.store(temp);
	  } while (prev < count-1 );
	  // os<<"kernel exit\n";
        });
    });
    std::cout<<"kernel launched\n";
    {
      struct timespec ts_start;
      clock_gettime(CLOCK_REALTIME, &ts_start);
      unsigned long start_time = rdtsc();
      volatile uint64_t *c_to_g = & host_map[cpu_to_gpu_index];
      volatile uint64_t *g_to_c = & host_map[gpu_to_cpu_index];

      for (int i = 0; i < count; i += 1) {
	*c_to_g = i;
	while (i != *g_to_c) cpu_relax();
      }
      unsigned long end_time = rdtscp();
      struct timespec ts_end;
      clock_gettime(CLOCK_REALTIME, &ts_end);
      double elapsed = (ts_start.tv_sec - ts_start.tv_sec) * 1000000000 +
	(ts_end.tv_nsec - ts_start.tv_nsec);
      std::cout << "count " << count << " tsc each " << (end_time - start_time) / count << "\n";
      std::cout << "count " << count << " tsc each " << elapsed / count << "\n";
    }

      
    e.wait_and_throw();
    std::cout<<"kernel over \n";
    sycl::free(device_mem, Q);
    return 0;
}

