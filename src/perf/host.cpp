#include<CL/sycl.hpp>
#include<stdlib.h>
#include<unistd.h>
#include<thread>
#include<iostream>
#include"level_zero/ze_api.h"
#include<CL/sycl/backend/level_zero.hpp>
#include<sys/mman.h>
#include"../common_includes/rdtsc.h"

constexpr size_t T = 32;
constexpr uint64_t cpu_to_gpu_index = 0;
constexpr uint64_t gpu_to_cpu_index = 8;

int main(int argc, char *argv[]) {
  uint64_t count;
  if (argc < 2) exit(1);
  count = atol(argv[1]);
  sycl::queue Q;
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() <<"\n";
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() <<"\n";
  
  uint64_t * device_mem = sycl::malloc_device<uint64_t>(T, Q);
  std::array<uint64_t,T> host_array;
  memset(&host_array[0], 0, sizeof(uint64_t) * T);
  
  //create mmap mapping of usm device memory on host
  sycl::context ctx = Q.get_context();
  ze_ipc_mem_handle_t ze_ipc_handle;
  ze_result_t ret = zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx), device_mem, &ze_ipc_handle);
  std::cout << "zeMemGetIpcHandle return : " << ret <<" \n";
  assert(ret == ZE_RESULT_SUCCESS);

  int fd;
  memcpy(&fd, &ze_ipc_handle, sizeof(fd));
  uint64_t *host_map = (uint64_t *)  mmap(NULL, sizeof(size_t)*T, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  assert(host_map != (uint64_t *) -1);
  //mapping created
  
  std::cout << " device_mem " << device_mem << "\n";
  std::cout << " host_map " << host_map << "\n";
  
  auto e = Q.submit([&](sycl::handler &h) {
      h.memcpy(device_mem, &host_array[0], sizeof(uint64_t) * T);
    });
  e.wait_and_throw();
    
  std::cout<<"kernel going to launch\n";

  e = Q.submit([&](sycl::handler &h) {
      sycl::stream os(1024, 128, h);
      //h.single_task([=]() {
      h.parallel_for(sycl::nd_range<1>{{1}, {1}}, [=](sycl::nd_item<1> idx) {
	  os<<"kernel start\n";
	  uint64_t prev;
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
	  } while (prev != count);
	  os<<"kernel exit\n";
        });
    });
    std::cout<<"kernel launched\n";
    {
      unsigned long start_time = rdtsc();
      sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(host_map[cpu_to_gpu_index]);
      sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> gpu_to_cpu(host_map[gpu_to_cpu_index]);
      for (int i = 0; i < count; i += 1) {
	cpu_to_gpu.store(i);
	while (i != gpu_to_cpu.load());
      }
      unsigned long end_time = rdtscp();
      std::cout << "count " << count << " tsc each " << count / (end_time - start_time) << "\n";
    }

      
    e.wait_and_throw();
    std::cout<<"kernel over \n";
    sycl::free(device_mem, Q);
    return 0;
}

