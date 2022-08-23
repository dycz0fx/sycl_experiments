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
  std::cout<<"count : "<< count << std::endl;
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  
  uint64_t * host_mem = sycl::malloc_host<uint64_t>(T, Q);
  std::array<uint64_t,T> host_array{};

  //create mmap mapping of usm device memory on host
  sycl::context ctx = Q.get_context();

  uint64_t *host_map = (uint64_t *) host_mem;

  std::cout << " host_mem " << host_mem << std::endl;
  std::cout << " host_map " << host_map << std::endl;
  
  auto e = Q.submit([&](sycl::handler &h) {
      h.memcpy(host_mem, host_array.data(), sizeof(uint64_t) * T);
    });
  e.wait_and_throw();
    
  std::cout<<"kernel going to launch" << std::endl;

  e = Q.submit([&](sycl::handler &h) {
      //sycl::stream os(1024, 128, h);
      //h.single_task([=]() {
      h.parallel_for(sycl::nd_range<1>{{1}, {1}}, [=](sycl::nd_item<1> idx) {
	  //	  os<<"kernel start" << sycl::stream_manipulator::endl;
	  uint64_t prev = (uint64_t) -1L;
	  sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(host_mem[cpu_to_gpu_index]);
	  sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> gpu_to_cpu(host_mem[gpu_to_cpu_index]);
#if 0	  
	  do {
	    uint64_t temp;
	    for (;;) {
	      temp = cpu_to_gpu.load();
	      if (temp != prev) break;
	    }
	    //os << "got " << temp << sycl::stream_manipulator::endl;
	    prev = temp;
	    gpu_to_cpu.store(temp);
	  } while (prev < count-1);
#endif
	  for (int i = 0; i < 1024; i += 1) host_mem[i] = i;
	  //os<<"kernel exit"  << sycl::stream_manipulator::endl;
        });
    });
  std::cout<<"kernel launched" << std::endl;
    e.wait_and_throw();
    std::cout<<"kernel over" << std::endl;


    {
      unsigned long start_time = rdtsc();
      sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(host_map[cpu_to_gpu_index]);
      sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> gpu_to_cpu(host_map[gpu_to_cpu_index]);
#if 0
      for (int i = 0; i < count; i += 1) {
	cpu_to_gpu.store(i);
	while (i != gpu_to_cpu.load());
      }
#endif
      for (int i = 0; i < 1024; i += 1) {
	if (host_mem[i] != i)
	  std::cout << "err idx " << i << " is " << host_mem[i] << std::endl;
      }
      unsigned long end_time = rdtscp();
      std::cout << "count " << count << " tsc each " << (end_time - start_time) / count << std::endl;
    }

      

    sycl::free(host_mem, Q);
    return 0;
}

