#include <CL/sycl.hpp>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

constexpr size_t CTLSIZE = (4096);

int main(int argc, char *argv[]) {
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
  sycl::queue Q(sycl::gpu_selector{}, prop_list);
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;


  uint64_t * ctl_mem = (uint64_t *) sycl::aligned_alloc_host(4096, CTLSIZE, Q);
  // if you alloccate the space on the device instead, the program doesn't hang
  //uint64_t * ctl_mem = (uint64_t *) sycl::aligned_alloc_device(4096, CTLSIZE, Q);

  std::cout << " ctl_mem " << ctl_mem << std::endl;
 
  sycl::context ctx = Q.get_context();

  uint64_t *device_flag;
  device_flag = &ctl_mem[0];

  sycl::event e = Q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(1), [=](sycl::group<1> grp) {
	  sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(device_flag[0]);
	  // either of the following lines causes a hang
	  // uint64_t data = cpu_to_gpu.load();
	  cpu_to_gpu.store(0);
	});
      });
  std::cout<<"kernel launched" << std::endl;
  e.wait_and_throw();
  return 0;
}

