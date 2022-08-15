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
#include <sys/stat.h>

constexpr size_t BUFSIZE = 65536;
constexpr size_t CTLSIZE = (4096);
constexpr uint64_t cpu_to_gpu_index = 0;
constexpr uint64_t gpu_to_cpu_index = 8;
#define cpu_relax() asm volatile("rep; nop")

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
    std::cout << "fd size " << statbuf.st_size << std::endl;

    void *base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(base != (void *) -1);

    return (T*)base;
}

int main(int argc, char *argv[]) {
  uint64_t count;
  uint64_t blocksize;
  if (argc < 3) exit(1);
  count = atol(argv[1]);
  blocksize = atol(argv[2]);
  if (blocksize > BUFSIZE) blocksize = BUFSIZE;
  sycl::queue Q;
  std::cout<<"count : "<< count << std::endl;
  std::cout<<"blocksize : "<< blocksize << std::endl;
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;

  //  uint64_t * device_data_mem = sycl::malloc_device<uint64_t>(BUFSIZE, Q);
  void * tmpptr =  sycl::malloc_device(sizeof(uint64_t) * BUFSIZE, Q);
  uint64_t *device_data_mem = (uint64_t *) tmpptr;
  std::cout << " device_data_mem " << device_data_mem << std::endl;
  std::array<uint64_t, BUFSIZE> host_data_array;
  memset(&host_data_array[0], 0, BUFSIZE);
  
  uint64_t * device_ctl_mem = sycl::malloc_device<uint64_t>(CTLSIZE, Q);
  std::cout << " device_ctl_mem " << device_ctl_mem << std::endl;
  std::array<uint64_t, CTLSIZE> host_ctl_array;
  memset(&host_ctl_array[0], 0, CTLSIZE);
  
  //create mmap mapping of usm device memory on host
  sycl::context ctx = Q.get_context();
  sleep(1);
  std::cout << "About to call mmap" << std::endl;
  
  uint64_t *host_data_map = get_mmap_address(device_data_mem, sizeof(uint64_t) * BUFSIZE, Q);
  std::cout << " host_data_map " << host_data_map << std::endl;

  uint64_t *host_ctl_map = get_mmap_address(device_ctl_mem, CTLSIZE, Q);
  std::cout << " host_ctl_map " << host_ctl_map << std::endl;
  
  auto e = Q.submit([&](sycl::handler &h) {
      h.memcpy(device_data_mem, &host_data_array[0], BUFSIZE);
    });
  e.wait_and_throw();
  e = Q.submit([&](sycl::handler &h) {
      h.memcpy(device_ctl_mem, &host_ctl_array[0], BUFSIZE);
    });
  e.wait_and_throw();
    
  std::cout << "kernel going to launch" << std::endl;

  e = Q.submit([&](sycl::handler &h) {
      //   sycl::stream os(1024, 128, h);
      //h.single_task([=]() {
      h.parallel_for(sycl::nd_range<1>{{1}, {1}}, [=](sycl::nd_item<1> idx) {
	  //  os<<"kernel start\n";
	  uint64_t prev = (uint64_t) -1L;
	  sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(device_ctl_mem[cpu_to_gpu_index]);
	  sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> gpu_to_cpu(device_ctl_mem[gpu_to_cpu_index]);
	  do {
	    uint64_t temp;
	    for (;;) {
	      temp = cpu_to_gpu.load();
	      if (temp != prev) break;
	    }
	    prev = temp;
	    if (blocksize > 0)
	      memset(device_data_mem, temp & 255, blocksize);
	    gpu_to_cpu.store(temp);
	  } while (prev < count-1 );
	  // os<<"kernel exit\n";
        });
    });
  std::cout<<"kernel launched" << std::endl;
    {
      struct timespec ts_start;
      clock_gettime(CLOCK_REALTIME, &ts_start);
      unsigned long start_time = rdtsc();
      volatile uint64_t *c_to_g = & host_ctl_map[cpu_to_gpu_index];
      volatile uint64_t *g_to_c = & host_ctl_map[gpu_to_cpu_index];

      for (int i = 0; i < count; i += 1) {
	*c_to_g = i;
	while (i != *g_to_c) cpu_relax();
	if (blocksize > 0) {
	  memcpy(&host_data_array[0], &host_data_map[0], blocksize);
	}
      }
      unsigned long end_time = rdtscp();
      struct timespec ts_end;
      clock_gettime(CLOCK_REALTIME, &ts_end);
      double elapsed = (ts_start.tv_sec - ts_start.tv_sec) * 1000000000 +
	(ts_end.tv_nsec - ts_start.tv_nsec);
      std::cout << "count " << count << " tsc each " << (end_time - start_time) / count << std::endl;
      std::cout << "count " << count << " tsc each " << elapsed / count << std::endl;
    }

      
    e.wait_and_throw();
    std::cout << "kernel over" << std::endl;
    munmap(host_data_map, BUFSIZE);
    munmap(host_ctl_map, CTLSIZE);
    sycl::free(device_data_mem, Q);
    sycl::free(device_ctl_mem, Q);
    return 0;
}

