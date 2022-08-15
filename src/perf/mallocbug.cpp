#include<CL/sycl.hpp>
#include<stdlib.h>
#include<unistd.h>
#include<thread>
#include<iostream>
#include"level_zero/ze_api.h"
#include<CL/sycl/backend/level_zero.hpp>
#include<sys/mman.h>
#include"../common_includes/rdtsc.h"
#include <sys/stat.h>

constexpr size_t BUFSIZE = 65536;
extern int errno;

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
    if (base == (void *) -1) {
      std::cout << "mmap returned -1" << std::endl;
      std::cout << strerror(errno) << std::endl;  
    }
    assert(base != (void *) -1);
    return (T*)base;
}

int main(int argc, char *argv[]) {
  sycl::queue Q;
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;

  //  uint64_t * device_data_mem = sycl::malloc_device<uint64_t>(BUFSIZE, Q);
  void * tmpptr =  sycl::malloc_device(sizeof(uint64_t) * BUFSIZE, Q);
  uint64_t *device_data_mem = (uint64_t *) tmpptr;

  std::cout << " device_data_mem " << device_data_mem << std::endl;
  
  //create mmap mapping of usm device memory on host
  sycl::context ctx = Q.get_context();
  
  std::cout << "About to call mmap" << std::endl;
  
  uint64_t *host_data_map = get_mmap_address(device_data_mem, sizeof(uint64_t) * BUFSIZE, Q);
  std::cout << " host_data_map " << host_data_map << std::endl;

    std::cout << "kernel over" << std::endl;
    munmap(host_data_map, BUFSIZE);
    sycl::free(device_data_mem, Q);
    return 0;
}

