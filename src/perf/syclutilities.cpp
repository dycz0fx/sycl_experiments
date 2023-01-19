#ifndef SYCLUTILITIES_CPP
#define SYCLUTILITIES_CPP

#include <CL/sycl.hpp>

// for mmap
#include <sys/mman.h>
#include "level_zero/ze_api.h"
#include <sys/stat.h>

#include <assert.h>
#include <iostream>
#include <stdio.h>    // strerror
#include <string.h>   // memcpy

#include "syclutilities.h"

template<typename T>
T *get_mmap_address(T * device_ptr, size_t size, sycl::queue Q) {
    sycl::context ctx = Q.get_context();
    ze_ipc_mem_handle_t ze_ipc_handle;
    ze_result_t ret = zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx), device_ptr, &ze_ipc_handle);
    //std::cout<<"zeMemGetIpcHandle return : " << ret << std::endl;
    assert(ret == ZE_RESULT_SUCCESS);
    int fd;
    memcpy(&fd, &ze_ipc_handle, sizeof(fd));
    //std::cout << " fd " << fd << std::endl;
    //struct stat statbuf;
    //fstat(fd, &statbuf);
    //std::cout << "requested size " << size << std::endl;
    //std::cout << "fd size " << statbuf.st_size << std::endl;
    void *base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == (void *) -1) {
      std::cout << "mmap returned -1" << std::endl;
      std::cout << strerror(errno) << std::endl;  
    }
    assert(base != (void *) -1);
    return (T*)base;
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

ulong kernelcycles(sycl::event e)
{
  uint64_t start =
    e.get_profiling_info<sycl::info::event_profiling::command_start>();
  uint64_t end =
    e.get_profiling_info<sycl::info::event_profiling::command_end>();
  return (end - start);
}

ulong max_frequency(sycl::queue q)
{
  sycl::device dev = q.get_device();
  ulong device_frequency = (uint) dev.get_info<sycl::info::device::max_clock_frequency>();
  return (device_frequency);
}
#endif // ifundef SYCLUTILITIES_CPP
