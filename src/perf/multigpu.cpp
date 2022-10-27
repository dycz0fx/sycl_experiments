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

constexpr size_t BUFSIZE = (65536);
constexpr double NSEC_IN_SEC = 1000000000.0;


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

// get a pointer to src_ptr on device src_q which is usable on device use_q
template<typename T>
T *get_ipc_address(T *src_ptr, size_t size, sycl::queue src_q, sycl::queue use_q) {
  // get the ipc handle from the source context
    sycl::context ctx = src_q.get_context();
    ze_ipc_mem_handle_t ze_ipc_handle;
    ze_result_t ret = zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx), src_ptr, &ze_ipc_handle);
    // std::cout<<"zeMemGetIpcHandle return : " << ret << std::endl;
    assert(ret == ZE_RESULT_SUCCESS);
    // get the ze_context_handle_t from the use_q
    ctx = use_q.get_context();
    ze_context_handle_t ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    // get the sycl::device in use_q
    sycl::device sdev = use_q.get_device();
    // get the ze device corresponding to the sdev
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sdev);
    void *ptr;
    ret = zeMemOpenIpcHandle(ze_ctx, ze_dev, ze_ipc_handle, 0, &ptr);
    assert(ret == ZE_RESULT_SUCCESS);
    return((T *) ptr);
}

/* command line mode arguments:
 *
 * storeflush iter setflagloc pollflagloc tocpumode fromcpumode
 *   tocpuloc 0 = device, 1 = host
 *   togpuloc 0 = device, 1 = host
 *   mode  0 = pointer, 1 = atomic, 2 = uncached
 */



void printduration(const char* name, sycl::event e)
  {
    uint64_t start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
    std::cout << name << " execution time: " << duration << " sec" << std::endl;
  }

char checkbuf[BUFSIZE];

void memcheck(const char *msg, int src, int dest, int val, size_t size) {
  unsigned errors = 0;
  for (int i = 0; i < size; i += 1) {
    if (checkbuf[i] != val) errors += 1;
    if (errors == 1) printf ("expected %d got %d\n", val, checkbuf[i]);
  }
  if (errors != 0) printf("%s %d to %d val %d errors %d\n", msg,  src, dest, val, errors);
}

void setandcopyback(sycl::queue q, uint64_t *p, int val, size_t size) {
  auto e = q.submit([&](sycl::handler &h) {
      h.memset(p, val, BUFSIZE);
    });
  e.wait_and_throw();
  
  e = q.submit([&](sycl::handler &h) {
      h.memcpy(&checkbuf[0], p, BUFSIZE);
    });
  e.wait_and_throw();
}

// ############################

int main(int argc, char *argv[]) {
  struct timespec ts_start, ts_end;
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

  // modelled after Jeff Hammond code in PRK repo
  std::vector<sycl::queue> qs;
  
  auto platforms = sycl::platform::get_platforms();
  for (auto & p : platforms) {
    auto pname = p.get_info<sycl::info::platform::name>();
    std::cout << "*Platform: " << pname << std::endl;
    if ( pname.find("Level-Zero") == std::string::npos) {
        std::cout << "non Level Zero GPU skipped" << std::endl;
        continue;
    }
    auto devices = p.get_devices();
    for (auto & d : devices ) {
        std::cout << "**Device: " << d.get_info<sycl::info::device::name>() << std::endl;
        if ( d.is_gpu() ) {
            std::cout << "**Device is GPU - adding to vector of queues" << std::endl;
            qs.push_back(sycl::queue(d, prop_list));
        }
    }
  }

  int haz_ngpu = qs.size();
  std::cout << "Number of GPUs found  = " << haz_ngpu << std::endl;

  int ngpu = 6;   // make a command line argument
  
  std::vector<uint64_t *> host_mem(ngpu);
  std::vector<uint64_t *> device_src(ngpu);
  std::vector<uint64_t *> host_view_device_src(ngpu);
  std::vector<uint64_t *> device_dest(ngpu);
  std::vector<uint64_t *> host_view_device_dest(ngpu);
  std::vector<uint64_t *> device_src_map(ngpu);
  std::vector<uint64_t *> device_dest_map(ngpu);
  // vector of a vector of pointers
  std::vector<uint64_t **> host_copy_device_src_map(ngpu);
  std::vector<uint64_t **> host_copy_device_dest_map(ngpu);

  for (int i = 0; i < ngpu; i += 1) {
    host_mem[i] = (uint64_t *) sycl::aligned_alloc_host(4096, BUFSIZE, qs[i]);
    device_src[i] = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, qs[i]);
    host_view_device_src[i]= get_mmap_address(device_src[i], BUFSIZE, qs[i]);
    device_src_map[i] = (uint64_t *) sycl::aligned_alloc_device(4096, ngpu * sizeof(uint64_t *), qs[i]);
    device_dest[i] = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, qs[i]);
    host_view_device_dest[i]= get_mmap_address(device_dest[i], BUFSIZE, qs[i]);

    device_dest_map[i] = (uint64_t *) sycl::aligned_alloc_device(4096, ngpu * sizeof(uint64_t *), qs[i]);
    host_copy_device_src_map[i] = (uint64_t **) sycl::aligned_alloc_host(4096, ngpu * sizeof(uint64_t *), qs[i]);
    
    host_copy_device_dest_map[i] = (uint64_t **) sycl::aligned_alloc_host(4096, ngpu * sizeof(uint64_t *), qs[i]);

    std::cout << " host_mem[" << i << "] " << host_mem[i] << std::endl;
    std::cout << " device_src[" << i << "] " << device_src[i] << std::endl;
    std::cout << " host_view_device_src[" << i << "] " << host_view_device_src[i] << std::endl;
    std::cout << " device_dest[" << i << "] " << device_dest[i] << std::endl;
    std::cout << " host_view_device_dest[" << i << "] " << host_view_device_dest[i] << std::endl;
  }

  // initialize IPC addresses for each GPU
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      if (i == j) {
	host_copy_device_src_map[j][i] = device_src[i];
	host_copy_device_dest_map[j][i] = device_dest[i];
      } else {
	host_copy_device_src_map[j][i] = get_ipc_address(device_src[i], BUFSIZE, qs[i],qs[j]);
	host_copy_device_dest_map[j][i] = get_ipc_address(device_dest[i], BUFSIZE, qs[i],qs[j]);
      }
    }
  }
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      std::cout << "ipc src view from " << i << " of " << j << ": " << host_copy_device_src_map[i][j] << std::endl; 
    }
    for (int j = 0; j < ngpu; j += 1) {
      std::cout << "ipc dest view from " << i << " of " << j << ": " << host_copy_device_dest_map[i][j] << std::endl; 
    }
  }
    



  
  // initialize memories
  for (int i = 0; i < ngpu; i += 1) {
      memset(host_mem[i], 28, BUFSIZE);
      auto e = qs[i].submit([&](sycl::handler &h) {
	  h.memset(device_src[i], i, BUFSIZE);
	});
      e.wait_and_throw();
      #if 0
      e = qs[i].submit([&](sycl::handler &h) {
	  h.memcpy(&device_src_map[i][0], &host_copy_device_src_map[i][0], ngpu * sizeof(uint64_t *));
	});
      e.wait_and_throw();
      e = qs[i].submit([&](sycl::handler &h) {
	  h.memcpy(&device_dest_map[i][0], &host_copy_device_dest_map[i][0], ngpu * sizeof(uint64_t *));
	});
      e.wait_and_throw();
      #endif
  }
  
  // initialize device_src
  std::cout << " memory initialized " << std::endl;

  // test printf
  for (int i = 0; i < ngpu; i += 1) {
    qs[i].submit([&](sycl::handler &h) {
	auto out = sycl::stream(1024, 768, h);
	auto task = 
	  [=]() {
	  out << "In task " << i << "\n";
	  };
	h.single_task(task);
      });
  }

    std::cout<<"store with ipc maps" << std::endl;

  // test cross-device stores
  // from i to j
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      setandcopyback(qs[i], &host_copy_device_src_map[i][i][0], i, BUFSIZE);
      memcheck("store source", i, j, i, BUFSIZE);
      setandcopyback(qs[j], &host_copy_device_dest_map[j][j][0], i, BUFSIZE);
      memcheck("store dest", i, j, i, BUFSIZE);
      auto e = qs[i].submit([&](sycl::handler &h) {
	  h.memcpy(&host_copy_device_dest_map[i][j][0], &host_copy_device_src_map[i][i][0], BUFSIZE);
	});
      e.wait_and_throw();
      e = qs[j].submit([&](sycl::handler &h) {
	  h.memcpy(checkbuf, &host_copy_device_dest_map[j][j][0], BUFSIZE);
	});
      e.wait_and_throw();
      memcheck("store", i, j, i, BUFSIZE);
    }
  }

  // test cross-device loads
  std::cout<<"load with ipc maps" << std::endl;
  
  // from i to j
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      setandcopyback(qs[i], &host_copy_device_src_map[i][i][0], i, BUFSIZE);
      memcheck("load src", i, -1, i, BUFSIZE);
      setandcopyback(qs[j], &host_copy_device_dest_map[j][j][0], 100 + i, BUFSIZE);
      memcheck("load dest", i, -1, 100 + i, BUFSIZE);
      auto e = qs[j].submit([&](sycl::handler &h) {
	  h.memcpy(&host_copy_device_dest_map[j][j][0], &host_copy_device_src_map[j][i][0], BUFSIZE);
	});
      e.wait_and_throw();
      e = qs[j].submit([&](sycl::handler &h) {
	  h.memcpy(checkbuf, &host_copy_device_dest_map[j][j][0], BUFSIZE);
	});
      e.wait_and_throw();
      memcheck("load ", i, j, i, BUFSIZE);
    }
  }


  // try without the maps
#if TRYNOMAPS
  std::cout<<"store without ipc maps" << std::endl;

    // test cross-device stores
  // from i to j
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      setandcopyback(qs[i], &device_src[i][0], i, BUFSIZE);
      memcheck("store source", i, j, i, BUFSIZE);
      setandcopyback(qs[j], &device_dest[j][0], j + 100, BUFSIZE);
      memcheck("store dest", i, j, j + 100, BUFSIZE);
      auto e = qs[i].submit([&](sycl::handler &h) {
	  h.memcpy(&device_dest[j][0], &device_src[i][0], BUFSIZE);
	});
      e.wait_and_throw();
      e = qs[j].submit([&](sycl::handler &h) {
	  h.memcpy(checkbuf, &device_dest[j][0], BUFSIZE);
	});
      e.wait_and_throw();
      memcheck("store", i, j, i, BUFSIZE);
    }
  }

  std::cout<<"load without ipc maps" << std::endl;
  
  // from i to j
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      setandcopyback(qs[i], &device_src[i][0], i, BUFSIZE);
      memcheck("load src", i, -1, i, BUFSIZE);
      setandcopyback(qs[j], &device_dest[j][0], 100 + j, BUFSIZE);
      memcheck("load dest", i, -1, 100 + j, BUFSIZE);
      auto e = qs[j].submit([&](sycl::handler &h) {
	  h.memcpy(&device_dest[j][0], &device_src[i][0], BUFSIZE);
	});
      e.wait_and_throw();
      e = qs[j].submit([&](sycl::handler &h) {
	  h.memcpy(checkbuf, &device_dest[j][0], BUFSIZE);
	});
      e.wait_and_throw();
      memcheck("load ", i, j, i, BUFSIZE);
    }
  }
#endif

  
  std::cout<<"kernel returned" << std::endl;
  return 0;

}

