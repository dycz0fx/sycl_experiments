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
#include <immintrin.h>

constexpr size_t BUFSIZE = (1L << 20);
constexpr size_t CTLSIZE = (4096);
constexpr uint64_t cpu_to_gpu_index = 0;
constexpr uint64_t gpu_to_cpu_index = 8;
#define cpu_relax() asm volatile("rep; nop")

constexpr int use_check = 0;
constexpr int gpu_memcpy = 0;
constexpr int gpu_loop = 0;
 int use_memcpy = 0;
 int use_loop = 0;
 int use_avx = 0;


void fillbuf(uint64_t *p, uint64_t size, int iter)
{
  for (size_t j = 0; j < size >> 3; j += 1) {
    p[j] = (((long) iter) << 32) + j;
  }
}

long unsigned checkbuf(uint64_t *p, uint64_t size, int iter)
{
  long unsigned errors = 0;
  for (size_t j = 0; j < size >> 3; j += 1) {
    if (p[j] != (((long) iter) << 32) + j) errors += 1;
  }
  return (errors);
}


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

void usage()
{
  std::cout << "./host_mmap cputogpu|gputocpu memcpy|loop|avx count rsize wsize" << std::endl;
  exit(1);
}

int main(int argc, char *argv[]) {
  int cputogpu;
  uint64_t count;
  uint64_t rblocksize, wblocksize;
  if (argc < 6) usage();
  if (strcmp("cputogpu", argv[1]) == 0)
    cputogpu = 1;
  else
    cputogpu = 0;
  if (strcmp("memcpy", argv[2]) == 0) use_memcpy = 1;
  if (strcmp("loop", argv[2]) == 0) use_loop = 1;
  if (strcmp("avx", argv[2]) == 0) use_avx = 1;

  
  count = atol(argv[3]);
  rblocksize = atol(argv[4]);
  wblocksize = atol(argv[5]);
  if (rblocksize > BUFSIZE/2) rblocksize = BUFSIZE/2;
  if (wblocksize > BUFSIZE/2) wblocksize = BUFSIZE/2;
  sycl::queue Q;
  std::cout<<"count : "<< count << std::endl;
  std::cout<<"rblocksize : "<< rblocksize << std::endl;
  std::cout<<"wblocksize : "<< wblocksize << std::endl;
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;

  void * tmpptr =  sycl::aligned_alloc_device(4096, BUFSIZE, Q);
  uint64_t *device_data_mem = (uint64_t *) tmpptr;
  std::cout << " device_data_mem " << device_data_mem << std::endl;

  tmpptr =  sycl::aligned_alloc_device(4096, BUFSIZE, Q);
  uint64_t *extra_device_mem = (uint64_t *) tmpptr;
  std::cout << " extra_device_mem " << extra_device_mem << std::endl;

  std::array<uint64_t, BUFSIZE> host_data_array;
  memset(&host_data_array[0], 0, BUFSIZE);
  
  uint64_t * device_ctl_mem = (uint64_t *) sycl::aligned_alloc_device(4096, CTLSIZE, Q);
  std::cout << " device_ctl_mem " << device_ctl_mem << std::endl;
  std::array<uint64_t, CTLSIZE> host_ctl_array;
  memset(&host_ctl_array[0], 0, CTLSIZE);
  
  //create mmap mapping of usm device memory on host
  sycl::context ctx = Q.get_context();
  sleep(1);
  std::cout << "About to call mmap" << std::endl;
  
  uint64_t *host_data_map = get_mmap_address(device_data_mem, BUFSIZE / 2, Q);
  std::cout << " host_data_map " << host_data_map << std::endl;

  uint64_t *host_ctl_map = get_mmap_address(device_ctl_mem, CTLSIZE, Q);
  std::cout << " host_ctl_map " << host_ctl_map << std::endl;
  
  auto e = Q.submit([&](sycl::handler &h) {
      h.memcpy(device_data_mem, &host_data_array[0], BUFSIZE/2);
    });
  e.wait_and_throw();
  e = Q.submit([&](sycl::handler &h) {
      h.memcpy(device_ctl_mem, &host_ctl_array[0], CTLSIZE);
    });
  e.wait_and_throw();
    
  std::cout << "kernel going to launch" << std::endl;
  unsigned long start_time, end_time;
  struct timespec ts_start, ts_end;
  if (cputogpu) {
    e = Q.submit([&](sycl::handler &h) {
	//   sycl::stream os(1024, 128, h);
	//h.single_task([=]() {
	h.parallel_for(sycl::nd_range<1>{{1}, {1}}, [=](sycl::nd_item<1> idx) {
	    //  os<<"kernel start\n";
	    uint64_t prev = (uint64_t) -1L;
	    sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(device_ctl_mem[cpu_to_gpu_index]);
	    sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> gpu_to_cpu(device_ctl_mem[gpu_to_cpu_index]);
	    do {
	      uint64_t err = 0;
	      uint64_t temp;
	      for (;;) {
		temp = cpu_to_gpu.load();
		if (temp != prev) break;
	      }
	      prev = temp;
	      if (rblocksize > 0) {
		if (gpu_memcpy) {
		  memcpy(extra_device_mem, device_data_mem, rblocksize);
		}
	        if (gpu_loop) {
		  for (int j = 0; j < rblocksize >> 3; j += 1) {
		    extra_device_mem[j] = device_data_mem[j];
		  }
		}
	      }
	      if (use_check) {
		err = checkbuf(device_data_mem, wblocksize, temp);
	      }
	      gpu_to_cpu.store((err << 32) + temp);
	    } while (prev < count-1 );
	    // os<<"kernel exit\n";
	  }
	  );
      });
    std::cout<<"kernel launched" << std::endl;
    {
      clock_gettime(CLOCK_REALTIME, &ts_start);
      start_time = rdtsc();
      volatile uint64_t *c_to_g = & host_ctl_map[cpu_to_gpu_index];
      volatile uint64_t *g_to_c = & host_ctl_map[gpu_to_cpu_index];
      
      for (int i = 0; i < count; i += 1) {
	if (wblocksize > 0) {
	  if (use_check) {
	    fillbuf(&host_data_array[0], wblocksize, i);
	    if (checkbuf(&host_data_array[0], wblocksize, i) != 0)
	      std::cout << "fillbuf failed" << std::endl;
	  }
	  if (use_memcpy) {
	    memcpy(&host_data_map[0], &host_data_array[0], wblocksize);
	  }
	  if (use_loop) {
	    for (int j = 0; j < wblocksize >> 3; j += 1) {
	      host_data_map[j] = host_data_array[j];
	    }
	  }
	  if (use_avx) {
	    for (int j = 0; j < wblocksize>>3; j += 8) {
	      __m512i temp = _mm512_load_epi32((void *) &host_data_array[j]);
	      _mm512_store_si512((void *) &host_data_map[j], temp);
	    }
	  }
	}
	*c_to_g = i;
	uint64_t temp;
	for (;;) {
	  temp = *g_to_c;
	  if ((temp & 0xffffffff) == i) break;
	  cpu_relax();
	} 
	if (use_check && ((temp >> 32) != 0)) {
	  std::cout << "iteration " << i << " errors " << (temp >> 32) << std::endl;
	}
      }
      end_time = rdtscp();
      clock_gettime(CLOCK_REALTIME, &ts_end);
    }
    } else {  // gputocpu
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
	      if (wblocksize > 0)
		if (gpu_memcpy) {
		  memcpy(device_data_mem, extra_device_mem, wblocksize);
		}
	        if (gpu_loop) {
		  for (int j = 0; j < rblocksize >> 3; j += 1) {
		    device_data_mem[j] = 0;
		  }
		}
		if (use_check) {
		  fillbuf(device_data_mem, rblocksize, temp);
		}
	      gpu_to_cpu.store(temp);
	    } while (prev < count-1 );
	    // os<<"kernel exit\n";
	  });
      });
    std::cout<<"kernel launched" << std::endl;
    {
      clock_gettime(CLOCK_REALTIME, &ts_start);
      start_time = rdtsc();
      volatile uint64_t *c_to_g = & host_ctl_map[cpu_to_gpu_index];
      volatile uint64_t *g_to_c = & host_ctl_map[gpu_to_cpu_index];
      
      for (int i = 0; i < count; i += 1) {
	*c_to_g = i;
	while (i != *g_to_c) cpu_relax();
	if (rblocksize > 0) {
	  if (use_memcpy) {
	    memcpy(&host_data_array[0], &host_data_map[0], rblocksize);
	  }
	  if (use_loop) {
	    for (int j = 0; j < rblocksize >> 3; j += 1) {
	      host_data_array[j] = host_data_map[j];
	    }
	  }
	  if (use_avx) {
	    for (int j = 0; j < rblocksize>>3; j += 8) {
	      __m512i temp = _mm512_load_epi32((void *) &host_data_map[j]);
	      _mm512_store_si512((void *) &host_data_array[j], temp);
	    }
	  }
	  if (use_check) {
	    int err = checkbuf(&host_data_array[0], rblocksize, i);
	    if (err != 0) std::cout << "iteration " << i << " errors " << err << std::endl;
	  }
	}
      }
      end_time = rdtscp();
      clock_gettime(CLOCK_REALTIME, &ts_end);
    }
  }
    e.wait_and_throw();
    /* common cleanup */
    double elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	((double) (ts_end.tv_nsec - ts_start.tv_nsec));
    //      std::cout << "count " << count << " tsc each " << (end_time - start_time) / count << std::endl;
    std::cout << argv[1];
    if (use_memcpy) std::cout << " memcpy ";
    if (use_loop) std::cout << " loop ";
    if (use_avx) std::cout << " avx ";
    double mbs = (rblocksize > wblocksize) ? rblocksize : wblocksize;
    mbs = (mbs * 1000) / (elapsed / ((double) count));
    std::cout << " count " << count << " w " << wblocksize << " r " << rblocksize << " nsec each " << elapsed / ((double) count) << " MB/s " << mbs << std::endl;
      
    std::cout << "kernel over" << std::endl;
    munmap(host_data_map, BUFSIZE);
    munmap(host_ctl_map, CTLSIZE);
    sycl::free(device_data_mem, Q);
    sycl::free(extra_device_mem, Q);
    sycl::free(device_ctl_mem, Q);
    return 0;
}

