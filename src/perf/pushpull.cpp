#include <CL/sycl.hpp>
#include <stdlib.h>
#include <unistd.h>
#include <thread>
#include <getopt.h>
#include <iostream>
#include "level_zero/ze_api.h"
#include <CL/sycl/backend/level_zero.hpp>
#include <sys/mman.h>
#include"../common_includes/rdtsc.h"
#include <time.h>
#include <sys/stat.h>
#include <immintrin.h>

constexpr size_t BUFSIZE = (1L << 20);
constexpr size_t CTLSIZE = (4096);
constexpr uint64_t cpu_to_gpu_index = 0;
constexpr uint64_t gpu_to_cpu_index = 8;
#define cpu_relax() asm volatile("rep; nop")

/* option codes */
#define CMD_COUNT 1001
#define CMD_READ 1002
#define CMD_WRITE 1003
#define CMD_HELP 1004
#define CMD_VALIDATE 1005
#define CMD_CPUTOGPU 1006
#define CMD_GPUTOCPU 1007
#define CMD_DEVICEDATA 1008
#define CMD_HOSTDATA 1009
#define CMD_DEVICECTL 1010
#define CMD_HOSTCTL 1011
#define CMD_SPLITCTL 1012
#define CMD_HOSTLOOP 1013
#define CMD_HOSTMEMCPY 1014
#define CMD_HOSTAVX 1015
#define CMD_HOSTLOOP 1016
#define CMD_HOSTMEMCPY 1017
/* option global variables */
uint64_t glob_count = 0;
uint64_t glob_readsize = 0;
uint64_t glob_writesize = 0;
int glob_validate = 0;
int glob_cputogpu = 0;
int glob_gputocpu = 0;
int glob_devicedata = 0;
int glob_hostdata = 0;
int glob_devicectl = 0;
int glob_hostctl = 0;
int glob_splitctl = 0;
int glob_hostloop = 0;
int glob_hostmemcpy = 0;
int glob_hostavx = 0;
int glob_deviceloop = 0;
int glob_devicememcpy = 0;


void ProcessArgs(int argc, char **argv)
{
  const char* short_opts = "c:r:w:vhDHALM";
  const option long_opts[] = {
    {"count", required_argument, nullptr, CMD_COUNT},
    {"read", required_argument, nullptr, CMD_READ},
    {"write", required_argument, nullptr, CMD_WRITE},
    {"help", no_argument, nullptr, CMD_HELP},
    {"validate", no_argument, nullptr, CMD_VALIDATE},
    {"cputogpu", no_argument, nullptr, CMD_CPUTOGPU},
    {"gputocpu", no_argument, nullptr, CMD_GPUTOCPU},
    {"devicedata", no_argument, nullptr, CMD_DEVICEDATA},
    {"hostdata", no_argument, nullptr, CMD_HOSTDATA},
    {"devicectl", no_argument, nullptr, CMD_DEVICECTL},
    {"hostctl", no_argument, nullptr, CMD_HOSTCTL},
    {"splitctl", no_argument, nullptr, CMD_SPLITCTL},

    {"hostloop", no_argument, nullptr, CMD_HOSTLOOP},
    {"hostmemcpy", no_argument, nullptr, CMD_HOSTMEMCPY},
    {"hostavx", no_argument, nullptr, CMD_HOSTAVX},
    {"deviceloop", no_argument, nullptr, CMD_DEVICELOOP},
    {"devicememcpy", no_argument, nullptr, CMD_DEVICEMEMCPY},
    {nullptr, no_argument, nullptr, 0}
  };
  while (true)
    {
      const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullpr);
      if (-1 == opt)
	break;
      switch (opt)
	{
	case CMD_COUNT: {
	  glob_count = std::stoi(optarg);
	  std::cout << "count " << glob_count << std::endl;
	  break;
	}
	case CMD_READ: {
	  glob_readsize = std::stoi(optarg);
	  std::cout << "read " << glob_readsize << std::endl;
	  break;
	}
	case CMD_WRITE: {
	  glob_writesize = std::stoi(optarg);
	  std::cout << "write " << glob_writesize << std::endl;
	  break;
	}
	case CMD_HELP: {
	  Usage();
	  break;
	}
	case CMD_HELP: {
	  glob_validate = true;
	  break;
	}
	case CMD_CPUTOGPU: {
	  glob_cputogpu = true;
	  break;
	}
	case CMD_GPUTOCPU: {
	  glob_gputocpu = true;
	  break;
	}
	case CMD_DEVICEDATA: {
	  glob_devicedata = true;
	  break;
	}
	case CMD_HOSTDATA: {
	  glob_hostdata = true;
	  break;
	}
	case CMD_DEVICECTL: {
	  glob_devicectl = true;
	  break;
	}
	case CMD_HOSTCTL: {
	  glob_hostctl = true;
	  break;
	}
	case CMD_SPLITCTL: {
	  glob_splitctl = true;
	  break;
	}

	case CMD_HOSTLOOP: {
	  glob_hostloop = true;
	  break;
	}
	case CMD_HOSTMEMCPY: {
	  glob_hostmemcpy = true;
	  break;
	}
	case CMD_HOSTAVX: {
	  glob_hostavx = true;
	  break;
	}
	case CMD_DEVICELOOP: {
	  glob_deviceloop = true;
	  break;
	}
	case CMD_DEVICEMEMCPY: {
	  glob_devicememcpy = true;
	  break;
	}

    };



}

void Usage()
{
  std::cout <<
    "--count <n>            set number of iterations\n"
    "--read <n>             set read size\n"
    "--write <n>            set write size\n"
    "--help                 usage message\n"
    "--validate             set and check data\n"
    "--cputogpu | --gputocpu   direction of data transfer\n"
    "--devicedata | --hostdata location of data buffer\n"
    "--devicectl | --hostctl | --splitctl    location of control flags\n"
    "--hostloop | --hostmemcpy | --hostavx   code for host\n"
    "--deviceloop | --devicememcpy           code for device\n";
  std::cout << std::endl;
  exit(1);
}


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


int main(int argc, char *argv[]) {
  ProcessArgs(argc, argv);
  uint64_t loc_count = glob_count;
  uint64_t loc_readsize = glob_readsize;
  uint64_t loc_writesize = glob_writesize;
  int loc_validate = glob_validate;
  int loc_cputogpu = glob_cputogpu;
  int loc_gputocpu = glob_gputocpu;
  int loc_devicedata = glob_devicedata;
  int loc_hostdata = glob_hostdata;
  int loc_devicectl = glob_devicectl;
  int loc_hostctl = glob_hostctl;
  int loc_splitctl = glob_splitctl;
  int loc_hostloop = glob_hostloop;
  int loc_hostmemcpy = glob_hostmemcpy;
  int loc_hostavx = glob_hostavx;
  int loc_deviceloop = glob_deviceloop;
  int loc_devicememcpy = glob_devicememcpy;

  if (loc_readsize > BUFSIZE/2) loc_readsize = BUFSIZE/2;
  if (loc_writesize > BUFSIZE/2) loc_writesize = BUFSIZE/2;
  sycl::queue Q;
  std::cout<<"count : "<< count << std::endl;
  std::cout<<"readsize : "<< loc_readsize << std::endl;
  std::cout<<"writesize : "<< loc_writesize << std::endl;
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;

  void * tmpptr =  sycl::aligned_alloc_device(4096, BUFSIZE, Q);
  uint64_t *device_data_mem = (uint64_t *) tmpptr;
  std::cout << " device_data_mem " << device_data_mem << std::endl;

  tmpptr =  sycl::aligned_alloc_host(4096, BUFSIZE, Q);
  uint64_t *host_data_mem = (uint64_t *) tmpptr;
  std::cout << " host_data_mem " << host_data_mem << std::endl;

  tmpptr =  sycl::aligned_alloc_device(4096, BUFSIZE, Q);
  uint64_t *extra_device_mem = (uint64_t *) tmpptr;
  std::cout << " extra_device_mem " << extra_device_mem << std::endl;

  std::array<uint64_t, BUFSIZE> host_data_array;
  memset(&host_data_array[0], 0, BUFSIZE);
  
  uint64_t * device_ctl_mem = (uint64_t *) sycl::aligned_alloc_device(4096, CTLSIZE, Q);
  std::cout << " device_ctl_mem " << device_ctl_mem << std::endl;
  std::array<uint64_t, CTLSIZE> host_ctl_array;

  uint64_t * host_ctl_mem = (uint64_t *) sycl::aligned_alloc_host(4096, CTLSIZE, Q);
  std::cout << " host_ctl_mem " << host_ctl_mem << std::endl;
  std::array<uint64_t, CTLSIZE> host_ctl_array;


  
  memset(&host_ctl_array[0], 0, CTLSIZE);


  //create mmap mapping of usm device memory on host
  sycl::context ctx = Q.get_context();
  sleep(1);
  uint64_t *host_data_map;   // pointer for host to use
  uint64_t *data_mem;   // pointer for device to use

    std::cout << "About to call mmap" << std::endl;

  if (use_devicemem) {
    data_mem = device_data_mem;
    host_data_map = get_mmap_address(device_data_mem, BUFSIZE / 2, Q);
  } else {
    data_mem = host_data_mem;
    host_data_map = data_mem;
  }

  uint64_t *ctl_mem;
  uint64_t *host_ctl_map;
  if (use_host_ctl) {
    host_ctl_map = host_ctl_mem;
    ctl_mem = host_ctl_mem;
  } else {
    host_ctl_map = get_mmap_address(device_ctl_mem, CTLSIZE, Q);
    ctl_mem = device_ctl_mem;
  }
  
  std::cout << " host_data_map " << host_data_map << std::endl;
  
  
  std::cout << " host_ctl_map " << host_ctl_map << std::endl;
  
  auto e = Q.submit([&](sycl::handler &h) {
      h.memcpy(data_mem, &host_data_array[0], BUFSIZE/2);
    });
  e.wait_and_throw();
  e = Q.submit([&](sycl::handler &h) {
      h.memcpy(ctl_mem, &host_ctl_array[0], CTLSIZE);
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
	    sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(ctl_mem[cpu_to_gpu_index]);
	    sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> gpu_to_cpu(ctl_mem[gpu_to_cpu_index]);
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
		  memcpy(extra_device_mem, data_mem, rblocksize);
		}
	        if (gpu_loop) {
		  for (int j = 0; j < rblocksize >> 3; j += 1) {
		    extra_device_mem[j] = data_mem[j];
		  }
		}
	      }
	      if (use_check) {
		err = checkbuf(extra_device_mem, rblocksize, temp);
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
	    sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> cpu_to_gpu(ctl_mem[cpu_to_gpu_index]);
	    sycl::atomic_ref<uint64_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> gpu_to_cpu(ctl_mem[gpu_to_cpu_index]);
	    do {
	      uint64_t temp;
	      for (;;) {
		temp = cpu_to_gpu.load();
		if (temp != prev) break;
	      }
	      prev = temp;
	      if (use_check) {
		fillbuf(extra_device_mem, wblocksize, temp);
	      }
	      if (wblocksize > 0)
		if (gpu_memcpy) {
		  memcpy(data_mem, extra_device_mem, wblocksize);
		}
	        if (gpu_loop) {
		  for (int j = 0; j < rblocksize >> 3; j += 1) {
		    data_mem[j] = extra_device_mem[j];
		  }
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
    if (use_devicemem) std::cout << "devicemem ";
    if (use_hostmem) std::cout << "hostmem ";
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

