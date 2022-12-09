#define DEBUG 0

#include <CL/sycl.hpp>
#include <stdlib.h>
#include <unistd.h>
#include <thread>
#include <getopt.h>
#include <iostream>
#include "level_zero/ze_api.h"
//#include <CL/sycl/backend/level_zero.hpp>
#include <sys/mman.h>
#include"../common_includes/rdtsc.h"
#include <time.h>
#include <sys/stat.h>
#include <immintrin.h>
#include <stdarg.h>
#include <pthread.h>
#include <assert.h>
#include <string.h>
#include <malloc.h>
#include "uncached.cpp"
#include "ringlib.cpp"

#define cpu_relax() asm volatile("rep; nop")

void dbprintf(int line, const char *format, ...)
{
  va_list arglist;
  printf("line %d: ", line);
  va_start(arglist, format);
  vprintf(format, arglist);
  va_end(arglist);
}

#define DP(...) if (DEBUG) dbprintf(__LINE__, __VA_ARGS__)

#define HERE()
#define CHERE(name) if (DEBUG) std::cout << name << " " <<  __FUNCTION__ << ": " << __LINE__ << std::endl; 

#define NSEC_IN_SEC 1000000000.0
/* how many messages per buffer */
/* option codes */
#define CMD_C2G_COUNT 1001
#define CMD_G2C_COUNT 1002
#define CMD_SIZE 1003
#define CMD_HELP 1004
#define CMD_VALIDATE 1005
#define CMD_CPU 1006
#define CMD_ATOMICSTORE 1019
#define CMD_ATOMICLOAD 1020
#define CMD_LATENCY 1021
#define CMD_C2G_BUF 1022
#define CMD_G2C_BUF 1023

/* option global variables */
// specify defaults
uint64_t glob_c2g_count = 0;
uint64_t glob_g2c_count = 0;
uint64_t glob_size = 0;
int glob_validate = 0;
int glob_cpu = 0;
int glob_interlock = 0;
int glob_use_atomic_load = 0;
int glob_use_atomic_store = 0;
int glob_latency = 0;
int glob_c2g_buf = 0;  /* 0 in host memory, 1 in gpu memory */
int glob_g2c_buf = 0;  /* 0 in host memory, 1 in gpu memory */

void Usage()
{
  std::cout <<
    "--cputogpubuf cpu|gpu          set location of cpu to gpu buffer\n"
    "--gputocpubuf cpu|gpu          set location of gpu to cpu buffer\n"
    "--cputogpucount <n>            set number of iterations\n"
    "--gputocpucount <n>            set number of iterations\n"
    "--use_cpu              use cpu for both ends\n"
    "--size <n>             set size\n"
    "--help                 usage message\n"
    "--validate             set and check data\n"
    "--interlock            wait for responses\n"
    "--atomicload=0/1 | --atomicstore=0/1         method of flag access on device\n";
  std::cout << std::endl;
  exit(1);
}

void ProcessArgs(int argc, char **argv)
{
  const option long_opts[] = {
    {"cputogpucount", required_argument, nullptr, CMD_C2G_COUNT},
    {"gputocpucount", required_argument, nullptr, CMD_G2C_COUNT},
    {"cputogpubuf", required_argument, nullptr, CMD_C2G_BUF},
    {"gputocpubuf", required_argument, nullptr, CMD_G2C_BUF},
    {"size", required_argument, nullptr, CMD_SIZE},
    {"help", no_argument, nullptr, CMD_HELP},
    {"validate", no_argument, nullptr, CMD_VALIDATE},
    {"use_cpu", no_argument, nullptr, CMD_CPU},
    {"atomicload", required_argument, nullptr, CMD_ATOMICLOAD},
    {"atomicstore", required_argument, nullptr, CMD_ATOMICSTORE},
    {"latency", no_argument, nullptr, CMD_LATENCY},
    {nullptr, no_argument, nullptr, 0}
  };
  while (true)
    {
      const auto opt = getopt_long(argc, argv, "", long_opts, nullptr);
      if (-1 == opt)
	break;
      switch (opt)
	{
	  case CMD_C2G_COUNT: {
	  glob_c2g_count = std::stoi(optarg);
	  std::cout << "cputogpucount " << glob_c2g_count << std::endl;
	  break;
	}
	  case CMD_G2C_COUNT: {
	  glob_g2c_count = std::stoi(optarg);
	  std::cout << "gputocpucount " << glob_g2c_count << std::endl;
	  break;
	}
	  case CMD_C2G_BUF: {
	    if (strcmp(optarg, "cpu") == 0)
	      glob_c2g_buf = 0;
	    else if (strcmp(optarg, "gpu") == 0)
	      glob_c2g_buf = 1;
	    else assert(0);
	  std::cout << "cputogpubuf " << glob_c2g_buf << std::endl;
	  break;
	}
	  case CMD_G2C_BUF: {
	    if (strcmp(optarg, "cpu") == 0)
	      glob_g2c_buf = 0;
	    else if (strcmp(optarg, "gpu") == 0)
	      glob_g2c_buf = 1;
	    else assert(0);
	  std::cout << "gputocpubuf " << glob_c2g_buf << std::endl;
	  break;
	}
	case CMD_SIZE: {
	  glob_size = std::stoi(optarg);
	  std::cout << "size " << glob_size << std::endl;
	  break;
	}
	case CMD_HELP: {
	  Usage();
	  break;
	}
	case CMD_VALIDATE: {
	  glob_validate = true;
	  break;
	}
	case CMD_CPU: {
	  glob_cpu = true;
	  break;
	}
	case CMD_ATOMICSTORE: {
	  glob_use_atomic_store = std::stoi(optarg);
	  std::cout << "atomicstore " << glob_use_atomic_store << std::endl;
	  break;
	}
	case CMD_ATOMICLOAD: {
	  glob_use_atomic_load = std::stoi(optarg);
	  std::cout << "atomicload " << glob_use_atomic_load << std::endl;
	  break;
	}
	case CMD_LATENCY: {
	  glob_latency = 1;
	  std::cout << "latency " << glob_latency << std::endl;
	  break;
	}
	default: {
	  Usage();
	  exit(1);
	}
    };
    }
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

constexpr size_t BUFSIZE = (1L << 20);  //allocate 1 MB usable


void *GPUThread(void *arg)
{  // cpu code
  struct RingState *s = (struct RingState *) arg;
  uint64_t msgdata[7] = {0xfeedface, 0, 0, 0, 0, 0, 0};
  std::cout << "thread starting " << s->name << " send " << s->send_count << " recv " << s->recv_count << std::endl;
  for (int i = 0; i < s->send_count; i += 1) {
    msgdata[1] = i;
    cpu_ring_send(s, MSG_PUT, glob_size, msgdata);
    //printf ("%s %d\n", s->name, i);
    if (glob_latency) {
      while (s->total_received < (i+1)) {
	if (DEBUG) printf("latency %s sent %d recd %d\n", s->name, i, s->total_received);
	cpu_relax();
	cpu_ring_poll(s);
      }
    }
  }
  cpu_ring_drain(s);
  return(NULL);
}


void printstats(struct RingState *s)
{
  std::cout << s->name << " sent " << s->total_sent << " recv " << s->total_received << " nop " << s->total_nop << std::endl;
}

int main(int argc, char *argv[]) {
  ProcessArgs(argc, argv);
  uint64_t loc_c2g_count = glob_c2g_count;
  uint64_t loc_g2c_count = glob_g2c_count;
  int loc_c2g_buf = glob_c2g_buf;
  int loc_g2c_buf = glob_g2c_buf;
  uint64_t loc_size = glob_size;
  int loc_validate = glob_validate;
  int loc_use_atomic_load = glob_use_atomic_load;
  int loc_use_atomic_store = glob_use_atomic_store;
  int loc_latency = glob_latency;
  int loc_cpu = glob_cpu;

  
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
  sycl::queue Q;
  if (loc_cpu) {
    Q = sycl::queue(sycl::cpu_selector_v, prop_list);
  } else {
    Q = sycl::queue(sycl::gpu_selector_v, prop_list);
  }
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;

  
  // allocate device memory

  struct RingMessage *c2g_data_mem;
  struct RingMessage *c2g_data_hostmap;   // pointer for host to use
  if (loc_c2g_buf == 1) {
    c2g_data_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, Q);
    if (loc_cpu) {
      c2g_data_hostmap = c2g_data_mem;
    } else {
      c2g_data_hostmap = get_mmap_address(c2g_data_mem, BUFSIZE, Q);
    }
  } else {
    c2g_data_mem = (struct RingMessage *) sycl::aligned_alloc_host(4096, BUFSIZE * 2, Q);
    c2g_data_hostmap = c2g_data_mem;
  }
  std::cout << " c2g_data_mem " << c2g_data_mem << std::endl;
  std::cout << " c2g_data_hostmap " << c2g_data_hostmap << std::endl;

  //allocate host mamory
  struct RingMessage *g2c_data_mem;
  struct RingMessage *g2c_data_hostmap;   // pointer for host to use
  if (loc_g2c_buf == 1) {
    g2c_data_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, Q);
    if (loc_cpu) {
      g2c_data_hostmap = g2c_data_mem;
    } else {
      g2c_data_hostmap = get_mmap_address(g2c_data_mem, BUFSIZE, Q);
    }
  } else {
    g2c_data_mem = (struct RingMessage *) sycl::aligned_alloc_host(4096, BUFSIZE, Q);
    g2c_data_hostmap = g2c_data_mem;
  }
  std::cout << " g2c_data_mem " << g2c_data_mem << std::endl;
  std::cout << " g2c_data_hostmap " << g2c_data_hostmap << std::endl;




  // initialize g2c mamory
  memset(g2c_data_hostmap, 0, BUFSIZE);

  // initialize c2g memory
  memset(c2g_data_hostmap, 0, BUFSIZE);
  
  struct RingState *cpu = (struct RingState *) sycl::aligned_alloc_host(4096, sizeof(struct RingState), Q);
  struct RingState *gpu;

  if (loc_cpu) {
    gpu = (struct RingState *) sycl::aligned_alloc_host(4096, sizeof(struct RingState), Q);
  } else {
    gpu = (struct RingState *) sycl::aligned_alloc_device(4096, sizeof(struct RingState), Q);
  }

  initstate(cpu, c2g_data_hostmap, g2c_data_hostmap, loc_c2g_count, loc_g2c_count, "cpu");
  auto e = Q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(1), [=](sycl::group<1> grp) {
	  initstate(gpu, g2c_data_mem, c2g_data_mem, loc_g2c_count, loc_c2g_count, "gpu");
	});
    });
  e.wait_and_throw();

  std::cout << "kernel going to launch" << std::endl;

  unsigned long start_time, end_time;
  struct timespec ts_start, ts_end;

  
  pthread_t gputhread;
  pthread_attr_t pt_attributes;
  pthread_attr_init(&pt_attributes);
  
  clock_gettime(CLOCK_REALTIME, &ts_start);
  start_time = rdtsc();
  
      if (! loc_cpu) {  // gpu
	e = Q.submit([&](sycl::handler &h) {
	    h.parallel_for_work_group(sycl::range(1), sycl::range(1), [=](sycl::group<1> grp) {
		uint64_t msgdata[7] = {0xdeadbeef, 0, 0, 0, 0, 0, 0};
		for (int i = 0; i < gpu->send_count; i += 1) {
		  msgdata[1] = i;
		  gpu_ring_send(gpu, MSG_PUT, loc_size, msgdata);
		  if (loc_latency) {
		    while (gpu->total_received < (i+1)) {
		      gpu_ring_poll(gpu);
		    }
		  }
		}
		gpu_ring_drain(gpu);
	      });
	  });
      } else {
  pthread_create(&gputhread, &pt_attributes, GPUThread, (void *) gpu);
      }

#if 0
      else { //  loc_cpu
	e = Q.submit([&](sycl::handler &h) {
	    h.parallel_for_work_group(sycl::range(1), sycl::range(1), [=](sycl::group<1> grp) {
		uint64_t msgdata[7] = {0xdeadbeef, 0, 0, 0, 0, 0, 0};
		for (int i = 0; i < gpu->send_count; i += 1) {
		  msgdata[1] = i;
		  cpu_ring_send(gpu, MSG_PUT, loc_size, msgdata);
		  if (loc_latency) {
		    while (gpu->total_received <= (i+1)) {
		      cpu_ring_poll(gpu);
		    }
		  }
		}
		cpu_ring_drain(gpu);
	      });
	  });
      }
#endif

      GPUThread(cpu);

      if (loc_cpu) {
	std::cout << "join gpu thread " << std::endl;
	pthread_join(gputhread, NULL);

      } else {
	e.wait_and_throw();
      }

      clock_gettime(CLOCK_REALTIME, &ts_end);
      end_time = rdtsc();
      printduration("gpu kernel ", e);
      
    /* common cleanup */
    double elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	((double) (ts_end.tv_nsec - ts_start.tv_nsec));

    double fcount;
    if (loc_c2g_count > loc_g2c_count)
      fcount = loc_c2g_count;
    else
      fcount = loc_g2c_count;
    double size = loc_size;
    double mbps = (size * 1000) / (elapsed / fcount);
    double nsec = elapsed / fcount;
    std::cout << "elapsed " << elapsed << " fcount " << fcount << std::endl;
    std::cout << argv[0];
    std::cout << " --g2c_count " << loc_g2c_count;
    std::cout << " --c2g_count " << loc_c2g_count;
    std::cout << " --g2c_buf " << loc_g2c_buf;
    std::cout << " --c2g_buf " << loc_c2g_buf;
    std::cout << " --size " << loc_size;
    if (loc_cpu) std::cout << " --use_cpu";
    if (loc_latency) std::cout << " --latency";

    std::cout << "  each " << nsec << " nsec " << mbps << "MB/s" << std::endl;
    printstats(cpu);
    Q.memcpy(cpu, gpu, sizeof(struct RingState));
    Q.wait_and_throw();
    cpu->name = "gpu";
    printstats(cpu);


    
    
    if (loc_cpu) {
    } else {
      munmap(c2g_data_hostmap, BUFSIZE);
      sycl::free(c2g_data_mem, Q);
      sycl::free(gpu, Q);
    }
    return 0;
}

