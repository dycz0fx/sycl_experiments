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
#define CMD_A2B_COUNT 1001
#define CMD_B2A_COUNT 1002
#define CMD_SIZE 1003
#define CMD_HELP 1004
#define CMD_VALIDATE 1005
#define CMD_CPU 1006
#define CMD_ATOMICSTORE 1019
#define CMD_ATOMICLOAD 1020
#define CMD_LATENCY 1021
#define CMD_A2B_BUF 1022
#define CMD_B2A_BUF 1023

/* option global variables */
// specify defaults
uint64_t glob_a2b_count = 0;
uint64_t glob_b2a_count = 0;
uint64_t glob_size = 0;
int glob_validate = 0;
int glob_cpu = 0;
int glob_interlock = 0;
int glob_use_atomic_load = 0;
int glob_use_atomic_store = 0;
int glob_latency = 0;
int glob_a2b_buf = 0;  /* 0 in host memory, 1 in gpu memory */
int glob_b2a_buf = 0;  /* 0 in host memory, 1 in gpu memory */

void Usage()
{
  std::cout <<
    "--a2bbuf a|b           set location of gpu-a to gpu-b buffer\n"
    "--b2abuf a|b           set location of gpu-b to gpu=a buffer\n"
    "--a2bcount <n>         set number of iterations\n"
    "--b2acount <n>         set number of iterations\n"
    "--size <n>             set size\n"
    "--help                 usage message\n"
    "--validate             set and check data\n"
    "--interlock            wait for responses\n"
    "--atomicload=0/1       method of flag access on device\n"
    "--atomicstore=0/1      method of flag access on device\n";
  std::cout << std::endl;
  exit(1);
}

void ProcessArgs(int argc, char **argv)
{
  const option long_opts[] = {
    {"a2bcount", required_argument, nullptr, CMD_A2B_COUNT},
    {"b2acount", required_argument, nullptr, CMD_B2A_COUNT},
    {"a2bbuf", required_argument, nullptr, CMD_A2B_BUF},
    {"b2abuf", required_argument, nullptr, CMD_B2A_BUF},
    {"size", required_argument, nullptr, CMD_SIZE},
    {"help", no_argument, nullptr, CMD_HELP},
    {"validate", no_argument, nullptr, CMD_VALIDATE},
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
	case CMD_A2B_COUNT: {
	  glob_a2b_count = std::stoi(optarg);
	  std::cout << "a2bcount " << glob_a2b_count << std::endl;
	  break;
	}
	case CMD_B2A_COUNT: {
	  glob_b2a_count = std::stoi(optarg);
	  std::cout << "b2acount " << glob_b2a_count << std::endl;
	  break;
	}
	case CMD_A2B_BUF: {
	  if (strcmp(optarg, "a") == 0)
	    glob_a2b_buf = 0;
	  else if (strcmp(optarg, "b") == 0)
	    glob_a2b_buf = 1;
	  else assert(0);
	  std::cout << "a2bbuf " << glob_a2b_buf << std::endl;
	  break;
	}
	case CMD_B2A_BUF: {
	  if (strcmp(optarg, "a") == 0)
	    glob_b2a_buf = 0;
	  else if (strcmp(optarg, "b") == 0)
	    glob_b2a_buf = 1;
	  else assert(0);
	  std::cout << "b2abuf " << glob_a2b_buf << std::endl;
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


constexpr size_t BUFSIZE = (1L << 20);  //allocate 1 MB usable

void GPUThread(struct RingState *s, size_t size, int latencyflag)
{  // gpu code
  uint64_t msgdata[7] = {0xfeedface, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < s->send_count; i += 1) {
    msgdata[1] = i;
    gpu_ring_send(s, MSG_PUT, size, msgdata);
    //gpu_ring_poll(s);
    if (latencyflag) {
      while (s->total_received < (i+1)) {
	gpu_ring_poll(s);
      }
    }
  }
  gpu_ring_drain(s);
}

void printstats(struct RingState *s)
{
  std::cout << s->name << " sent " << s->total_sent << " recv " << s->total_received << " nop " << s->total_nop << std::endl;
  std::cout << "send buf " << s->sendbuf << std::endl;
  std::cout << "recv buf " << s->recvbuf << std::endl;
  std::cout << "next send " << s->next_send << std::endl;
  std::cout << "next receive " << s->next_receive << std::endl;
  std::cout << "peer next receive " << s->peer_next_receive << std::endl;
  std::cout << "peer next receive sent " << s->peer_next_receive_sent << std::endl;
  std::cout << "send count " << s->send_count << std::endl;
  std::cout << "recv count " << s->recv_count << std::endl;
}

int main(int argc, char *argv[]) {
  ProcessArgs(argc, argv);
  uint64_t loc_a2b_count = glob_a2b_count;
  uint64_t loc_b2a_count = glob_b2a_count;
  int loc_a2b_buf = glob_a2b_buf;
  int loc_b2a_buf = glob_b2a_buf;
  uint64_t loc_size = glob_size;
  int loc_validate = glob_validate;
  int loc_use_atomic_load = glob_use_atomic_load;
  int loc_use_atomic_store = glob_use_atomic_store;
  int loc_latency = glob_latency;
  int loc_cpu = glob_cpu;

  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
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
	if (d.is_gpu()) {
	  qs.push_back(sycl::queue(d, prop_list));
	}
    }
  }

  sycl::queue qa = qs[0];
  sycl::queue qb = qs[1];

  std::cout<<"qa selected device : "<<qa.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"qa device vendor : "<<qa.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  std::cout<<"qb selected device : "<<qb.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"qb device vendor : "<<qb.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  if (qa.get_device() == qb.get_device()) {
    std::cout << "qa and qb are the same device\n" << std::endl;
  }

  
  // allocate device memory

  struct RingMessage *a2b_data_mem;
  struct RingMessage *b2a_data_mem;
  
  if (loc_a2b_buf == 1) {
    a2b_data_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, qb);
  } else {
    a2b_data_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, qa);
  }
  if (loc_b2a_buf == 1) {
    b2a_data_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, qb);
  } else {
    b2a_data_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, qa);
  }
  std::cout << " a2b_data_mem " << a2b_data_mem << std::endl;
  std::cout << " b2a_data_mem " << b2a_data_mem << std::endl;

  qa.memset(a2b_data_mem, 0, BUFSIZE);
  qa.wait_and_throw();
  qb.memset(b2a_data_mem, 0, BUFSIZE);
  qb.wait_and_throw();

  struct RingState *gpua;
  struct RingState *gpub;

  gpua = (struct RingState *) sycl::aligned_alloc_device(4096, sizeof(struct RingState), qa);
  gpub = (struct RingState *) sycl::aligned_alloc_device(4096, sizeof(struct RingState), qb);
  std::cout << " gpua " << gpua << std::endl;
  std::cout << " gpub " << gpub << std::endl;

  qa.single_task( [=]() {
      //nitstate(gpua, a2b_data_mem, b2a_data_mem, loc_a2b_count, loc_b2a_count, NULL);
      initstate(gpua, a2b_data_mem, b2a_data_mem, loc_a2b_count, 0, NULL);
    });
  qa.wait_and_throw();
  qb.single_task( [=]() {
      //initstate(gpub, b2a_data_mem, a2b_data_mem, loc_b2a_count, loc_a2b_count, NULL);
      initstate(gpub, b2a_data_mem, a2b_data_mem, loc_b2a_count, 0, NULL);
    });
  qb.wait_and_throw();
  std::cout << "printstate" << std::endl;

  struct RingState *myrsa;
  struct RingState *myrsb;

  myrsa = (struct RingState *) sycl::aligned_alloc_host(4096, sizeof(struct RingState), qa);
  myrsb = (struct RingState *) sycl::aligned_alloc_host(4096, sizeof(struct RingState), qb);

  qa.memcpy(myrsa, gpua, sizeof(struct RingState));
  qa.wait_and_throw();
  myrsa->name = "gpua";
  printstats(myrsa);
  qb.memcpy(myrsb, gpub, sizeof(struct RingState));
  qb.wait_and_throw();
  myrsb->name = "gpub";
  printstats(myrsb);

  std::cout << " a2b_data_mem " << a2b_data_mem << std::endl;
  std::cout << " b2a_data_mem " << b2a_data_mem << std::endl;

  std::cout << "kernel going to launch" << std::endl;

  unsigned long start_time, end_time;
  struct timespec ts_start, ts_end;
  
  clock_gettime(CLOCK_REALTIME, &ts_start);
  start_time = rdtsc();
  
  auto ea = qa.single_task([=]() {
	  GPUThread(gpua, loc_size, loc_latency);
	});
  auto eb = qb.single_task([=]() {
	  GPUThread(gpub, loc_size, loc_latency);
	});
  
  ea.wait_and_throw();
  eb.wait_and_throw();

  clock_gettime(CLOCK_REALTIME, &ts_end);
  end_time = rdtsc();
  printduration("gpua kernel ", ea);
  printduration("gpub kernel ", eb);
      
  /* common cleanup */
  double elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
    ((double) (ts_end.tv_nsec - ts_start.tv_nsec));
  
  double fcount;
  if (loc_a2b_count > loc_b2a_count)
    fcount = loc_a2b_count;
  else
    fcount = loc_b2a_count;
  double size = loc_size;
  double mbps = (size * 1000) / (elapsed / fcount);
  double nsec = elapsed / fcount;
  std::cout << "elapsed " << elapsed << " fcount " << fcount << std::endl;
  std::cout << argv[0];
  std::cout << " --b2a_count " << loc_b2a_count;
  std::cout << " --a2b_count " << loc_a2b_count;
  std::cout << " --b2a_buf " << loc_b2a_buf;
  std::cout << " --a2b_buf " << loc_a2b_buf;
  std::cout << " --size " << loc_size;
  if (loc_cpu) std::cout << " --use_cpu";
  if (loc_latency) std::cout << " --latency";
  
  std::cout << "  each " << nsec << " nsec " << mbps << "MB/s" << std::endl;

  qa.memcpy(myrsa, gpua, sizeof(struct RingState));
  qa.wait_and_throw();
  myrsa->name = "gpua";
  printstats(myrsa);
  qb.memcpy(myrsb, gpub, sizeof(struct RingState));
  qb.wait_and_throw();
  myrsb->name = "gpub";
  printstats(myrsb);
  return 0;
}

