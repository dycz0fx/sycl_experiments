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
#include "ringlib2.cpp"
#include <new>   // in order to use placement new

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
#define CMD_LATENCY 1021
#define CMD_A2B_BUF 1022
#define CMD_B2A_BUF 1023
#define CMD_A 1024
#define CMD_B 1025
#define CMD_ATHREADS 1026
#define CMD_BTHREADS 1027

/* option global variables */
// specify defaults
uint64_t glob_a2b_count = 0;
uint64_t glob_b2a_count = 0;
uint64_t glob_size = 8;
int glob_validate = 0;
int glob_cpu = 0;
int glob_interlock = 0;
int glob_latency = 0;
int glob_athreads = 1;
int glob_bthreads = 1;
int glob_a2b_buf = 0;  /* 0: cpu 1: p0 2: p1 3: t0 4: t1 */
int glob_b2a_buf = 0;  /* 0: cpu 1: p0 2: p1 3: t0 4: t1 */
int glob_a = 0;   /* 0: cpu 1: p0 2: p1 3: t0 4: t1 */
int glob_b = 0;   /* 0: cpu 1: p0 2: p1 3: t0 4: t1 */

void Usage()
{
  std::cout <<
    "--a= p0|p1|t0|t1|c      set a end to pvc 0 or 1, tile 0 or 1, or cpu\n"
    "--b= p0|p1|t0|t1|c      set a end to pvc 0 or 1, tile 0 or 1, or cpu\n"
    "--a2bbuf p0|p1|t0|t1|c  set location of gpu-a to gpu-b buffer\n"
    "--b2abuf p0|p1|t0|t1|c  set location of gpu-b to gpu=a buffer\n"
    "--a2bcount <n>          set number of iterations\n"
    "--b2acount <n>          set number of iterations\n"
    "--size <n>              set size\n"
    "--athreads <n>          number threads for end a\n"
    "--bthreads <n>          number threads for end b\n"
    "--help                  usage message\n"
    "--validate              set and check data\n"
    "--interlock             wait for responses\n";
  std::cout << std::endl;
  exit(1);
}

#define LOC_CPU 0
#define LOC_PVC0 1
#define LOC_PVC1 2
#define LOC_TILE0 3
#define LOC_TILE1 4

int location_to_code(const char *s)
{
  int v = -1;
  if (strcmp(s, "c")==0) return LOC_CPU;
  if (strcmp(s, "p0")==0) return LOC_PVC0;
  if (strcmp(s, "p1")==0) return LOC_PVC1;
  if (strcmp(s, "t0")==0) return LOC_TILE0;
  if (strcmp(s, "t1")==0) return LOC_TILE1;
  printf("unknown option value %s\n", s);
  assert(0);
}

void ProcessArgs(int argc, char **argv)
{
  const option long_opts[] = {
    {"a2bcount", required_argument, nullptr, CMD_A2B_COUNT},
    {"b2acount", required_argument, nullptr, CMD_B2A_COUNT},
    {"a2bbuf", required_argument, nullptr, CMD_A2B_BUF},
    {"b2abuf", required_argument, nullptr, CMD_B2A_BUF},
    {"a", required_argument, nullptr, CMD_A},
    {"b", required_argument, nullptr, CMD_B},
    {"size", required_argument, nullptr, CMD_SIZE},
    {"athreads", required_argument, nullptr, CMD_ATHREADS},
    {"bthreads", required_argument, nullptr, CMD_BTHREADS},
    {"help", no_argument, nullptr, CMD_HELP},
    {"validate", no_argument, nullptr, CMD_VALIDATE},
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
	  std::cout << "--a2bcount=" << glob_a2b_count << std::endl;
	  break;
	}
	case CMD_B2A_COUNT: {
	  glob_b2a_count = std::stoi(optarg);
	  std::cout << "--b2acount=" << glob_b2a_count << std::endl;
	  break;
	}
	case CMD_A: {
	  glob_a = location_to_code(optarg);
	  std::cout << "--a=" << glob_a << std::endl;
	  break;
	}
	case CMD_B: {
	  glob_b = location_to_code(optarg);
	  std::cout << "--b=" << glob_b << std::endl;
	  break;
	}
	case CMD_A2B_BUF: {
	  glob_a2b_buf = location_to_code(optarg);
	  std::cout << "--a2bbuf=" << glob_a2b_buf << std::endl;
	  break;
	}
	case CMD_B2A_BUF: {
	  glob_b2a_buf = location_to_code(optarg);
	  std::cout << "--b2abuf=" << glob_b2a_buf << std::endl;
	  break;
	}
	case CMD_SIZE: {
	  glob_size = std::stoi(optarg);
	  std::cout << "--size=" << glob_size << std::endl;
	  break;
	}
	case CMD_ATHREADS: {
	  glob_athreads = std::stoi(optarg);
	  std::cout << "--athreads=" << glob_athreads << std::endl;
	  break;
	}
	case CMD_BTHREADS: {
	  glob_bthreads = std::stoi(optarg);
	  std::cout << "--bthreads=" << glob_bthreads << std::endl;
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
	case CMD_LATENCY: {
	  glob_latency = 1;
	  std::cout << "--latency=" << glob_latency << std::endl;
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

struct CPURingCommand {
  CPURing *cpur;
  int send_count;
  int recv_count;
  int latency_flag;
};
  
constexpr size_t BUFSIZE = (1L << 20);  //allocate 1 MB usable

void CPUThread(CPURingCommand *r)
{  // cpu code
  CPURing *s = r->cpur;
  int send_count = r->send_count;
  int recv_count = r->recv_count;
  int latencyflag = r->latency_flag;
  struct RingMessage msg;
  msg.header = MSG_PUT;
  msg.data[0] = 0xdeadbeef;
  for (int i = 0; i < send_count; i += 1) {
    msg.data[1] = i;
    s->Send(&msg);
    if (latencyflag) {
      while ((s->next_receive-RingN) < i) {
	s->Poll();
      }
    } else if (recv_count > 0) {
      s->Drain();  // rate balancing
    }
  }
  while ((s->next_receive-RingN) < recv_count) s->Drain();
}

void *ThreadFn(void * arg)
{
  struct CPURingCommand *r = (CPURingCommand *) arg;
  CPUThread(r);
  return (NULL);
}

struct JoinStruct {
  sycl::event e;
  pthread_t thread;
  int is_pthread;
  int * completion;
};

void *Joiner(void *arg)
{ 
  struct JoinStruct *p = (struct JoinStruct *) arg;
  if (p->is_pthread) {
    pthread_join(p->thread, NULL);
  } else {
    p->e.wait();
  }
  *p->completion = 1;
  return(NULL);
}

void GPUThread(GPURing *s, int send_count, int recv_count, int latency_flag)
{
  struct RingMessage msg;
  msg.header = MSG_PUT;
  msg.data[0] = 0xfeedface;
  for (int i = 0; i < send_count; i += 1) {
    msg.data[1] = i;
    s->Send(&msg);
    if (recv_count > 0) s->Drain();
  }
  while ((s->next_receive-RingN) < recv_count) s->Drain();
}

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

int main(int argc, char *argv[]) {
  ProcessArgs(argc, argv);
  uint64_t loc_a2b_count = glob_a2b_count;
  uint64_t loc_b2a_count = glob_b2a_count;
  int loc_a2b_buf = glob_a2b_buf;
  int loc_b2a_buf = glob_b2a_buf;
  uint64_t loc_size = glob_size;
  int loc_athreads = glob_athreads;
  int loc_bthreads = glob_bthreads;
  int loc_validate = glob_validate;
  int loc_latency = glob_latency;
  int loc_a = glob_a;
  int loc_b = glob_b;

  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
  std::vector<sycl::queue> qs;


  // xxxxxxxxxxxx
  std::vector<sycl::queue> pvcq;
  std::vector<sycl::queue> tileq;

  int qcount = 0;	   
  
  auto platforms = sycl::platform::get_platforms();
  for (auto & p : platforms) {
    auto pname = p.get_info<sycl::info::platform::name>();
    //std::cout << "*Platform: " << pname << std::endl;
    if ( pname.find("Level-Zero") == std::string::npos) {
      //std::cout << "non Level Zero GPU skipped" << std::endl;
        continue;
    }
    auto devices = p.get_devices();
    //std::cout << "number of devices: " << devices.size() << std::endl;
    for (auto & d : devices ) {
      //std::cout << "**Device: " << d.get_info<sycl::info::device::name>() << std::endl;
      if (d.is_gpu()) {
	if (qcount < 2) {
	  pvcq.push_back(sycl::queue(d, prop_list));
	  //std::cout << "create pvcq[" << pvcq.size() - 1 << "]" << std::endl;
	  //std::cout << "**max wg: " << d.get_info<cl::sycl::info::device::max_work_group_size>()  << std::endl;
	  qcount += 1;
	} else {
	  std::vector<sycl::device> sd = d.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::next_partitionable);
	  //std::cout << " subdevices " << sd.size() << std::endl;
	  for (auto &subd: sd) {
	    tileq.push_back(sycl::queue(subd, prop_list));
	    //std::cout << "create tileq[" << tileq.size() - 1 << "]" << std::endl;
	    //std::cout << "**max wg: " << subd.get_info<cl::sycl::info::device::max_work_group_size>()  << std::endl;
	    qcount += 1;
	  }
	}
	if (qcount >= 4) break;
      }
      if (qcount >= 4) break;
    }
  }
  sycl::queue hostq = sycl::queue(sycl::gpu_selector_v, prop_list);
  //  std::cout << "create host queue" << std::endl;

  // xxxxxxxxxxxxx


  sycl::queue qa;
  sycl::queue qb;
  //  assert(loc_a != loc_b);
  if (loc_a == LOC_CPU) qa = hostq;
  else if (loc_a == LOC_PVC0) qa = pvcq[0];
  else if (loc_a == LOC_PVC1) qa = pvcq[1];
  else if (loc_a == LOC_TILE0) qa = tileq[0];
  else if (loc_a == LOC_TILE1) qa = tileq[1];
  else assert(0);
  if (loc_b == LOC_CPU) qb = hostq;
  else if (loc_b == LOC_PVC0) qb = pvcq[0];
  else if (loc_b == LOC_PVC1) qb = pvcq[1];
  else if (loc_b == LOC_TILE0) qb = tileq[0];
  else if (loc_b == LOC_TILE1) qb = tileq[1];
  else assert(0);

  
  //std::cout<<"qa selected device : "<<qa.get_device().get_info<sycl::info::device::name>() << std::endl;
  //std::cout<<"qa device vendor : "<<qa.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  //std::cout<<"qb selected device : "<<qb.get_device().get_info<sycl::info::device::name>() << std::endl;
  //std::cout<<"qb device vendor : "<<qb.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  //if (qa.get_device() == qb.get_device()) {
  //std::cout << "qa and qb are the same device\n" << std::endl;
  //}


  struct RingMessage *cpu_a2b_mem = (struct RingMessage *) sycl::aligned_alloc_host(4096, BUFSIZE * 2, hostq);
  struct RingMessage *pvc0_a2b_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, pvcq[0]);
  struct RingMessage *pvc1_a2b_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, pvcq[1]);
  struct RingMessage *tile0_a2b_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, tileq[0]);
  struct RingMessage *tile1_a2b_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, tileq[1]);
  struct RingMessage *cpu_b2a_mem = (struct RingMessage *) sycl::aligned_alloc_host(4096, BUFSIZE * 2, hostq);
  struct RingMessage *pvc0_b2a_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, pvcq[0]);
  struct RingMessage *pvc1_b2a_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, pvcq[1]);
  struct RingMessage *tile0_b2a_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, tileq[0]);
  struct RingMessage *tile1_b2a_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, tileq[1]);

  struct RingMessage *pvc0_a2b_hostmap = get_mmap_address(pvc0_a2b_mem, BUFSIZE, pvcq[0]);
  struct RingMessage *pvc1_a2b_hostmap = get_mmap_address(pvc1_a2b_mem, BUFSIZE, pvcq[1]);
  struct RingMessage *tile0_a2b_hostmap = get_mmap_address(tile0_a2b_mem, BUFSIZE, tileq[0]);
  struct RingMessage *tile1_a2b_hostmap = get_mmap_address(tile1_a2b_mem, BUFSIZE, tileq[1]);
  struct RingMessage *pvc0_b2a_hostmap = get_mmap_address(pvc0_b2a_mem, BUFSIZE, pvcq[0]);
  struct RingMessage *pvc1_b2a_hostmap = get_mmap_address(pvc1_b2a_mem, BUFSIZE, pvcq[1]);
  struct RingMessage *tile0_b2a_hostmap = get_mmap_address(tile0_b2a_mem, BUFSIZE, tileq[0]);
  struct RingMessage *tile1_b2a_hostmap = get_mmap_address(tile1_b2a_mem, BUFSIZE, tileq[1]);
  
  // allocate device memory

  struct RingMessage *gpua_tx_mem;
  struct RingMessage *gpua_rx_mem;
  struct RingMessage *gpub_tx_mem;
  struct RingMessage *gpub_rx_mem;
  struct RingMessage *gpua_tx_mem_hostmap;
  struct RingMessage *gpua_rx_mem_hostmap;
  struct RingMessage *gpub_tx_mem_hostmap;
  struct RingMessage *gpub_rx_mem_hostmap;

  // well the below is a mess
 
  if (loc_a == LOC_CPU) {
    if (loc_a2b_buf == LOC_CPU) gpua_tx_mem = cpu_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC0) gpua_tx_mem = pvc0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_PVC1) gpua_tx_mem = pvc1_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE0) gpua_tx_mem = tile0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE1) gpua_tx_mem = tile1_a2b_hostmap;
    else assert(0);
    gpua_tx_mem_hostmap = gpua_tx_mem;
  } else {
    if (loc_a2b_buf == LOC_CPU) gpua_tx_mem = cpu_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC0) gpua_tx_mem = pvc0_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC1) gpua_tx_mem = pvc1_a2b_mem;
    else if (loc_a2b_buf == LOC_TILE0) gpua_tx_mem = tile0_a2b_mem;
    else if (loc_a2b_buf == LOC_TILE1) gpua_tx_mem = tile1_a2b_mem;
    else assert(0);
    if (loc_a2b_buf == LOC_CPU) gpua_tx_mem_hostmap = cpu_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC0) gpua_tx_mem_hostmap = pvc0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_PVC1) gpua_tx_mem_hostmap = pvc1_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE0) gpua_tx_mem_hostmap = tile0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE1) gpua_tx_mem_hostmap = tile1_a2b_hostmap;
    else assert(0);
  }
  if (loc_a == LOC_CPU) {
    if (loc_b2a_buf == LOC_CPU) gpua_rx_mem = cpu_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC0) gpua_rx_mem = pvc0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_PVC1) gpua_rx_mem = pvc1_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE0) gpua_rx_mem = tile0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE1) gpua_rx_mem = tile1_b2a_hostmap;
    else assert(0);
    gpua_rx_mem_hostmap = gpua_rx_mem;
  } else {
    if (loc_b2a_buf == LOC_CPU) gpua_rx_mem = cpu_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC0) gpua_rx_mem = pvc0_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC1) gpua_rx_mem = pvc1_b2a_mem;
    else if (loc_b2a_buf == LOC_TILE0) gpua_rx_mem = tile0_b2a_mem;
    else if (loc_b2a_buf == LOC_TILE1) gpua_rx_mem = tile1_b2a_mem;
    else assert(0);
    if (loc_b2a_buf == LOC_CPU) gpua_rx_mem_hostmap = cpu_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC0) gpua_rx_mem_hostmap = pvc0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_PVC1) gpua_rx_mem_hostmap = pvc1_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE0) gpua_rx_mem_hostmap = tile0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE1) gpua_rx_mem_hostmap = tile1_b2a_hostmap;
    else assert(0);
  }
  if (loc_b == LOC_CPU) {
    if (loc_b2a_buf == LOC_CPU) gpub_tx_mem = cpu_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC0) gpub_tx_mem = pvc0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_PVC1) gpub_tx_mem = pvc1_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE0) gpub_tx_mem = tile0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE1) gpub_tx_mem = tile1_b2a_hostmap;
    else assert(0);
    gpub_tx_mem_hostmap = gpub_tx_mem;
  } else {
    if (loc_b2a_buf == LOC_CPU) gpub_tx_mem = cpu_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC0) gpub_tx_mem = pvc0_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC1) gpub_tx_mem = pvc1_b2a_mem;
    else if (loc_b2a_buf == LOC_TILE0) gpub_tx_mem = tile0_b2a_mem;
    else if (loc_b2a_buf == LOC_TILE1) gpub_tx_mem = tile1_b2a_mem;
    else assert(0);
    if (loc_b2a_buf == LOC_CPU) gpub_tx_mem_hostmap = cpu_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC0) gpub_tx_mem_hostmap = pvc0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_PVC1) gpub_tx_mem_hostmap = pvc1_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE0) gpub_tx_mem_hostmap = tile0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE1) gpub_tx_mem_hostmap = tile1_b2a_hostmap;
    else assert(0);
  }
  if (loc_b == LOC_CPU) {
    if (loc_a2b_buf == LOC_CPU) gpub_rx_mem = cpu_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC0) gpub_rx_mem = pvc0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_PVC1) gpub_rx_mem = pvc1_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE0) gpub_rx_mem = tile0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE1) gpub_rx_mem = tile1_a2b_hostmap;
    else assert(0);
    gpub_rx_mem_hostmap = gpub_rx_mem;
  } else {
    if (loc_a2b_buf == LOC_CPU) gpub_rx_mem = cpu_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC0) gpub_rx_mem = pvc0_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC1) gpub_rx_mem = pvc1_a2b_mem;
    else if (loc_a2b_buf == LOC_TILE0) gpub_rx_mem = tile0_a2b_mem;
    else if (loc_a2b_buf == LOC_TILE1) gpub_rx_mem = tile1_a2b_mem;
    else assert(0);
    if (loc_a2b_buf == LOC_CPU) gpub_rx_mem_hostmap = cpu_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC0) gpub_rx_mem_hostmap = pvc0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_PVC1) gpub_rx_mem_hostmap = pvc1_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE0) gpub_rx_mem_hostmap = tile0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE1) gpub_rx_mem_hostmap = tile1_a2b_hostmap;
    else assert(0);
  }
  std::cout << "gpua_tx_mem" << gpua_tx_mem << std::endl;
  std::cout << "gpub_rx_mem" << gpub_rx_mem << std::endl;
  std::cout << "gpub_tx_mem" << gpub_tx_mem << std::endl;
  std::cout << "gpua_rx_mem" << gpua_rx_mem << std::endl;
		       
  CPURing *cpua;
  CPURing *cpub;
  void * ringspacea;
  void * ringspaceb;
  GPURing *ringspacea_host_map;
  GPURing *ringspaceb_host_map;


  CPURing *cpu_host_a;
  CPURing *cpu_host_b;

  if (loc_a == LOC_CPU) {
    cpua = new (sycl::aligned_alloc_host<CPURing>(4096, 1, qa)) CPURing(); 
    cpu_host_a = cpua;
    std::cout << " cpua " << &cpua << std::endl;
  } else {
    ringspacea = sycl::aligned_alloc_device(4096, 4096, qa);
    ringspacea_host_map = (GPURing *) get_mmap_address(ringspacea, 4096, qa);
  }
  if (loc_b == LOC_CPU) {
    cpub = new (sycl::aligned_alloc_host<CPURing>(4096, 1, qb)) CPURing();
    cpu_host_b = cpub;
    std::cout << " cpub " << &cpub << std::endl;
  } else {
    ringspaceb = sycl::aligned_alloc_device(4096, 4096, qb);
    ringspaceb_host_map = (GPURing *) get_mmap_address(ringspaceb, 4096, qb);
  }


 

  std::cout << "kernel going to launch" << std::endl;

  unsigned long start_time, end_time;
  struct timespec ts_start, ts_end;
  
  clock_gettime(CLOCK_REALTIME, &ts_start);
  start_time = rdtsc();

  sycl::event ea;
  sycl::event eb;

  if (loc_a == LOC_CPU) {
    memset(gpua_tx_mem, 101, 4096);
  } else {
    qa.memset(gpua_tx_mem, 101, 4096);
    qa.wait_and_throw();
  }
  if (loc_b == LOC_CPU) {
    memset(gpub_tx_mem, 102, 4096);
  } else {
    qb.memset(gpub_tx_mem, 102, 4096);
    qb.wait_and_throw();
  }
  // check
  for (int i = 0; i < 20; i += 1) {
    uint8_t v = ((uint8_t *) gpua_rx_mem_hostmap)[i];
    if (v != 102) printf("gpua_rx_mem[%d] == %d expected %d \n", i, v, 102);
  }
  for (int i = 0; i < 20; i += 1) {
    uint8_t v = ((uint8_t *) gpub_rx_mem_hostmap)[i];
    if (v != 101) printf("gpub_rx_mem[%d] == %d expected %d \n", i, v, 101);
  }
    
  
  if (loc_a == LOC_CPU) {
    memset(gpua_tx_mem, 0, BUFSIZE);
    for (int i = 0; i < RingN; i += 1) {
      assert(gpua_tx_mem[i].header == 0);
    }
  } else  {
    auto e = qa.single_task( [=]() {
	memset(gpua_tx_mem, 0, BUFSIZE);
      });
    e.wait_and_throw();
    for (int i = 0; i < RingN; i += 1) {
      assert(gpua_tx_mem_hostmap[i].header == 0);
      assert(gpub_rx_mem_hostmap[i].header == 0);
    }
  }
  if (loc_b == LOC_CPU) {
    memset(gpub_tx_mem, 0, BUFSIZE);
    for (int i = 0; i < RingN; i += 1) {
      assert(gpub_tx_mem[i].header == 0);
    }
  } else {
    auto e = qb.single_task( [=]() {
	memset(gpub_tx_mem, 0, BUFSIZE);
      });
    e.wait_and_throw();
    for (int i = 0; i < RingN; i += 1) {
      assert(gpub_tx_mem_hostmap[i].header == 0);
      assert(gpua_rx_mem_hostmap[i].header == 0);
    }
  }
  
  if (loc_a == LOC_CPU) {
    cpua->InitState(gpua_tx_mem, gpua_rx_mem);
  } else {
  }
  if (loc_b == LOC_CPU) {
    cpub->InitState(gpub_tx_mem, gpub_rx_mem);
  } else {
  }
  if (loc_a == LOC_CPU) {
    for (int i = 0; i < RingN; i += 1) {
      assert(cpua->sendbuf[i].header == 0);
      assert(cpua->recvbuf[i].header == 0);
    }
  }
  if (loc_b == LOC_CPU) {
    for (int i = 0; i < RingN; i += 1) {
      assert(cpub->sendbuf[i].header == 0);
      assert(cpub->recvbuf[i].header == 0);
    }
  }
  
  
  pthread_t athread;
  pthread_t bthread;
  pthread_attr_t pt_attributes;
  pthread_attr_init(&pt_attributes);
  struct JoinStruct ajoin, bjoin;
  int acompletion = 0;
  int bcompletion = 0;
  ajoin.completion = &acompletion;
  bjoin.completion = &bcompletion;
  struct CPURingCommand acmd;
  acmd.cpur = cpu_host_a;
  acmd.send_count = loc_a2b_count;
  acmd.recv_count = loc_b2a_count;
  acmd.latency_flag = loc_latency;
  struct CPURingCommand bcmd;
  bcmd.cpur = cpu_host_b;
  bcmd.send_count = loc_b2a_count;
  bcmd.recv_count = loc_a2b_count;
  bcmd.latency_flag = loc_latency;
  
  if (loc_a == LOC_CPU) {
    pthread_create(&athread, &pt_attributes, ThreadFn, (void *) &acmd);
    ajoin.is_pthread = 1;
    ajoin.thread = athread;
  } else {
    int send_count = loc_a2b_count;
    int recv_count = loc_b2a_count;
    std::cout << "GPUA send_count " << send_count << " recv_count " << recv_count << std::endl;
    if (loc_athreads == 1) {
      ea = qa.single_task( [=]() {
	  GPURing *gpua = new(ringspacea) GPURing(gpua_tx_mem, gpua_rx_mem);
	  GPUThread(gpua, send_count, recv_count, loc_latency);
	});
    } else {
      assert(0);
      ea = qa.submit([&](sycl::handler &h) {
	  auto out = sycl::stream(1024, 768, h);
	  h.parallel_for_work_group(sycl::range(1), sycl::range(loc_athreads), [=](sycl::group<1> grp) {
	      GPURing gpua(gpua_tx_mem, gpua_rx_mem);
	      grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		  struct RingMessage msg;
		  msg.header = MSG_PUT;
		  msg.data[0] = 0xfeedface;
		  msg.data[1] = it.get_global_id()[0];
		  gpua.Send(&msg);
		  if (recv_count > 0) gpua.Drain();
		});
	      out << "GPUA next_send " << gpua.next_send << " next_receive " << gpua.next_receive << "\n";
	      
	      while (gpua.next_receive < recv_count) gpua.Drain();
	    });
	});
    }
    ajoin.is_pthread = 0;
    ajoin.e = ea;
  }
  if (loc_b == LOC_CPU) {
    pthread_create(&bthread, &pt_attributes, ThreadFn, (void *) &bcmd);
    bjoin.is_pthread = 1;
    bjoin.thread = bthread;
  } else {
    int send_count = loc_b2a_count;
    int recv_count = loc_a2b_count;
    std::cout << "GPUB send_count " << send_count << " recv_count " << recv_count << std::endl;
    if (loc_bthreads == 1) {
      eb = qb.single_task( [=]() {
	  GPURing *gpub = new(ringspaceb) GPURing(gpub_tx_mem, gpub_rx_mem);
	  GPUThread(gpub, send_count, recv_count, loc_latency);
	});
    } else {
      assert(0);
      eb = qb.submit([&](sycl::handler &h) {
	  auto out = sycl::stream(1024, 768, h);
	  h.parallel_for_work_group(sycl::range(1), sycl::range(loc_bthreads), [=](sycl::group<1> grp) {
	      GPURing gpub(gpub_tx_mem, gpub_rx_mem);
	      grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		  struct RingMessage msg;
		  msg.header = MSG_PUT;
		  msg.data[0] = 0xfeedface;
		  msg.data[1] = it.get_global_id()[0];
		  gpub.Send(&msg);
		  if (recv_count > 0) gpub.Drain();
		});
	      out << "GPUB next_send " << gpub.next_send << " next_receive " << gpub.next_receive << "\n";
	      while (gpub.next_receive < recv_count) gpub.Drain();
	    });
	});
    }
    bjoin.is_pthread = 0;
    bjoin.e = eb;
  }
  pthread_t ajoiner, bjoiner;
  pthread_create(&ajoiner, &pt_attributes, Joiner, &ajoin);
  pthread_create(&bjoiner, &pt_attributes, Joiner, &bjoin);

  //std::cout << "starting timeout check" << std::endl;
  for (;;) {
    clock_gettime(CLOCK_REALTIME, &ts_end);
    end_time = rdtsc();
    if (acompletion && bcompletion) break;
    if (ts_end.tv_sec - ts_start.tv_sec > 8) {
      printf("TIMEOUT acomplete %d bcomplete %d\n", acompletion, bcompletion);
      fflush(stdout);
      break;
    }
  }
  //std::cout << "got completion flags" << std::endl;
  if (acompletion) pthread_join(ajoiner, NULL);
  if (bcompletion) pthread_join(bjoiner, NULL);
  std::cout << "joined complete joiners" << std::endl;
  if ((loc_a != LOC_CPU) && acompletion) printduration("gpua kernel ", ea);
  if ((loc_b != LOC_CPU) && bcompletion) printduration("gpub kernel ", eb);
  std::cout << "printduration" << std::endl;

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
  std::cout << " --b " << loc_b;
  std::cout << " --a " << loc_a;
  std::cout << " --size " << loc_size;
  if (loc_latency) std::cout << " --latency";
  std::cout << "  each " << nsec << " nsec " << mbps << "MB/s" << std::endl;
  if (loc_a2b_count <= 2000) {
    for (int i = 0; i < loc_a2b_count; i += 1) {
      int32_t v = gpua_rx_mem_hostmap[i%RingN].sequence;
      if (v != (i + RingN) ) {
	printf("gpua_rx_mem[%d] == %d expected %d seq %i\n", i, v, i, gpub_rx_mem_hostmap[i%RingN].sequence);
      }
    }
  }
  if (loc_a2b_count <= 2000) {
    for (int i = 0; i < loc_a2b_count; i += 1) {
      int32_t v = gpub_rx_mem_hostmap[i%RingN].sequence;
      if (v != (i + RingN)) {
	printf("gpub_rx_mem[%d] == %d expected %d seq %d\n", i, v, i, gpub_rx_mem_hostmap[i%RingN].sequence);
      }
    }
  }
  fflush(stdout);
  printf("gpua next_peer_receive %d\n", gpua_rx_mem_hostmap[RingN].sequence);
  printf("gpub next_peer_receive %d\n", gpub_rx_mem_hostmap[RingN].sequence);
  if (loc_a == LOC_CPU) {
    cpua->Print("cpua");
  } else {
    ringspacea_host_map->Print("gpua");
  }
  if (loc_b == LOC_CPU) {
    cpub->Print("cpub");
  } else {
    ringspaceb_host_map->Print("gpub");
  }
  std::cout << "main returning" << std::endl;
  return 0;
}

