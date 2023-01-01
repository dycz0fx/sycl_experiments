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

/* option global variables */
// specify defaults
uint64_t glob_a2b_count = 0;
uint64_t glob_b2a_count = 0;
uint64_t glob_size = 0;
int glob_validate = 0;
int glob_cpu = 0;
int glob_interlock = 0;
int glob_latency = 0;
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


constexpr size_t BUFSIZE = (1L << 20);  //allocate 1 MB usable

void CPUThread(CPURing *s, size_t size, int latencyflag)
{  // cpu code
  struct RingMessage msg;
  msg.header = MSG_PUT;
  msg.data[0] = 0xfeedface;
  for (int i = 0; i < s->send_count; i += 1) {
    msg.data[1] = i;
    s->Send(&msg);
    //gpu_ring_poll(s);
    if (latencyflag) {
      while (s->total_received[MSG_PUT] < (i+1)) {
	s->Poll();
      }
    }
  }
  s->Drain();
}

void *ThreadFn(void * arg)
{
  struct CPURing *s = (CPURing *) arg;
  CPUThread(s, glob_size, glob_latency);
  return (NULL);
}

struct JoinStruct {
  sycl::event e;
  pthread_t thread;
  int loc_cpu;
  int * completion;
};
void *Joiner(void *arg)
{ 
  struct JoinStruct *p = (struct JoinStruct *) arg;
  if (p->loc_cpu) {
    pthread_join(p->thread, NULL);
  } else {
    p->e.wait();
  }
  *p->completion = 1;
  return(NULL);
}

void GPUThread(GPURing *s, size_t size, int latencyflag)
{  // gpu code
  struct RingMessage msg;
  msg.header = MSG_PUT;
  msg.data[0] = 0xfeedface;
  for (int i = 0; i < s->send_count; i += 1) {
    msg.data[1] = i;
    s->Send(&msg);
    if (latencyflag) {
      while (s->total_received[MSG_PUT] < (i+1)) {
	s->Poll();
      }
    }
  }
  s->Drain();
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
  uint64_t loc_a2b_count = glob_a2b_count;
  uint64_t loc_b2a_count = glob_b2a_count;
  int loc_a2b_buf = glob_a2b_buf;
  int loc_b2a_buf = glob_b2a_buf;
  uint64_t loc_size = glob_size;
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
    std::cout << "*Platform: " << pname << std::endl;
    if ( pname.find("Level-Zero") == std::string::npos) {
        std::cout << "non Level Zero GPU skipped" << std::endl;
        continue;
    }
    auto devices = p.get_devices();
    std::cout << "number of devices: " << devices.size() << std::endl;
    for (auto & d : devices ) {
      std::cout << "**Device: " << d.get_info<sycl::info::device::name>() << std::endl;
      if (d.is_gpu()) {
	if (qcount < 2) {
	  pvcq.push_back(sycl::queue(d, prop_list));
	  std::cout << "create pvcq[" << pvcq.size() - 1 << "]" << std::endl;
	  std::cout << "**max wg: " << d.get_info<cl::sycl::info::device::max_work_group_size>()  << std::endl;
	  qcount += 1;
	} else {
	  std::vector<sycl::device> sd = d.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::next_partitionable);
	  std::cout << " subdevices " << sd.size() << std::endl;
	  for (auto &subd: sd) {
	    tileq.push_back(sycl::queue(subd, prop_list));
	    std::cout << "create tileq[" << tileq.size() - 1 << "]" << std::endl;
	    std::cout << "**max wg: " << subd.get_info<cl::sycl::info::device::max_work_group_size>()  << std::endl;
	    qcount += 1;
	  }
	}
	if (qcount >= 4) break;
      }
      if (qcount >= 4) break;
    }
  }
  sycl::queue hostq = sycl::queue(sycl::cpu_selector_v, prop_list);
  std::cout << "create host queue" << std::endl;

  // xxxxxxxxxxxxx

  qs.push_back(tileq[0]);
  qs.push_back(tileq[1]);
  
  

  sycl::queue qa = qs[0];
  sycl::queue qb = qs[1];

  std::cout<<"qa selected device : "<<qa.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"qa device vendor : "<<qa.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  std::cout<<"qb selected device : "<<qb.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"qb device vendor : "<<qb.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  if (qa.get_device() == qb.get_device()) {
    std::cout << "qa and qb are the same device\n" << std::endl;
  }


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

  // well the below is a mess
 
  if (loc_a == LOC_CPU) {
    if (loc_a2b_buf == LOC_CPU) gpua_tx_mem = cpu_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC0) gpua_tx_mem = pvc0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_PVC1) gpua_tx_mem = pvc1_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE0) gpua_tx_mem = tile0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE1) gpua_tx_mem = tile1_a2b_hostmap;
    else assert(0);
  } else {
    if (loc_a2b_buf == LOC_CPU) gpua_tx_mem = cpu_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC0) gpua_tx_mem = pvc0_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC1) gpua_tx_mem = pvc1_a2b_mem;
    else if (loc_a2b_buf == LOC_TILE0) gpua_tx_mem = tile0_a2b_mem;
    else if (loc_a2b_buf == LOC_TILE1) gpua_tx_mem = tile1_a2b_mem;
    else assert(0);
  }
  if (loc_a == LOC_CPU) {
    if (loc_b2a_buf == LOC_CPU) gpua_rx_mem = cpu_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC0) gpua_rx_mem = pvc0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_PVC1) gpua_rx_mem = pvc1_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE0) gpua_rx_mem = tile0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE1) gpua_rx_mem = tile1_b2a_hostmap;
    else assert(0);
  } else {
    if (loc_b2a_buf == LOC_CPU) gpua_rx_mem = cpu_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC0) gpua_rx_mem = pvc0_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC1) gpua_rx_mem = pvc1_b2a_mem;
    else if (loc_b2a_buf == LOC_TILE0) gpua_rx_mem = tile0_b2a_mem;
    else if (loc_b2a_buf == LOC_TILE1) gpua_rx_mem = tile1_b2a_mem;
    else assert(0);
  }
  if (loc_b == LOC_CPU) {
    if (loc_b2a_buf == LOC_CPU) gpub_tx_mem = cpu_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC0) gpub_tx_mem = pvc0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_PVC1) gpub_tx_mem = pvc1_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE0) gpub_tx_mem = tile0_b2a_hostmap;
    else if (loc_b2a_buf == LOC_TILE1) gpub_tx_mem = tile1_b2a_hostmap;
    else assert(0);
  } else {
    if (loc_b2a_buf == LOC_CPU) gpub_tx_mem = cpu_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC0) gpub_tx_mem = pvc0_b2a_mem;
    else if (loc_b2a_buf == LOC_PVC1) gpub_tx_mem = pvc1_b2a_mem;
    else if (loc_b2a_buf == LOC_TILE0) gpub_tx_mem = tile0_b2a_mem;
    else if (loc_b2a_buf == LOC_TILE1) gpub_tx_mem = tile1_b2a_mem;
    else assert(0);
  }
  if (loc_b == LOC_CPU) {
    if (loc_a2b_buf == LOC_CPU) gpub_rx_mem = cpu_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC0) gpub_rx_mem = pvc0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_PVC1) gpub_rx_mem = pvc1_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE0) gpub_rx_mem = tile0_a2b_hostmap;
    else if (loc_a2b_buf == LOC_TILE1) gpub_rx_mem = tile1_a2b_hostmap;
    else assert(0);
  } else {
    if (loc_a2b_buf == LOC_CPU) gpub_rx_mem = cpu_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC0) gpub_rx_mem = pvc0_a2b_mem;
    else if (loc_a2b_buf == LOC_PVC1) gpub_rx_mem = pvc1_a2b_mem;
    else if (loc_a2b_buf == LOC_TILE0) gpub_rx_mem = tile0_a2b_mem;
    else if (loc_a2b_buf == LOC_TILE1) gpub_rx_mem = tile1_a2b_mem;
    else assert(0);
  }

  if (loc_a == LOC_CPU) {
    memset(gpua_tx_mem, 0, BUFSIZE);
  } else  {
    qa.memset(gpua_tx_mem, 0, BUFSIZE);
    qa.wait_and_throw();
  }
  if (loc_b == LOC_CPU) {
    memset(gpub_tx_mem, 0, BUFSIZE);
  } else {
    qb.memset(gpub_tx_mem, 0, BUFSIZE);
    qb.wait_and_throw();
  }
  GPURing *gpua;
  GPURing *gpub;
  CPURing *cpua;
  CPURing *cpub;

  GPURing *gpu_host_a;
  GPURing *gpu_host_b;
  CPURing *cpu_host_a;
  CPURing *cpu_host_b;

  if (loc_a == LOC_CPU) {
    cpua = new (sycl::aligned_alloc_host<CPURing>(4096, 1, qa)) CPURing(); 
    cpu_host_a = cpua;
    std::cout << " cpua " << &cpua << std::endl;
  } else {
    gpua = new (sycl::aligned_alloc_device<GPURing>(4096, 1,qa)) GPURing();
    gpu_host_a = get_mmap_address(gpua, 4096, qa);
    std::cout << " gpua " << &gpua << std::endl;
    std::cout << " gpu_host_a " << gpu_host_a << std::endl;
  }
  if (loc_b == LOC_CPU) {
    cpub = new (sycl::aligned_alloc_host<CPURing>(4096, 1, qb)) CPURing();
    cpu_host_b = cpub;
    std::cout << " cpub " << &cpub << std::endl;
  } else {
    gpub = new (sycl::aligned_alloc_device<GPURing>(4096, 1,qb)) GPURing();
    gpu_host_b = get_mmap_address(gpub, 4096, qb);
    std::cout << " gpub " << &gpub << std::endl;
    std::cout << " gpu_host_b " << gpu_host_b << std::endl;
  }


  if (loc_a == LOC_CPU) {
    cpua->InitState(gpua_tx_mem, gpua_rx_mem, loc_a2b_count, loc_b2a_count);
  } else {
    qa.single_task( [=]() {
	gpua->InitState(gpua_tx_mem, gpua_rx_mem, loc_a2b_count, loc_b2a_count);
      });
    qa.wait_and_throw();
  }
  if (loc_b == LOC_CPU) {
    cpub->InitState(gpub_tx_mem, gpub_rx_mem, loc_b2a_count, loc_a2b_count);
  } else {
    qb.single_task( [=]() {
	gpub->InitState(gpub_tx_mem, gpub_rx_mem, loc_b2a_count, loc_a2b_count);
      });
    qb.wait_and_throw();
  }
  std::cout << "printstate" << std::endl;

  if (loc_a == LOC_CPU) {
    cpua->Print("cpua");
  } else {
    gpu_host_a->Print("gpua");
  }
  if (loc_b == LOC_CPU) {
    cpub->Print("cpub");
  } else {
    gpu_host_b->Print("gpub");
  }

  std::cout << "kernel going to launch" << std::endl;

  unsigned long start_time, end_time;
  struct timespec ts_start, ts_end;
  
  clock_gettime(CLOCK_REALTIME, &ts_start);
  start_time = rdtsc();

  sycl::event ea;
  sycl::event eb;

  pthread_t athread;
  pthread_t bthread;
  pthread_attr_t pt_attributes;
  pthread_attr_init(&pt_attributes);
  struct JoinStruct ajoin, bjoin;
  int acompletion = 0;
  int bcompletion = 0;
  ajoin.completion = &acompletion;
  bjoin.completion = &bcompletion;
  
  if (loc_a == LOC_CPU) {
    pthread_create(&athread, &pt_attributes, ThreadFn, (void *) cpu_host_a);
    ajoin.loc_cpu = 1;
    ajoin.thread = athread;
  } else {
    ea = qa.single_task([=]() {
	GPUThread(gpua, loc_size, loc_latency);
      });
    ajoin.loc_cpu = 0;
    ajoin.e = ea;
  }
  if (loc_b == LOC_CPU) {
    pthread_create(&bthread, &pt_attributes, ThreadFn, (void *) cpu_host_b);
    bjoin.loc_cpu = 1;
    bjoin.thread = bthread;
  } else {
    eb = qb.single_task([=]() {
	GPUThread(gpub, loc_size, loc_latency);
      });
    bjoin.loc_cpu = 0;
    bjoin.e = eb;
  }
  pthread_t ajoiner, bjoiner;
  pthread_create(&ajoiner, &pt_attributes, Joiner, &ajoin);
  pthread_create(&bjoiner, &pt_attributes, Joiner, &bjoin);


  for (;;) {
    clock_gettime(CLOCK_REALTIME, &ts_end);
    end_time = rdtsc();
    if (acompletion && bcompletion) break;
    if (ts_end.tv_sec - ts_start.tv_sec > 8) {
      printf("TIMEOUT\n acomplete %d bcomplete %d", acompletion, bcompletion);
      break;
    }
  }
  if (acompletion) pthread_join(ajoiner, NULL);
  if (bcompletion) pthread_join(bjoiner, NULL);
  if (loc_a != LOC_CPU) printduration("gpua kernel ", ea);
  if (loc_b != LOC_CPU) printduration("gpub kernel ", eb);
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

  if (loc_a == LOC_CPU) {
    cpua->Print("cpua");
  } else {
    gpu_host_a->Print("gpua");
  }
  if (loc_b == LOC_CPU) {
    cpub->Print("cpub");
  } else {
    gpu_host_b->Print("gpub");
  }

  return 0;
}

