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
#include "ringlib3.cpp"
#include <omp.h>
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
#define CMD_DISCARD 1028
#define CMD_VERBOSE 1029
#define CMD_NMSG 2030

/* option global variables */
// specify defaults
uint64_t glob_a2b_count = 0;
uint64_t glob_b2a_count = 0;
int glob_validate = 0;
int glob_cpu = 0;
int glob_interlock = 0;
int glob_latency = 0;
int glob_discard = 0;
int glob_athreads = 1;
int glob_bthreads = 1;
int glob_a2b_buf = 0;  /* 0: cpu 1: p0 2: p1 3: t0 4: t1 */
int glob_b2a_buf = 0;  /* 0: cpu 1: p0 2: p1 3: t0 4: t1 */
int glob_a = 0;   /* 0: cpu 1: p0 2: p1 3: t0 4: t1 */
int glob_b = 0;   /* 0: cpu 1: p0 2: p1 3: t0 4: t1 */
int glob_verbose = 0;
int glob_nmsg = 1;
void Usage()
{
  std::cout <<
    "--a= p0|p1|t0|t1|c      set a end to pvc 0 or 1, tile 0 or 1, or cpu\n"
    "--b= p0|p1|t0|t1|c      set a end to pvc 0 or 1, tile 0 or 1, or cpu\n"
    "--a2bbuf p0|p1|t0|t1|c  set location of gpu-a to gpu-b buffer\n"
    "--b2abuf p0|p1|t0|t1|c  set location of gpu-b to gpu=a buffer\n"
    "--a2bcount <n>          set number of iterations\n"
    "--b2acount <n>          set number of iterations\n"
    "--athreads <n>          number threads for end a\n"
    "--bthreads <n>          number threads for end b\n"
    "--help                  usage message\n"
    "--validate              set and check data\n"
    "--latency               wait for responses\n"
    "--discard               discard receive traffic\n"
    "--verbose               more printing\n"
    "--nmsg=<n>              Poll dwell tries\n";
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

const char *code_to_location(int code)
{
  if (code == LOC_CPU) return("c");
  if (code == LOC_PVC0) return("p0");
  if (code == LOC_PVC1) return("p1");
  if (code == LOC_TILE0) return("t0");
  if (code == LOC_TILE1) return("t1");
  return("unknown");
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
    {"athreads", required_argument, nullptr, CMD_ATHREADS},
    {"bthreads", required_argument, nullptr, CMD_BTHREADS},
    {"help", no_argument, nullptr, CMD_HELP},
    {"validate", no_argument, nullptr, CMD_VALIDATE},
    {"latency", no_argument, nullptr, CMD_LATENCY},
    {"discard", no_argument, nullptr, CMD_DISCARD},
    {"verbose", no_argument, nullptr, CMD_VERBOSE},
    {"nmsg", required_argument, nullptr, CMD_NMSG},
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
	  //std::cout << "--a2bcount=" << glob_a2b_count << std::endl;
	  break;
	}
	case CMD_B2A_COUNT: {
	  glob_b2a_count = std::stoi(optarg);
	  //std::cout << "--b2acount=" << glob_b2a_count << std::endl;
	  break;
	}
	case CMD_A: {
	  glob_a = location_to_code(optarg);
	  //std::cout << "--a=" << glob_a << std::endl;
	  break;
	}
	case CMD_B: {
	  glob_b = location_to_code(optarg);
	  //std::cout << "--b=" << glob_b << std::endl;
	  break;
	}
	case CMD_A2B_BUF: {
	  glob_a2b_buf = location_to_code(optarg);
	  //std::cout << "--a2bbuf=" << glob_a2b_buf << std::endl;
	  break;
	}
	case CMD_B2A_BUF: {
	  glob_b2a_buf = location_to_code(optarg);
	  //std::cout << "--b2abuf=" << glob_b2a_buf << std::endl;
	  break;
	}
	case CMD_ATHREADS: {
	  glob_athreads = std::stoi(optarg);
	  //std::cout << "--athreads=" << glob_athreads << std::endl;
	  break;
	}
	case CMD_BTHREADS: {
	  glob_bthreads = std::stoi(optarg);
	  //std::cout << "--bthreads=" << glob_bthreads << std::endl;
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
	  //std::cout << "--latency=" << glob_latency << std::endl;
	  break;
	}
	case CMD_DISCARD: {
	  glob_discard = 1;
	  break;
	}
	case CMD_VERBOSE: {
	  glob_verbose = 1;
	  break;
	}
	case CMD_NMSG: {
	  glob_nmsg = std::stoi(optarg);
	  //std::cout << "--nmsg=" << glob_nmsg << std::endl;
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
  int discard_flag;
};
  
constexpr size_t BUFSIZE = (1L << 24);  //allocate 1 MB usable

void CPUSendThread(CPURingCommand *r)
{  // cpu code
  CPURing *s = r->cpur;
  int send_count = r->send_count;
  struct RingMessage msg;
  msg.header = MSG_PUT;
  msg.data[0] = 0xdeadbeef;
  for (int i = 0; i < send_count; i += 1) {
    //msg.data[1] = i;
    s->Send(&msg);
  }
}

void *ThreadSendFn(void * arg)
{
  struct CPURingCommand *r = (CPURingCommand *) arg;
  CPUSendThread(r);
  return (NULL);
}

void CPUReceiveThread(CPURingCommand *r)
{  // cpu code
  CPURing *s = r->cpur;
  int recv_count = r->recv_count;
  while ((s->next_receive-RingN) < recv_count) s->Drain();
}

void *ThreadReceiveFn(void * arg)
{
  struct CPURingCommand *r = (CPURingCommand *) arg;
  CPUReceiveThread(r);
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
  usleep(5000 * omp_get_thread_num()); // do this to avoid race condition while printing
  //std::cout << "Number of available threads: " << omp_get_num_threads() << std::endl;
  // each thread can also get its own number
  //std::cout << "Current thread number: " << omp_get_thread_num() << std::endl;
  ProcessArgs(argc, argv);
  uint64_t loc_a2b_count = glob_a2b_count;
  uint64_t loc_b2a_count = glob_b2a_count;
  int loc_a2b_buf = glob_a2b_buf;
  int loc_b2a_buf = glob_b2a_buf;
  int loc_athreads = glob_athreads;
  int loc_bthreads = glob_bthreads;
  int loc_validate = glob_validate;
  int loc_latency = glob_latency;
  int loc_discard = glob_discard;
  int loc_a = glob_a;
  int loc_b = glob_b;
  int loc_nmsg = glob_nmsg;
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
  std::vector<sycl::queue> qs;


  // xxxxxxxxxxxx
  std::vector<sycl::queue> pvcq1;
  std::vector<sycl::queue> tileq1;
  std::vector<sycl::queue> pvcq2;
  std::vector<sycl::queue> tileq2;

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
	  pvcq1.push_back(sycl::queue(d, prop_list));
	  pvcq2.push_back(sycl::queue(d, prop_list));
	  //std::cout << "create pvcq[" << pvcq.size() - 1 << "]" << std::endl;
	  //std::cout << "**max wg: " << d.get_info<cl::sycl::info::device::max_work_group_size>()  << std::endl;
	  qcount += 1;
	} else {
	  std::vector<sycl::device> sd = d.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::next_partitionable);
	  //std::cout << " subdevices " << sd.size() << std::endl;
	  for (auto &subd: sd) {
	    tileq1.push_back(sycl::queue(subd, prop_list));
	    tileq2.push_back(sycl::queue(subd, prop_list));
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


  sycl::queue qa1;
  sycl::queue qb1;
  //  assert(loc_a != loc_b);
  if (loc_a == LOC_CPU) qa1 = hostq;
  else if (loc_a == LOC_PVC0) qa1 = pvcq1[0];
  else if (loc_a == LOC_PVC1) qa1 = pvcq1[1];
  else if (loc_a == LOC_TILE0) qa1 = tileq1[0];
  else if (loc_a == LOC_TILE1) qa1 = tileq1[1];
  else assert(0);
  if (loc_b == LOC_CPU) qb1 = hostq;
  else if (loc_b == LOC_PVC0) qb1 = pvcq1[0];
  else if (loc_b == LOC_PVC1) qb1 = pvcq1[1];
  else if (loc_b == LOC_TILE0) qb1 = tileq1[0];
  else if (loc_b == LOC_TILE1) qb1 = tileq1[1];
  else assert(0);

  sycl::queue qa2;
  sycl::queue qb2;
  //  assert(loc_a != loc_b);
  if (loc_a == LOC_CPU) qa2 = hostq;
  else if (loc_a == LOC_PVC0) qa2 = pvcq2[0];
  else if (loc_a == LOC_PVC1) qa2 = pvcq2[1];
  else if (loc_a == LOC_TILE0) qa2 = tileq2[0];
  else if (loc_a == LOC_TILE1) qa2 = tileq2[1];
  else assert(0);
  if (loc_b == LOC_CPU) qb2 = hostq;
  else if (loc_b == LOC_PVC0) qb2 = pvcq2[0];
  else if (loc_b == LOC_PVC1) qb2 = pvcq2[1];
  else if (loc_b == LOC_TILE0) qb2 = tileq2[0];
  else if (loc_b == LOC_TILE1) qb2 = tileq2[1];
  else assert(0);


  
  //std::cout<<"qa1 selected device : "<<qa1.get_device().get_info<sycl::info::device::name>() << std::endl;
  //std::cout<<"qa1 device vendor : "<<qa1.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  //std::cout<<"qb1 selected device : "<<qb1.get_device().get_info<sycl::info::device::name>() << std::endl;
  //std::cout<<"qb1 device vendor : "<<qb1.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  //if (qa1.get_device() == qb1.get_device()) {
  //std::cout << "qa1 and qb1 are the same device\n" << std::endl;
  //}
  //std::cout<<"qa2 selected device : "<<qa2.get_device().get_info<sycl::info::device::name>() << std::endl;
  //std::cout<<"qa2 device vendor : "<<qa2.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  //std::cout<<"qb2 selected device : "<<qb2.get_device().get_info<sycl::info::device::name>() << std::endl;
  //std::cout<<"qb2 device vendor : "<<qb2.get_device().get_info<sycl::info::device::vendor>() << std::endl;


  struct RingMessage *cpu_a2b_mem = (struct RingMessage *) sycl::aligned_alloc_host(4096, BUFSIZE * 2, hostq);
  struct RingMessage *pvc0_a2b_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, pvcq1[0]);
  struct RingMessage *pvc1_a2b_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, pvcq1[1]);
  struct RingMessage *tile0_a2b_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, tileq1[0]);
  struct RingMessage *tile1_a2b_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, tileq1[1]);
  struct RingMessage *cpu_b2a_mem = (struct RingMessage *) sycl::aligned_alloc_host(4096, BUFSIZE * 2, hostq);
  struct RingMessage *pvc0_b2a_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, pvcq1[0]);
  struct RingMessage *pvc1_b2a_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, pvcq1[1]);
  struct RingMessage *tile0_b2a_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, tileq1[0]);
  struct RingMessage *tile1_b2a_mem = (struct RingMessage *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, tileq1[1]);

  struct RingMessage *pvc0_a2b_hostmap = get_mmap_address(pvc0_a2b_mem, BUFSIZE, pvcq1[0]);
  struct RingMessage *pvc1_a2b_hostmap = get_mmap_address(pvc1_a2b_mem, BUFSIZE, pvcq1[1]);
  struct RingMessage *tile0_a2b_hostmap = get_mmap_address(tile0_a2b_mem, BUFSIZE, tileq1[0]);
  struct RingMessage *tile1_a2b_hostmap = get_mmap_address(tile1_a2b_mem, BUFSIZE, tileq1[1]);
  struct RingMessage *pvc0_b2a_hostmap = get_mmap_address(pvc0_b2a_mem, BUFSIZE, pvcq1[0]);
  struct RingMessage *pvc1_b2a_hostmap = get_mmap_address(pvc1_b2a_mem, BUFSIZE, pvcq1[1]);
  struct RingMessage *tile0_b2a_hostmap = get_mmap_address(tile0_b2a_mem, BUFSIZE, tileq1[0]);
  struct RingMessage *tile1_b2a_hostmap = get_mmap_address(tile1_b2a_mem, BUFSIZE, tileq1[1]);
  
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
  //std::cout << "gpua_tx_mem" << gpua_tx_mem << std::endl;
  //std::cout << "gpub_rx_mem" << gpub_rx_mem << std::endl;
  //std::cout << "gpub_tx_mem" << gpub_tx_mem << std::endl;
  //std::cout << "gpua_rx_mem" << gpua_rx_mem << std::endl;
		       
  CPURing *cpua;
  CPURing *cpub;
  GPURing *ringspacea;
  GPURing *ringspaceb;
  GPURing *ringspacea_host_map;
  GPURing *ringspaceb_host_map;


  CPURing *cpu_host_a;
  CPURing *cpu_host_b;

  if (loc_a == LOC_CPU) {
    cpua = new (sycl::aligned_alloc_host<CPURing>(4096, 1, qa1)) CPURing(); 
    cpu_host_a = cpua;
    //std::cout << " cpua " << &cpua << std::endl;
  } else {
    ringspacea = sycl::aligned_alloc_device<GPURing>(4096, 2, qa1);
    ringspacea_host_map = (GPURing *) get_mmap_address(ringspacea, sizeof(GPURing) * 2, qa1);
  }
  if (loc_b == LOC_CPU) {
    cpub = new (sycl::aligned_alloc_host<CPURing>(4096, 1, qb1)) CPURing();
    cpu_host_b = cpub;
    //std::cout << " cpub " << &cpub << std::endl;
  } else {
    ringspaceb = sycl::aligned_alloc_device<GPURing>(4096, 2, qb1);
    ringspaceb_host_map = (GPURing *) get_mmap_address(ringspaceb, sizeof(GPURing) * 2, qb1);
  }

  std::cout << "kernel going to launch" << std::endl;

  unsigned long start_time, end_time;
  struct timespec ts_start, ts_end;
  

  sycl::event ea1;
  sycl::event eb1;
  sycl::event ea2;
  sycl::event eb2;

  if (loc_a == LOC_CPU) {
    memset(gpua_tx_mem, 101, 4096);
  } else {
    qa1.memset(gpua_tx_mem, 101, 4096);
    qa1.wait_and_throw();
  }
  if (loc_b == LOC_CPU) {
    memset(gpub_tx_mem, 102, 4096);
  } else {
    qb1.memset(gpub_tx_mem, 102, 4096);
    qb1.wait_and_throw();
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
    auto e = qa1.single_task( [=]() {
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
    auto e = qb1.single_task( [=]() {
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
  
  /* used for LOC_CPU cases only */
  pthread_t a1thread;
  pthread_t b1thread;
  pthread_t a2thread;
  pthread_t b2thread;
  
  pthread_attr_t pt_attributes;
  pthread_attr_init(&pt_attributes);

  struct JoinStruct a1join, b1join;
  int a1completion = 0;
  int b1completion = 0;
  a1join.completion = &a1completion;
  b1join.completion = &b1completion;

  struct CPURingCommand acmd;
  acmd.cpur = cpu_host_a;
  acmd.send_count = loc_a2b_count;
  acmd.recv_count = loc_b2a_count;
  acmd.latency_flag = loc_latency;
  acmd.discard_flag = loc_discard;
  struct CPURingCommand bcmd;
  bcmd.cpur = cpu_host_b;
  bcmd.send_count = loc_b2a_count;
  bcmd.recv_count = loc_a2b_count;
  bcmd.latency_flag = loc_latency;
  bcmd.discard_flag = loc_discard;

  /* used in cases with separate receive threads */
  struct JoinStruct a2join, b2join;
  int a2completion = 0;
  int b2completion = 0;
  a2join.completion = &a2completion;
  b2join.completion = &b2completion;

  clock_gettime(CLOCK_REALTIME, &ts_start);
  start_time = rdtsc();
  
  if (loc_a == LOC_CPU) {
    pthread_create(&a1thread, &pt_attributes, ThreadSendFn, (void *) &acmd);
    a1join.is_pthread = 1;
    a1join.thread = a1thread;
    pthread_create(&a2thread, &pt_attributes, ThreadReceiveFn, (void *) &acmd);
    a2join.is_pthread = 1;
    a2join.thread = a2thread;
  } else {
    int send_count = loc_a2b_count;
    int recv_count = loc_b2a_count;
    std::cout << "GPUA send_count " << send_count << " recv_count " << recv_count << std::endl;
    ea1 = qa1.submit([&](sycl::handler &h) {
	//auto out = sycl::stream(1024, 768, h);
	h.parallel_for_work_group(sycl::range(1), sycl::range(loc_athreads), [=](sycl::group<1> grp) {
	    GPURing *gpua = new(&ringspacea[0]) GPURing(23, gpua_tx_mem, gpua_rx_mem);
	    grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		struct RingMessage msg;
		msg.header = MSG_PUT;
		int si = it.get_global_id()[0];  // send index
		while (si < send_count) {
		  msg.data[1] = si;
		  gpua->Send(&msg);
		  si += loc_athreads;
		}
	      });
	  });
      });
    ea2 = qa2.submit([&](sycl::handler &h) {
	//auto out = sycl::stream(1024, 768, h);
	h.parallel_for_work_group(sycl::range(1), sycl::range(loc_athreads), [=](sycl::group<1> grp) {
	    GPURing *gpua = new(&ringspacea[1]) GPURing(24, gpua_tx_mem, gpua_rx_mem);
	    grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		while (gpua->receive_count < recv_count) gpua->Poll(loc_nmsg);
	      });
	  });
      });
    a1join.is_pthread = 0;
    a1join.e = ea1;
    a2join.is_pthread = 0;
    a2join.e = ea2;
  }
  if (loc_b == LOC_CPU) {
    pthread_create(&b1thread, &pt_attributes, ThreadSendFn, (void *) &bcmd);
    b1join.is_pthread = 1;
    b1join.thread = b1thread;
    pthread_create(&b2thread, &pt_attributes, ThreadReceiveFn, (void *) &bcmd);
    b2join.is_pthread = 1;
    b2join.thread = b2thread;
  } else {
    int send_count = loc_b2a_count;
    int recv_count = loc_a2b_count;
    std::cout << "GPUB send_count " << send_count << " recv_count " << recv_count << std::endl;
    eb1 = qb1.submit([&](sycl::handler &h) {
	//auto out = sycl::stream(1024, 768, h);
	h.parallel_for_work_group(sycl::range(1), sycl::range(loc_bthreads), [=](sycl::group<1> grp) {
	    GPURing *gpub = new(&ringspaceb[0]) GPURing(43, gpub_tx_mem, gpub_rx_mem);
	    grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		struct RingMessage msg;
		msg.header = MSG_PUT;
		int si = it.get_global_id()[0];  // send index
		while (si < send_count) {
		  msg.data[1] = si;
		  gpub->Send(&msg);
		  si += loc_bthreads;
		}
	      });
	  });
      });
    eb2 = qb2.submit([&](sycl::handler &h) {
	//auto out = sycl::stream(1024, 768, h);
	h.parallel_for_work_group(sycl::range(1), sycl::range(loc_bthreads), [=](sycl::group<1> grp) {
	    GPURing *gpub = new(&ringspaceb[1]) GPURing(44, gpub_tx_mem, gpub_rx_mem);
	    grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		while (gpub->receive_count < recv_count) gpub->Poll(loc_nmsg);
	      });
	  });
      });
    b1join.is_pthread = 0;
    b1join.e = eb1;
    b2join.is_pthread = 0;
    b2join.e = eb2;
  }
  pthread_t a1joiner, b1joiner;
  pthread_create(&a1joiner, &pt_attributes, Joiner, &a1join);
  pthread_create(&b1joiner, &pt_attributes, Joiner, &b1join);
  pthread_t a2joiner, b2joiner;
  pthread_create(&a2joiner, &pt_attributes, Joiner, &a2join);
  pthread_create(&b2joiner, &pt_attributes, Joiner, &b2join);

  //std::cout << "starting timeout check" << std::endl;
  for (;;) {
    clock_gettime(CLOCK_REALTIME, &ts_end);
    end_time = rdtsc();
    if (a1completion && b1completion && a2completion && b2completion) break;
    if (ts_end.tv_sec - ts_start.tv_sec > 16) {
      printf("TIMEOUT a1complete %d b1complete %d a2complete %d b2complete %d\n",
	     a1completion, b1completion, a2completion, b2completion);
      fflush(stdout);
      break;
    }
  }
  //std::cout << "got completion flags" << std::endl;
  if (a1completion) pthread_join(a1joiner, NULL);
  if (b1completion) pthread_join(b1joiner, NULL);
  if (a2completion) pthread_join(a2joiner, NULL);
  if (b2completion) pthread_join(b2joiner, NULL);
  //std::cout << "joined complete joiners" << std::endl;
  if ((loc_a != LOC_CPU) && a1completion) printduration("gpua tx kernel ", ea1);
  if ((loc_a != LOC_CPU) && a2completion) printduration("gpua rx kernel ", ea2);
  if ((loc_b != LOC_CPU) && b1completion) printduration("gpub tx kernel ", eb1);
  if ((loc_b != LOC_CPU) && b2completion) printduration("gpub rx kernel ", eb2);
  //std::cout << "printduration" << std::endl;

  /* common cleanup */
  double elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
    ((double) (ts_end.tv_nsec - ts_start.tv_nsec));
  
  double fcount;
  if (loc_a2b_count > loc_b2a_count)
    fcount = loc_a2b_count;
  else
    fcount = loc_b2a_count;
  double nsec = elapsed / fcount;
  std::cout << "elapsed " << elapsed << " fcount " << fcount << std::endl;
  std::cout << argv[0];
  std::cout << " --a=" << code_to_location(loc_a);
  std::cout << " --b=" << code_to_location(loc_b);
  std::cout << " --a2bbuf=" << code_to_location(loc_a2b_buf);
  std::cout << " --b2abuf=" << code_to_location(loc_b2a_buf);
  std::cout << " --a2bcount=" << loc_a2b_count;
  std::cout << " --b2acount=" << loc_b2a_count;
  std::cout << " --athreads=" << loc_athreads;
  std::cout << " --bthreads=" << loc_bthreads;
  std::cout << " --nmsg=" << loc_nmsg;
  if (loc_latency) std::cout << " --latency";
  std::cout << " nsec=" << nsec << std::endl;
  if (loc_b2a_count <= 2000) {
    int err = 0;
    for (int i = 0; i < loc_b2a_count; i += 1) {
      unsigned v = gpua_rx_mem_hostmap[i%RingN].sequence;
      if (v != (i + RingN) ) {
	err += 1;
	printf("gpua_rx_mem[%d] == %d expected %d seq %i\n", i, v, i, gpub_rx_mem_hostmap[i%RingN].sequence);
      }
    }
    printf("gpua_rx_mem errors %d\n", err);
  }
  if (loc_a2b_count <= 2000) {
    int err = 0;
    for (int i = 0; i < loc_a2b_count; i += 1) {
      unsigned v = gpub_rx_mem_hostmap[i%RingN].sequence;
      if (v != (i + RingN)) {
	err += 1;
	printf("gpub_rx_mem[%d] == %d expected %d seq %d\n", i, v, i, gpub_rx_mem_hostmap[i%RingN].sequence);
      }
    }
    printf("gpub_rx_mem errors %d\n", err);
  }
  fflush(stdout);
  if (glob_verbose || (a1completion == 0) || (b1completion == 0)) {
    printf("gpua next_peer_receive %d\n", gpua_rx_mem_hostmap[RingN].sequence);
    printf("gpub next_peer_receive %d\n", gpub_rx_mem_hostmap[RingN].sequence);
    
    
    if (loc_a == LOC_CPU) {
      cpua->Print("cpua");
    } else {
      unsigned low = ringspacea_host_map[0].receive_count - 20;
      unsigned high = ringspacea_host_map[0].receive_count + 20;
      for (unsigned i = low; i <= high; i += 1) {
	std::cout << "gpua tx recvbuf[" << i << "] = " << gpua_rx_mem_hostmap[i%RingN].sequence << std::endl;
      }
      ringspacea_host_map[0].Print("gpua tx");
       low = ringspacea_host_map[1].receive_count - 20;
       high = ringspacea_host_map[1].receive_count + 20;
      for (unsigned i = low; i <= high; i += 1) {
	std::cout << "gpua 4x recvbuf[" << i << "] = " << gpua_rx_mem_hostmap[i%RingN].sequence << std::endl;
      }
      ringspacea_host_map[1].Print("gpua rx");
    }
    if (loc_b == LOC_CPU) {
      cpub->Print("cpub");
    } else {
      unsigned low = ringspaceb_host_map[0].receive_count - 20;
      unsigned high = ringspaceb_host_map[0].receive_count + 20;
      for (unsigned i = low; i <= high; i += 1) {
	std::cout << "gpub tx recvbuf[" << i << "] = " << gpub_rx_mem_hostmap[i%RingN].sequence << std::endl;
      }
      ringspaceb_host_map[0].Print("gpub tx");
       low = ringspaceb_host_map[1].receive_count - 20;
       high = ringspaceb_host_map[1].receive_count + 20;
      for (unsigned i = low; i <= high; i += 1) {
	std::cout << "gpub rx recvbuf[" << i << "] = " << gpub_rx_mem_hostmap[i%RingN].sequence << std::endl;
      }
      ringspaceb_host_map[1].Print("gpub rx");
    }
  }
  std::cout << "main returning" << std::endl;
  return 0;
}

