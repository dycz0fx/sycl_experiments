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
#include <stdarg.h>

#define DEBUG 1
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
#define CHERE() if (DEBUG) std::cout << __FUNCTION__ << ": " << __LINE__ << std::endl; 

#define NSEC_IN_SEC 1000000000.0
/* how many messages per buffer */
constexpr int N = 1024;

constexpr size_t BUFSIZE = (1L << 20);  //allocate 1 MB usable

/* message types, all non-zero */
#define MSG_IDLE 0
#define MSG_NOP 1
#define MSG_PUT 2
#define MSG_GET 3

/* should the length be implicit or explicit? */
struct Message {
  int32_t header;
  int32_t last_received;
  uint64_t data[7];
};
constexpr int MaxData = sizeof(uint64_t) * 7;

/* each end of the link has a State record */
struct State {
  int32_t next_send;        // next slot in sendbuf
  int32_t next_receive;     // next slot in recvbuf
  int32_t peer_last_received;  // last msg read by peer in sendbuf
  struct Message *sendbuf;   // remote buffer
  struct Message *recvbuf;   // local buffer
};

void initstate(struct State *s, struct Message *sendbuf, struct Message *recvbuf)
{
  HERE();
  s->next_send = 0;
  s->next_receive = 0;
  s->peer_last_received = (N - 1) % N;
  s->sendbuf = sendbuf;
  s->recvbuf = recvbuf;
}
int spaceavailable(struct State *s)
{
  HERE();
  return ((N + s->peer_last_received - s->next_send) % N);
}

void processmessage(struct Message *msg)
{
  HERE();
  /* do what the message says */
}

void cpureceive(struct State *s)
{
  CHERE();
  volatile struct Message *msg = &(s->recvbuf[s->next_receive]);
  if (msg->header != MSG_IDLE) {
    processmessage((struct Message *) msg);
    msg->header = 0;
    s->peer_last_received = msg->last_received;
    s->next_receive = (s->next_receive + 1) % N;
  }
}

void cpusend(struct State *s, int type, int length, void *data)
{
  CHERE();
  while (spaceavailable(s) == 0) cpureceive(s);
  struct Message msg;  // local composition of message
  msg.header = type;
  msg.last_received = (s->next_receive + N - 1) % N;
  assert (length <= MaxData);
  memcpy(&msg.data, data, length);  // local copy
  __m512i temp = _mm512_load_epi32((void *) &msg);
  void *mp = (void *) &(s->sendbuf[s->next_send]);
  _mm512_store_si512(mp, temp);
	      //  _movdir64b(&(s->sendbuf[s->next_send]), &msg);   // send message (atomic!)
  s->next_send = (s->next_send + 1) % N;
}

void cpudrain(struct State *s)
{
  CHERE();
  while (spaceavailable(s) < (N-1)) cpureceive(s);
}

void gpureceive(struct State *s)
{
  HERE();
  struct Message *msg = &(s->recvbuf[s->next_receive]);
  if (msg->header != MSG_IDLE) {
    processmessage(msg);
    msg->header = 0;
    s->peer_last_received = msg->last_received;
    s->next_receive = (s->next_receive + 1) % N;
  }
}

void gpusend(struct State *s, int type, int length, void *data)
{
  HERE();
  while (spaceavailable(s) == 0) gpureceive(s);
  struct Message msg;  // local composition of message
  msg.header = type;
  msg.last_received = (s->next_receive + N - 1) % N;
  assert (length <= MaxData);
  memcpy(&msg.data, data, length);  // local copy
  s->sendbuf[s->next_send] = msg;   // send message (atomic!)
  s->next_send = (s->next_send + 1) % N;
}

void gpudrain(struct State *s)
{
  HERE();
  while (spaceavailable(s) < (N-1)) gpureceive(s);
}

#define cpu_relax() asm volatile("rep; nop")
#define nullptr NULL

/* option codes */
#define CMD_COUNT 1001
#define CMD_SIZE 1002
#define CMD_HELP 1004
#define CMD_VALIDATE 1005
#define CMD_ATOMICSTORE 1019
#define CMD_ATOMICLOAD 1020


/* option global variables */
// specify defaults
uint64_t glob_count = 1;
uint64_t glob_size = 0;
int glob_validate = 0;

int glob_use_atomic_load = 0;
int glob_use_atomic_store = 0;

void Usage()
{
  std::cout <<
    "--count <n>            set number of iterations\n"
    "--size <n>             set size\n"
    "--help                 usage message\n"
    "--validate             set and check data\n"
    "--atomicload=0/1 | --useatomicstore=0/1         method of flag access on device\n";
  std::cout << std::endl;
  exit(1);
}



void ProcessArgs(int argc, char **argv)
{
  const char* short_opts = "c:r:w:vhDHALM";
  const option long_opts[] = {
    {"count", required_argument, nullptr, CMD_COUNT},
    {"size", required_argument, nullptr, CMD_SIZE},
    {"help", no_argument, nullptr, CMD_HELP},
    {"validate", no_argument, nullptr, CMD_VALIDATE},
    {"atomicload", required_argument, nullptr, CMD_ATOMICLOAD},
    {"atomicstore", required_argument, nullptr, CMD_ATOMICSTORE},
    {nullptr, no_argument, nullptr, 0}
  };
  while (true)
    {
      const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
      if (-1 == opt)
	break;
      switch (opt)
	{
	case CMD_COUNT: {
	  glob_count = std::stoi(optarg);
	  std::cout << "count " << glob_count << std::endl;
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

constexpr int cpuonly = 1;

int main(int argc, char *argv[]) {
  ProcessArgs(argc, argv);
  uint64_t loc_count = glob_count;
  uint64_t loc_size = glob_size;
  int loc_validate = glob_validate;
  int loc_use_atomic_load = glob_use_atomic_load;
  int loc_use_atomic_store = glob_use_atomic_store;

  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
  sycl::queue Q(sycl::gpu_selector{}, prop_list);
  std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() << std::endl;

  // allocate device memory

  struct Message *device_data_mem = (struct Message *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, Q);

  //allocate host mamory
  struct Message *host_data_mem = (struct Message *) sycl::aligned_alloc_host(4096, BUFSIZE, Q);
  std::cout << " host_data_mem " << host_data_mem << std::endl;

  //create mmap mapping of usm device memory on host
  sycl::context ctx = Q.get_context();

  struct Message *device_data_hostmap;   // pointer for host to use
  device_data_hostmap = get_mmap_address(device_data_mem, BUFSIZE, Q);

  // initialize host mamory
  memset(&host_data_mem[0], 0, BUFSIZE);

  // initialize device memory  
  auto e = Q.submit([&](sycl::handler &h) {
      h.memcpy(device_data_mem, &host_data_mem[0], BUFSIZE);
    });
  e.wait_and_throw();
  printduration("memcpy kernel ", e);

  struct State *cpu = (struct State *) sycl::aligned_alloc_host(4096, sizeof(struct State), Q);
  struct State *gpu = (struct State *) sycl::aligned_alloc_device(4096, sizeof(struct State), Q);

  /* initialize state records */
  memset(cpu, 0, sizeof(struct State));
// initialize device memory  
  e = Q.submit([&](sycl::handler &h) {
      h.memcpy(gpu, cpu, sizeof(struct State));
    });
  e.wait_and_throw();
  initstate(cpu, device_data_hostmap, host_data_mem);
  e = Q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(1), [=](sycl::group<1> grp) {
	  initstate(gpu, host_data_mem, device_data_mem);
	});
    });
  e.wait_and_throw();
  
  

  std::cout << "kernel going to launch" << std::endl;
  unsigned long start_time, end_time;
  struct timespec ts_start, ts_end;

  e = Q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(1), [=](sycl::group<1> grp) {
	  uint64_t msgdata[7] = {0xdeadbeef, 0, 0, 0, 0, 0, 0};
	  for (int i = 0; i < loc_count; i += 1) {
	    msgdata[1] = i;
	    gpusend(gpu, MSG_NOP, loc_size, msgdata);
	  }
	  gpudrain(gpu);
	});
    });

    {  // cpu code
      uint64_t msgdata[7] = {0xfeedface, 0, 0, 0, 0, 0, 0};
      for (int i = 0; i < loc_count; i += 1) {
	msgdata[1] = i;
	cpusend(cpu, MSG_NOP, loc_size, msgdata);
      }
      cpudrain(cpu);
    }
    
  e.wait_and_throw();
  printduration("gpu kernel ", e);
    /* common cleanup */
    double elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	((double) (ts_end.tv_nsec - ts_start.tv_nsec));
    //      std::cout << "count " << loc_count << " tsc each " << (end_time - start_time) / loc_count << std::endl;
    std::cout << argv[0];
    std::cout << "--count " << loc_count << " ";
    std::cout << "--size " << loc_size << " ";

    double size = loc_size;
    double mbps = (size * 1000) / (elapsed / ((double) loc_count));
    double nsec = elapsed / ((double) loc_count);
    std::cout << " nsec each " << nsec << " MB/s " << mbps << std::endl;

    munmap(device_data_hostmap, BUFSIZE);
    sycl::free(device_data_mem, Q);
    sycl::free(gpu, Q);
    return 0;
}

