#include <stdlib.h>
#include <unistd.h>
#include <thread>
#include <getopt.h>
#include <iostream>
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

#define DEBUG 0
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

constexpr int N = 1024;
constexpr int NOP_THRESHOLD = 256;   /* N/4 ? See analysis */

constexpr size_t BUFSIZE = (1L << 20);  //allocate 1 MB usable

/* message types, all non-zero */
#define MSG_IDLE 0
#define MSG_NOP 1
#define MSG_PUT 2
#define MSG_GET 3

/* should the length be implicit or explicit? */
struct RingMessage {
  int32_t header;
  int32_t next_receive;
  uint64_t data[7];
};
constexpr int MaxData = sizeof(uint64_t) * 7;

/* when peer_next_read and next_write pointers are equal, the buffer is empty.
 * when next_write pointer is one less than peer_next_read pointer, 
 * the buffer is full
 *
 * must not send last message without returning credit
 * should return NOP with credit when half of credits waiting to return
 *
 * need to know how many credits waiting to return
 *   next_receive - next_received_sent
 */

/* each end of the link has a RingState record */

struct RingState {
  int32_t next_send;        // next slot in sendbuf
  int32_t next_receive;     // next slot in recvbuf
  int32_t peer_next_receive;  // last msg read by peer in sendbuf
  int32_t peer_next_receive_sent; // last time next_receive was sent
  struct RingMessage *sendbuf;   // remote buffer
  struct RingMessage *recvbuf;   // local buffer
  int32_t total_receive;     // for accounting
  const char *name;
};

void initstate(struct RingState *s, struct RingMessage *sendbuf, struct RingMessage *recvbuf, const char *name)
{
  s->next_send = 0;
  s->next_receive = 0;
  s->peer_next_receive = 0;
  s->sendbuf = sendbuf;
  s->recvbuf = recvbuf;
  s->total_receive = 0;
  s->name = name;
}

/* how many messages can we send? */
int ring_send_space_available(struct RingState *s)
{
  int space = ((N - 1) + s->peer_next_receive - s->next_send) % N;
  if (DEBUG) printf("space %s: %d\n", s->name, space);
  return (space);
}

/* how many credits can we return? 
 * s->next_receive is the next message we haven't received yet
 * s->peer_next_receive_sent is the last next_receive we told the peer about
 * s->next_receive - s->peer_next_receive is the number of messages the peer can
 * send but we haven't told them yet.
 * The idea is that if this number gets larger than N/2 we should send an
 * otherwise empty message to update the peer
 *
 * if there is traffic in both directions, this won't trigger.  If it is needed
 * the trigger threshold should be set to N/2 or some fraction of N such that
 * the update reaches the peer before they run out of credits
 * 
 * The number of slots in the ring buffer should be enough to cover about 2 x
 * the round trip latency so that we can return credits before the peer runs out
 * when all the traffic is one-way
 */
int ring_internal_send_nop_p(struct RingState *s)
{
  int cr_to_send =  (N + s->next_receive - s->peer_next_receive_sent) % N;
  if (DEBUG) printf("send_nop_p %s: %d\n", s->name, cr_to_send);
  return (cr_to_send > NOP_THRESHOLD);
}

void ring_process_message(struct RingState *s, struct RingMessage *msg)
{
  if (DEBUG) printf("process %s %d (credit %d\n)\n", s->name, s->total_receive, s->peer_next_receive);
    s->total_receive += 1;
  /* do what the message says */
}

/* called by ring_cpu_send when we need space to send a message */
void ring_internal_cpu_receive(struct RingState *s)
{
  /* volatile */ struct RingMessage *msg = &(s->recvbuf[s->next_receive]);
  if (msg->header != MSG_IDLE) {
    s->peer_next_receive = msg->next_receive;
    ring_process_message(s, (struct RingMessage *) msg);
    msg->header = MSG_IDLE;
    s->next_receive = (s->next_receive + 1) % N;
  }
}

void send_nop(struct RingState *s)
{
  if (DEBUG) printf("send %s next %d\n", s->name, s->next_send);
  do {
    ring_internal_cpu_receive(s);
  } while (ring_send_space_available(s) == 0);
  struct RingMessage msg;  // local composition of message
  msg.header = MSG_NOP;
  msg.next_receive = s->next_receive;
  s->peer_next_receive_sent = s->next_receive;
  void *mp = (void *) &(s->sendbuf[s->next_send]);
  __m512i temp = _mm512_load_epi32((void *) &msg);
  _mm512_store_si512(mp, temp);
  //  _movdir64b(&(s->sendbuf[s->next_send]), &msg);   // send message (atomic!)
  s->next_send = (s->next_send + 1) % N;
  
}

/* called by users to see if any receives messages are available */
void ring_cpu_poll(struct RingState *s)
{
  /* volatile */ struct RingMessage *msg = &(s->recvbuf[s->next_receive]);
  if (msg->header != MSG_IDLE) {
    s->peer_next_receive = msg->next_receive;
    ring_process_message(s, (struct RingMessage *) msg);
    msg->header = MSG_IDLE;
    s->next_receive = (s->next_receive + 1) % N;
    s->peer_next_receive_sent = s->next_receive;
    if (ring_internal_send_nop_p(s)) send_nop(s);
  }
}

void ring_cpu_send(struct RingState *s, int type, int length, void *data)
{
  if (DEBUG) printf("send %s next %d\n", s->name, s->next_send);
  do {
    ring_internal_cpu_receive(s);
  } while (ring_send_space_available(s) == 0);
  struct RingMessage msg;  // local composition of message
  msg.header = type;
  msg.next_receive = s->next_receive;
  assert (length <= MaxData);
  memcpy(&msg.data, data, length);  // local copy
  void *mp = (void *) &(s->sendbuf[s->next_send]);
  __m512i temp = _mm512_load_epi32((void *) &msg);
  _mm512_store_si512(mp, temp);
  //  _movdir64b(&(s->sendbuf[s->next_send]), &msg);   // send message (atomic!)
  s->next_send = (s->next_send + 1) % N;
}

void ring_cpu_drain(struct RingState *s)
{
  CHERE(s->name);
  while (s->total_receive < glob_count) ring_cpu_poll(s);
}

#define cpu_relax() asm volatile("rep; nop")
#define nullptr NULL

void *GPUThread(void *arg)
{  // cpu code
  struct RingState *s = (struct RingState *) arg;
  uint64_t msgdata[7] = {0xfeedface, 0, 0, 0, 0, 0, 0};
  std::cout << "thread starting" << std::endl;
  for (int i = 0; i < glob_count; i += 1) {
    msgdata[1] = i;
    ring_cpu_send(s, MSG_NOP, glob_size, msgdata);
  }
  ring_cpu_drain(s);
  return(NULL);
}

int main(int argc, char *argv[]) {
  ProcessArgs(argc, argv);
  uint64_t loc_count = glob_count;
  uint64_t loc_size = glob_size;
  int loc_validate = glob_validate;
  int loc_use_atomic_load = glob_use_atomic_load;
  int loc_use_atomic_store = glob_use_atomic_store;


  // allocate device memory

  struct RingMessage *device_data_mem = (struct RingMessage *) aligned_alloc(4096, BUFSIZE * 2);

  //allocate host mamory
  struct RingMessage *host_data_mem = (struct RingMessage *) aligned_alloc(4096, BUFSIZE);
  std::cout << " host_data_mem " << host_data_mem << std::endl;


  struct RingMessage *device_data_hostmap;   // pointer for host to use
  device_data_hostmap = device_data_mem;

  // initialize host mamory
  memset(host_data_mem, 0, BUFSIZE);

  // initialize device memory
  memset(device_data_mem, 0, BUFSIZE);


  struct RingState *cpu = (struct RingState *) aligned_alloc(4096, sizeof(struct RingState));
  struct RingState *gpu = (struct RingState *) aligned_alloc(4096, sizeof(struct RingState));

  /* initialize state records */
  memset(cpu, 0, sizeof(struct RingState));
// initialize device memory
  memset(gpu, 0, sizeof(struct RingState));

  initstate(cpu, device_data_hostmap, host_data_mem, "cpu");
  initstate(gpu, host_data_mem, device_data_hostmap, "gpu");
  
  

  std::cout << "kernel going to launch" << std::endl;
  unsigned long start_time, end_time;
  struct timespec ts_start, ts_end;


  pthread_t gputhread;
  pthread_attr_t pt_attributes;
  pthread_attr_init(&pt_attributes);

      clock_gettime(CLOCK_REALTIME, &ts_start);
      start_time = rdtsc();

  pthread_create(&gputhread, &pt_attributes, GPUThread, (void *) gpu);

  GPUThread(cpu);
    
  pthread_join(gputhread, NULL);
      clock_gettime(CLOCK_REALTIME, &ts_end);
      end_time = rdtsc();

    /* common cleanup */
    double elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	((double) (ts_end.tv_nsec - ts_start.tv_nsec));

    std::cout << argv[0];
    std::cout << "--count " << loc_count << " ";
    std::cout << "--size " << loc_size << " ";

    double size = loc_size;
    double mbps = (size * 1000) / (elapsed / ((double) loc_count));
    double nsec = elapsed / ((double) loc_count);
    std::cout << " nsec each " << nsec << " MB/s " << mbps << std::endl;

    return 0;
}

