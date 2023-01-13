#ifndef RINGLIB4_H
#define RINGLIB4_H
#import <atomic>

#define TRACE 0

constexpr int RingN = 1024;
constexpr int GroupN = 8;
constexpr int MsgsPerGroup = RingN / GroupN;  // 128

int groupof(int sequence)
{
  return ((sequence / MsgsPerGroup) % GroupN);
}

int roundup(int sequence)
{
  return ((sequence + MsgsPerGroup) & (~(MsgsPerGroup - 1)));
}

/* message types, all non-zero Non-zero acts as the ready flag*/
#define MSG_IDLE 0
#define MSG_NOP 1
#define MSG_PUT 2
#define MSG_GET 3
#define NUM_MESSAGE_TYPES 4

/* should the length be implicit or explicit? */
struct RingMessage {
  int32_t sequence;
  int32_t header;
  uint64_t data[7];
};

/* Possibly a union of different message types */
union RingMessages {
  ulong8 data;
  struct RingMessage msg;
};
// stores only
#define LOAD_PEER_NEXT_RECV() (recvbuf[RingN].sequence)
// loads only
#define STORE_PEER_NEXT_RECV(x) (sendbuf[RingN].sequence = (x))
#define GPU_STORE_PEER_NEXT_RECV(x) (ucs_uint((uint *) &sendbuf[RingN].sequence, (uint) x))


/* each end of the link has a Ring record 
 */

class Ring {
 public:
  struct RingMessage *sendbuf;   // remote buffer
  struct RingMessage *recvbuf;   // local buffer
  // functions
  void Print();

};


// GPU uses sycl::atomic_ref rather than std::atomic
class GPURing : public Ring {
 public:
  int32_t receive_count;
  int32_t next_send;        // next slot in sendbuf
  int32_t credit_groups[GroupN]; // atomic, set up at point of use
  int32_t next_receive;
  
  // ordering may be excessive
  sycl::atomic_ref<int32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> atomic_receive_count;
  
  sycl::atomic_ref<int32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> atomic_next_send;
  
  sycl::atomic_ref<int32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> atomic_next_receive;
  
  // must be called on the GPU!
  GPURing(struct RingMessage *sendbuf, struct RingMessage *recvbuf);
    
  void Print(const char *name);  // memory may not be addressible
  void Send(struct RingMessage *msgp, int count);
  void Send(struct RingMessage *msgp);
  int Poll();
  void Discard(int32_t seq);
  void Drain();  // call poll until there are no messages immediately available
 private:
  void ProcessMessage(struct RingMessage *msg);
};

class CPURing : public Ring {
 public:
  int32_t next_receive;     // next slot in recvbuf
  std::atomic<int32_t> atomic_next_send;
  std::atomic<int32_t> atomic_receive_lock;
  void InitState(struct RingMessage *sendbuf, struct RingMessage *recvbuf);

  void Print(const char *name);
  void Send(struct RingMessage *msg);
  int Poll();
  void Discard(int32_t seq);
  void Drain();
 private:
  void ProcessMessage(struct RingMessage *msg);
};



#endif //! ifndef RINGLIB_H
