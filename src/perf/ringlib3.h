#ifndef RINGLIB3_H
#define RINGLIB3_H
#import <atomic>

#define TRACE 0

constexpr int RingN = 2048;
constexpr int GroupN = 8;
constexpr int MsgsPerGroup = RingN / GroupN;  // 128
constexpr int TrackN = 32;
constexpr int GroupMsgPerTrack = MsgsPerGroup / TrackN; //16

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

/* Principles of operation
 *
 * This is a ring buffer for situations with asymmetric memory access
 * such as across a PCIe bus.  It is assumed that Stores are faster than
 * Loads, (because they are fire and forget), so only stores are used for remote * operations.  Local operations can be cached and both loads and stores are
 * fast.  Atomic operations are only supported on local locations.
 *
 * Messages are transmitted using a single vector store instruction
 * of 64 bytes.  The sequence field of each message is reserved
 * Each end maintains a write counter for its outgoing traffic
 * and read counter for its incoming traffic
 *
 * Each end also has a delayed view of the other end's read counter, called
 * peer_next_receive
 *
 * An end can transmit if (write_counter - peer_next_receive) < (RingN - 1)
 *
 * There is a dedicated location on the receive side where each end 
 * occasionally stores its next_read counter
 *
 * The read and write counters wrap at 32 bits, but the ring itself is only
 * RingN, which must be a power of two
 *
 * Multiple senders work by 
 * 1) atomic allocation of a future send-buffer by fetch_add(1) to their own
 *    send counter
 * Allocations may be far in the future, so the sender must wait until the
 * allocated slot is actually free, which occurs when peer_next_receive has
 * caught up within RingN-1
 *
 * 2) wait until
 *    (write_counter - peer_next_receive) < (RingN - 1)
 * while waiting, the sending thread can assist in receiving messages
 *
 * 3) send the message (single MOVDIR64B on the host or ucs_ulong8 on the GPU
 *
 * Receiving works by 
 * 1) polling the next expected receive slot for the expected sequence number
 * 2) handling the message 
 * 3) incrementing the next_receive counter
 * 4) occasionally storing local next_receive to remote peer_next_receive
 *
 * For thread safety on receive, only one thread can be in the 1-4 critical 
 * section (for now)
 *
 * an atomic exchange is used to seize the receive lock.
 * If the lock is already busy,the thread attempting to call Poll can do 
 * other work instead.
 */

/* when peer_next_read and next_write pointers are equal, the buffer is empty.
 * when next_write pointer is one less than peer_next_read pointer, 
 * the buffer is full
 *
 */

/* Handling of next_receive and peer_next_receive
 * next_receive is a local variable for the receiver
 * peer_next_receive is a local varaible for the sender
 * The <receiver> occasionally stores (remote) peer_next_receive from
 *    (local) next_receive.  This is a bus store
 * The peer_next_receive cell must be remotely-storable so we put it at the
 * end of the send buffer (which is located at the receiver)
 * This is an unfortunate mixing of send and receive associated storage but
 * it reduces the number of segments that must be remotely accessible
 *
 * The <sender> occasionally copies the remotely updated peer_next_receive to a  * local cached copy in the Ring structure
 *
 * Receiver locations
 *   next_receive
 *   sendbuf[RingN].next_receive  (A)
 * Sender locations
 *   recvbuf[RingN].next_receive  (same as A)
 *   peer_next_receive
 *
 * while the sender is waiting for free space in sendbuf, the sender polls
 * recvbuf[RingN].next_receive and copies it to peer_next_receive
 * This is how the sending end learns that the receiver has caught up
 */
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
  // accounting and debug from here down
  int32_t wait_in_send;
  int32_t wait_in_drain;
  int64_t send_wait_count;
  // functions
  void Print();

};
// GPU uses sycl::atomic_ref rather than std::atomic
class GPURing : public Ring {
 public:
  int32_t receive_count;
  int32_t next_send;        // next slot in sendbuf
  int32_t credit_groups[GroupN]; // atomic, set up at point of use
  int32_t next_track;
  int32_t track_lock[TrackN];  // atomic, set up at point of use
  int32_t track_next_receive[TrackN]; // used within track_lock so not atomic
  // ordering may be excessive
  sycl::atomic_ref<int32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> atomic_receive_count;
  
  sycl::atomic_ref<int32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> atomic_next_send;
  
  sycl::atomic_ref<int32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> atomic_next_track;
  
  // must be called on the GPU!
  GPURing(struct RingMessage *sendbuf, struct RingMessage *recvbuf);
    
  void Print(const char *name);  // memory may not be addressible
  void Send(struct RingMessage *msg);
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
