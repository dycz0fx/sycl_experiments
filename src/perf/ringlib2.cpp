#ifndef RINGLIB_CPP
#define RINGLIB_CPP

#include "ringlib2.h"
#include "uncached.cpp"

#define DEBUG 0

/* 
 * s->next_receive is the next message we haven't received yet
 *
 * The number of slots in the ring buffer should be enough to cover about 2 x
 * the round trip latency so that we can return credits before the peer runs out
 * when all the traffic is one-way
 */


//==============  Common version

void Ring::Print()
{
  std::cout << "  send buf " << sendbuf << std::endl;
  std::cout << "  recv buf " << recvbuf << std::endl;
  std::cout << "  next receive " << next_receive << std::endl;
  //std::cout << "  peer next receive " << peer_next_receive << std::endl;
  std::cout << "  wait_in_send " << wait_in_send << std::endl;
  std::cout << "  wait_in_drain " << wait_in_drain << std::endl;
  std::cout << "  send_wait_count " << send_wait_count << std::endl;
}


//============== GPU Versions

GPURing::GPURing(struct RingMessage *sendbuf, struct RingMessage *recvbuf) : atomic_next_send(next_send), atomic_receive_lock(receive_lock)
{
  this->next_send = RingN;
  this->next_receive = RingN;
  this->sendbuf = sendbuf;
  this->recvbuf = recvbuf;
  GPU_STORE_PEER_NEXT_RECV(RingN);
#if TRACE == 1
  this->wait_in_send = 0;
  this->wait_in_drain = 0;
#endif
}

void GPURing::ProcessMessage(struct RingMessage *msg)
{
  int msgtype = msg->header;
  /* do what the message says */
}


void GPURing::Drain()
{
  #if TRACE == 1
  wait_in_drain = 5;
  #endif
  while (Poll());
  #if TRACE == 1
  wait_in_drain = 0;
  #endif
}

/* called by ring_send when we need space to send a message */
int GPURing::Poll()
{
  int32_t lockwasbusy = atomic_receive_lock.exchange(1);
  int res = 0;
  if (lockwasbusy == 0) {
    struct RingMessage *msgp = &recvbuf[next_receive % RingN]; // msg ptr
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    union RingMessages rm;
    rm.data = ucl_ulong8((ulong8 *) msgp);
    if (rm.msg.sequence == next_receive) {
      /* release lock ASAP */
      ProcessMessage(&rm.msg);  // we're not holding lock here!
      res = 1;
      next_receive = next_receive + 1;
      if ((next_receive & 0xff) == 0) {
	GPU_STORE_PEER_NEXT_RECV(next_receive);  // could happen OOO
      }
    }
    atomic_receive_lock.store(0);  // release lock
  }
  // lock was already taken if we get here, so just return
  return(res);
}


// on entry type and data are filled in
// instead of saddling "send" with updating the credits, just let send_nop
// do it 4 times per ring cycle
// the poll thread has to be there anyway in case there isn't any send activity

void GPURing::Send(struct RingMessage *msg)
{
  int32_t my_send_index = atomic_next_send.fetch_add(1);   // allocate slot
  struct RingMessage *mp = &(sendbuf[my_send_index % RingN]);  // ring ptr
  // wait for previous uses of the buffer to be complete
  msg->sequence = my_send_index;
  #if TRACE == 1
  wait_in_send = 117;
  #endif
  while ((my_send_index - LOAD_PEER_NEXT_RECV()) > (RingN - 10)) {
    send_wait_count += 1;
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
  }
    
  #if TRACE == 1
  wait_in_send = 0;
  #endif
  ucs_ulong8((ulong8 *) mp, *((ulong8 *) msg)); // could be OOO
  // is this fence actually needed?
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

void GPURing::Print(const char *name)
{
  std::cout << "Name " << name << std::endl;
  Ring::Print();
  std::cout << "  next send " << next_send << std::endl;
}

//==============  CPU version
void CPURing::InitState(struct RingMessage *sendbuf, struct RingMessage *recvbuf)
{
  this->atomic_next_send = RingN;
  this->next_receive = RingN;
  this->sendbuf = sendbuf;
  this->recvbuf = recvbuf;
  STORE_PEER_NEXT_RECV(RingN);
}

/* look at the next expected arriving message sequence and if it
 * has the expected sequence number in it, process the messageg
 * and then increment the ring pointer
 */
/* ideas 
 * copy message from ring and release lock, like GPU
 * maintain two next_recv, the leader is used to allocate receive slots
 * and the trailer to update peer_next_receive, so multiple processing threads
 * can work in-place
 */

int CPURing::Poll()
{
  int32_t lockwasbusy = atomic_receive_lock.exchange(1);
  int res = 0;
  if (lockwasbusy == 0) {
  struct RingMessage *msg = &recvbuf[next_receive % RingN]; // msg ptr
    int32_t sequence = msg->sequence;
    if (sequence == next_receive) {
      ProcessMessage(msg);
      res = 1;
      next_receive = next_receive + 1;
      if ((next_receive & 0xff) == 0) STORE_PEER_NEXT_RECV(next_receive);
    }
    atomic_receive_lock.store(0);  // release lock
  }
  return (res);
}

#define cpu_relax() asm volatile("rep; nop")

/* send a noop message
 * first wait for credit to send anything
 * then build a nop message
 * and then copy it to the destination ring
 * and then increment the transmit counter and transmit ring pointer
 * (for multithread, the incremnting of the ring pointer must be locked)
 */
void CPURing::Send(RingMessage *msg)
{
  int32_t my_send_index = atomic_next_send.fetch_add(1);
  msg->sequence = my_send_index;
  struct RingMessage *mp = &(sendbuf[my_send_index % RingN]);
  #if TRACE == 1
  wait_in_send = 117;
  #endif
  while ((my_send_index - LOAD_PEER_NEXT_RECV()) > (RingN - 10)) {
    send_wait_count += 1;
    cpu_relax();
  }
  #if TRACE == 1
  wait_in_send = 0;
  #endif
  _movdir64b(mp, msg); //memcpy(mp, msg, sizeof(RingMessage));
}

void CPURing::ProcessMessage(struct RingMessage *msg)
{
  int msgtype = msg->header;
  /* do what the message says */
}


void CPURing::Drain()
{
  #if TRACE == 1
  wait_in_drain = 5;
  #endif
  while (Poll());
  #if TRACE == 1
  wait_in_drain = 0;
  #endif
}

void CPURing::Print(const char *name)
{
  std::cout << "Name " << name << std::endl;
  Ring::Print();
  std::cout << "  next send " << atomic_next_send << std::endl;
}

#endif //! ifndef RINGLIB_CPP
