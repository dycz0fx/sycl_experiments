#ifndef RINGLIB_CPP
#define RINGLIB_CPP

#include "ringlib2.h"
#include "uncached.cpp"

#define DEBUG 0

/* how many credits can we return? 
 * s->next_receive is the next message we haven't received yet
 * s->peer_next_receive_sent is the last next_receive we told the peer about
 * s->next_receive - s->peer_next_receive_sent is the number of messages the peer can
 * send but we haven't told them yet.
 * The idea is that if this number gets larger than N/4 we should send an
 * otherwise empty message to update the peer
 *
 * The number of slots in the ring buffer should be enough to cover about 2 x
 * the round trip latency so that we can return credits before the peer runs out
 * when all the traffic is one-way
 */


//==============  Common version

void Ring::Print()
{
  std::cout << "  total_sent MSG_IDLE " << total_sent[MSG_IDLE] << std::endl;
  std::cout << "  total_sent MSG_NOP " << total_sent[MSG_NOP] << std::endl;
  std::cout << "  total_sent MSG_PUT " << total_sent[MSG_PUT] << std::endl;
  std::cout << "  total_sent MSG_GET " << total_sent[MSG_GET] << std::endl;
  std::cout << "  total_received MSG_IDLE " << total_received[MSG_IDLE] << std::endl;
  std::cout << "  total_received MSG_NOP " << total_received[MSG_NOP] << std::endl;
  std::cout << "  total_received MSG_PUT " << total_received[MSG_PUT] << std::endl;
  std::cout << "  total_received MSG_GET " << total_received[MSG_GET] << std::endl;

  std::cout << "  send buf " << sendbuf << std::endl;
  std::cout << "  recv buf " << recvbuf << std::endl;
  std::cout << "  next receive " << next_receive << std::endl;
  std::cout << "  peer next receive " << peer_next_receive << std::endl;
  std::cout << "  peer next receive sent " << peer_next_receive_sent << std::endl;
  std::cout << "  send count " << send_count << std::endl;
  std::cout << "  recv count " << recv_count << std::endl;
  std::cout << "  wait_in_send_nop " << wait_in_send_nop << std::endl;
  std::cout << "  wait_in_send " << wait_in_send << std::endl;
  std::cout << "  wait_in_drain " << wait_in_drain << std::endl;
}


//============== GPU Versions

void GPURing::InitState(struct RingMessage *sendbuf, struct RingMessage *recvbuf, int send_count, int recv_count)
{
  next_send = 0;
  next_receive = 0;
  peer_next_receive = 0;
  peer_next_receive_sent = 0;
  sendbuf = sendbuf;
  recvbuf = recvbuf;
  send_count = send_count;
  recv_count = recv_count;
#if TRACE == 1
  for (int i = 0; i < NUM_MESSAGE_TYPES; i += 1) {
    total_sent[i] = 0;
    total_received[i] = 0;
  }
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
  while (total_received[MSG_PUT] < recv_count) Poll();
  #if TRACE == 1
  wait_in_drain = 0;
  #endif
}

/* called by ring_send when we need space to send a message */
void GPURing::InternalReceive()
{
  int32_t lockwasfree = atomic_receive_lock.exchange(1);
  if (lockwasfree == 0) {
    /* volatile */ struct RingMessage *msg = &(recvbuf[next_receive % RingN]);
    int32_t msgtype = msg->header;
    if (msgtype != MSG_IDLE) {
      if ((msgtype < 0) || (msgtype >= NUM_MESSAGE_TYPES)) msgtype = 0;
#if TRACE == 1
      total_received[msgtype] += 1;
#endif
      if (msg->header == MSG_NOP) {
	peer_next_receive = msg->next_receive;
      } else {
	ProcessMessage(msg);
      }
      msg->header = MSG_IDLE;
      next_receive = next_receive + 1;
    }
    atomic_receive_lock.store(0);  // release lock
  }
}


// on entry type and data are filled in
// instead of saddling "send" with updating the credits, just let send_nop
// do it 4 times per ring cycle
// the poll thread has to be there anyway in case there isn't any send activity

void GPURing::Send(struct RingMessage *msg)
{

  ulong8 *umsg = (ulong8 *) msg;
  // allocate a slot:
  int32_t my_send_index = atomic_next_send.fetch_add(1);

  // calculate message pointer
  struct RingMessage *mp = &(sendbuf[my_send_index % RingN]);
  // wait for previous uses of the buffer to be complete
  #if TRACE == 1
  wait_in_send = 117;
  #endif
  while (my_send_index - peer_next_receive > (RingN - 1)) {
    InternalReceive();
  }
  #if TRACE == 1
  wait_in_send = 0;
  #endif

  /* these could send out of order, is that OK? */
  ucs_ulong8((ulong8 *) mp, *umsg);
  // is this fence actually needed?
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
  // check for array bounds !
  #if TRACE
  total_sent[msg->header] += 1;  // XXX should be atomic if used at all
  #endif
}

void GPURing::Print(const char *name)
{
  std::cout << "Name " << name << std::endl;
  Ring::Print();
  std::cout << "  next send " << next_send << std::endl;
}

//==============  CPU version
void CPURing::InitState(struct RingMessage *sbuf, struct RingMessage *rbuf, int scount, int rcount)
{
  atomic_next_send = 0;
  next_receive = 0;
  peer_next_receive = 0;
  peer_next_receive_sent = 0;
  sendbuf = sbuf;
  recvbuf = rbuf;
  send_count = scount;
  recv_count = rcount;
#if TRACE == 1
  for (int i = 0; i < NUM_MESSAGE_TYPES; i += 1) {
    total_sent[i] = 0;
    total_received[i] = 0;
  }
#endif
}

/* called by ring_send when we need space to send a message */
/* look at the next expected arriving message header and if it
 * has something in it, process the message and then clear the flag
 * and then increment the ring pointer
 */
void CPURing::InternalReceive()
{
  /* volatile */ struct RingMessage *msg = &(recvbuf[next_receive % RingN]);
  int32_t msgtype = msg->header;
  if (msgtype != MSG_IDLE) {
    if ((msgtype < 0) || (msgtype >= NUM_MESSAGE_TYPES)) msgtype = 0;
#if TRACE == 1
    total_received[msgtype] += 1;
#endif
    if (msg->header == MSG_NOP) {
      peer_next_receive = msg->next_receive;
    } else {
      ProcessMessage(msg);
    }
    msg->header = MSG_IDLE;
    next_receive = next_receive + 1;
  }
}

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

  #if TRACE == 1
  wait_in_send = 117;
  #endif
  while (my_send_index - peer_next_receive > (RingN - 1)) {
    InternalReceive();
  }
  #if TRACE == 1
  wait_in_send = 0;
  #endif
  struct RingMessage *mp = &(sendbuf[my_send_index % RingN]);
  _movdir64b(mp, &msg);
  #if TRACE == 1
  total_sent[msg->header] += 1;
  #endif
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
  while (total_received[MSG_PUT] < recv_count) Poll();
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
