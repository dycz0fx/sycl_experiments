#ifndef RINGLIB3_CPP
#define RINGLIB3_CPP

#include "ringlib3.h"
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
  //std::cout << "  peer next receive " << peer_next_receive << std::endl;
  std::cout << "  wait_in_send " << wait_in_send << std::endl;
  std::cout << "  wait_in_drain " << wait_in_drain << std::endl;
  std::cout << "  send_wait_count " << send_wait_count << std::endl;
}


//============== GPU Versions

GPURing::GPURing(struct RingMessage *sendbuf, struct RingMessage *recvbuf) : atomic_next_send(next_send), atomic_next_track(next_track), atomic_receive_count(receive_count)
{
  this->next_send = RingN;
  this->sendbuf = sendbuf;
  this->recvbuf = recvbuf;
  this->receive_count = 0;
  this->groups[0].atomic_credit.fetch_add(1); // initial carry in
  // this should be in the GPUTrack constructor, somehow
  for (int i = 0; i < TrackN; i += 1) {
    this->track[i].next_receive = RingN + i;  // in place of next_receive
  }
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
  atomic_receive_count += 1;
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

/* does minimal work by waiting for and discarding messages up to seq */
void GPURing::Discard(int32_t seq)
{

}



/* Principles of Operation
 * entering thread assigned a track and locks it.
 * a track is receive buffer slots % TrackN = i
 *
 * 
 */
int GPURing::Poll()
{
  int res = 0;
  int32_t my_track = atomic_next_track.fetch_add(1) % TrackN;

  int32_t lockwasbusy = track[my_track].atomic_lock.exchange(1); // try track lock
  if (lockwasbusy == 0) {
    int32_t my_slot = track[my_track].atomic_next_receive.load();
    struct RingMessage *msgp = &recvbuf[my_slot % RingN]; // msg ptr
    union RingMessages rm;
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    rm.data = ucl_ulong8((ulong8 *) msgp);
    if (rm.msg.sequence == my_slot) {
      ProcessMessage(&rm.msg);
      res = 1;
      track[my_track].atomic_next_receive += TrackN;   // may be picked up by another thread next
      int32_t my_group = groupof(my_slot);
      /* note reduce contention on atomic_credit_group by first acccumulating
	 in the track counter */
      int32_t my_next_receive = roundup(my_slot);
      for (;;) {
	if (groups[my_group].atomic_credit.fetch_add(1) == MsgsPerGroup) {
	  groups[my_group].atomic_credit.fetch_add(-(MsgsPerGroup+1));
	  /* my_next_receive is the slot after the end of my_group */
	  //GPU_STORE_PEER_NEXT_RECV(my_next_receive);  // could do this only at the end of the loop
	  // reread Pardo message about goiing back in time (or add fence here)
	  my_group = (my_group + 1) % GroupN;
	  my_next_receive += MsgsPerGroup;
	} else {
	  // move GPU_STORE_NEXT_RECV here to minimize time reversal opportunities
	  GPU_STORE_PEER_NEXT_RECV(my_next_receive - MsgsPerGroup);
	  break;
	}
      }
    }
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system); // force update of next_recv
    track[my_track].atomic_lock.store(0); // release track lock
  }
  return (res);
}

// count must be less than RingN!  Should we check?
void GPURing::Send(struct RingMessage *msgp, int count)
{
  int32_t my_send_index = atomic_next_send.fetch_add(count);   // allocate slots
  // wait for previous uses of the buffer to be complete
  while (((my_send_index+count) - LOAD_PEER_NEXT_RECV()) > RingN) {
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
  }
  while (count--) {
    union RingMessages rm;
    struct RingMessage *mp;
    mp = &(sendbuf[my_send_index % RingN]);  // ring ptr
    rm.msg = *msgp++;
    rm.msg.sequence = my_send_index;
    ucs_ulong8((ulong8 *) mp, rm.data); // could be OOO
    my_send_index += 1;
  }
  // is this fence actually needed?
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

void GPURing::Send(RingMessage *msgp)
{
  int32_t my_send_index = atomic_next_send.fetch_add(1);   // allocate slot
  struct RingMessage *mp = &(sendbuf[my_send_index % RingN]);  // ring ptr
  union RingMessages rm;
  rm.msg = *msgp;
  rm.msg.sequence = my_send_index;
  // wait for previous uses of the buffer to be complete
  while ((my_send_index - LOAD_PEER_NEXT_RECV()) > RingN) {
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
  }
  ucs_ulong8((ulong8 *) mp, rm.data); // could be OOO
  // is this fence actually needed?
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

#if 0
void GPURing::Send(RingMessage *msgp)
{
  int32_t my_send_index = atomic_next_send.fetch_add(1);   // allocate slot
  struct RingMessage *mp = &(sendbuf[my_send_index % RingN]);  // ring ptr
  msgp->sequence = my_send_index;
  // wait for previous uses of the buffer to be complete
  while ((my_send_index - LOAD_PEER_NEXT_RECV()) > RingN) {
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
  }
  ucs_ulong8((ulong8 *) mp, *((ulong8 *) msgp)); // could be OOO
  // is this fence actually needed?
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}
#endif

void GPURing::Print(const char *name)
{
  std::cout << "Name " << name << std::endl;
  std::cout << "  receive_count " << receive_count << std::endl;
  Ring::Print();
  std::cout << "  next send " << next_send << std::endl;
  for (int i = 0; i < GroupN; i += 1) {
    std::cout << "  groups[" << i << "].credit = " << groups[i].credit << std::endl;
  }
  std::cout << "  next_track " << next_track << std::endl;
  for (int i = 0; i < TrackN; i += 1) {
    std::cout << "  track[" << i << "].next_receive = " << track[i].next_receive << std::endl;
  }
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
    //send_wait_count += 1;
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

/* does minimal work by waiting for and discarding messages up to seq */
void CPURing::Discard(int32_t seq)
{
  int32_t current = next_receive;
  union RingMessages rm;
  struct RingMessage *mp;
  do {
    do {
      mp = &(recvbuf[current % RingN]);  // ring ptr
    } while (mp->sequence != current);
    STORE_PEER_NEXT_RECV(current);
    current += 0x100;
  } while (current < seq);
  current = seq;
  do {
    mp = &(recvbuf[current % RingN]);  // ring ptr
  } while (mp->sequence != current);
  STORE_PEER_NEXT_RECV(current);
}


void CPURing::Print(const char *name)
{
  std::cout << "Name " << name << std::endl;
  Ring::Print();
  std::cout << "  next send " << atomic_next_send << std::endl;
  std::cout << "  next receive " << next_receive << std::endl;
}

#endif //! ifndef RINGLIB_CPP
