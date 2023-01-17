#ifndef RINGLIB3_CPP
#define RINGLIB3_CPP

#include "ringlib3.h"

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
}


//============== GPU Versions

GPURing::GPURing(int id, struct RingMessage *sendbuf, struct RingMessage *recvbuf) : atomic_next_send(next_send), atomic_next_track(next_track), atomic_receive_count(receive_count)
{
  this->ringid = id;
  this->next_send = RingN;
  this->sendbuf = sendbuf;
  this->recvbuf = recvbuf;
  this->receive_count = 0;
  // this should be in the GPUTrack constructor, somehow
  this->groups[0].atomic_credit.store(RingN);
  for (int g = 1; g < GroupN; g += 1) {
    // ending count for each group one RingN ago
    this->groups[g].atomic_credit.store(MsgsPerGroup + (g * MsgsPerGroup));
  }
  for (int t = 0; t < TrackN; t += 1) {
    this->track[t].next_receive = RingN + t;  // in place of next_receive
  }
  GPU_STORE_PEER_NEXT_RECV(RingN);
}

void GPURing::ProcessMessage(struct RingMessage *msg)
{
  int msgtype = msg->header;
  /* do what the message says */
  atomic_receive_count += 1;
}


void GPURing::Drain()
{
  while (Poll(1));
}

/* does minimal work by waiting for and discarding messages up to seq */
void GPURing::Discard(unsigned seq)
{

}



/* Principles of Operation
 * entering thread assigned a track and locks it.
 * a track is receive buffer slots % TrackN = i
 *
 * 
 */

/* notes
 * after the group is done and credited, the track is free to move ahead
 * 
 * The carry in works as long as it  we don't confuse credit messages
 * 
 * instead, make carry-in be a high bit (above RingN) to disambiguate rounds, and don't reset them
 *
 * let's make the group credit value exactly be the peer_receive value we need, so it should be
 * equal to my_next_receive
 *
 * each group gets individual counts as messages in the current group and track complets, plus
 * a carry in equal to the receive sequence number of the first message in the group
 * that carry-in is only applied when the previous group finishes
 *
 * compute my_group = group_of(my_slot)
 * next_group_base = roundup(my_slot)
 * add_in = 1;
 * for (;;) {
 * if (groups[my_group].atomic_credit.fetch_add(addin) == (next_group_base-1)) {
     GPU_STORE_PEER_NEXT_RECV(next_group_base);
 *   my_group = (my_group + 1) % GroupN
 *   add_in = next_group_base
 *   next_group_base += MsgsPerGroup
 * else break;
 */
// receive at most nmsg
int GPURing::Poll(int nmsg)   
{
  int res = 0;  // default we did not receive a message
  unsigned my_track = atomic_next_track.fetch_add(1) % TrackN;  // round robin assign threads to tracks
  unsigned lockwasbusy = track[my_track].atomic_lock.exchange(1); // try track lock
  union RingMessages rm;
  if (lockwasbusy == 0) {  // we got the lock
    unsigned my_slot = track[my_track].next_receive;  // current receive slot for this track
    struct RingMessage *msgp = &recvbuf[my_slot % RingN]; // compute msg ptr for my_slot
    // fence may not be needed, because the load is uncached?
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    rm.data = ucl_ulong8((ulong8 *) msgp);  // load tentative message into registers
    if (rm.msg.sequence == my_slot) {   // has the expected message actually arrived?
      res += 1;  // eventually return that we did receive a message
      ProcessMessage(&rm.msg);  // deal with received message
      track[my_track].next_receive = my_slot + TrackN;   // compute next expected message slot
      unsigned my_group = groupof(my_slot);  // the message we got is in my_group
      /* perhaps reduce contention on atomic_credit_group by first acccumulating
	 in the track counter */
      unsigned next_group_base = roundup(my_slot);  // sequence number of beginning of the <next> group
      unsigned add_in = 1;  // at first, we count only the message received
      for (;;) {
	// a group is finished when its count is the sequence number of the base of the <next> group
	if (groups[my_group].atomic_credit.fetch_add(add_in) == (next_group_base - add_in)) {
	  // inform sender of progress once per group
	  GPU_STORE_PEER_NEXT_RECV(next_group_base);  // could do this only at the end of the loop
	  my_group = (my_group + 1) % GroupN; // set up to carry-in to the following group
	  add_in = RingN - MsgsPerGroup;  // the amount to carry to next group is the size of the Ring less the size of the group
	  next_group_base += MsgsPerGroup;  // to be used if the next group is finished
	} else {
	  break;  // This message does not end the group
	}
      }
    }
    // fence not needed because atomic_lock is marked sequential consistency
    // sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    track[my_track].atomic_lock.store(0); // release track lock
  } // else lockwasbusy!=0, the track was already locked
  return (res);  // returns number of messages received
}

// count must be less than RingN!  Should we check?
void GPURing::Send(struct RingMessage *msgp, int count)
{
  unsigned my_send_index = atomic_next_send.fetch_add(count);   // allocate slots
  // wait for previous uses of the buffer to be complete
  while (((my_send_index+count) - LOAD_PEER_NEXT_RECV()) > RingN) {
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    //Poll(1); different ring object!
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
  unsigned my_send_index = atomic_next_send.fetch_add(1);   // allocate slot
  struct RingMessage *mp = &(sendbuf[my_send_index % RingN]);  // ring ptr
  union RingMessages rm;
  rm.msg = *msgp;
  rm.msg.sequence = my_send_index;
  // wait for previous uses of the buffer to be complete
  while ((my_send_index - LOAD_PEER_NEXT_RECV()) > RingN) {
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    // Poll(1); different ring object
  }
  ucs_ulong8((ulong8 *) mp, rm.data); // could be OOO
  // is this fence actually needed?
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

#if 0
void GPURing::Send(RingMessage *msgp)
{
  unsigned my_send_index = atomic_next_send.fetch_add(1);   // allocate slot
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
  std::cout << "  ringid " << ringid << std::endl;
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
  unsigned lockwasbusy = atomic_receive_lock.exchange(1);
  int res = 0;
  if (lockwasbusy == 0) {
  struct RingMessage *msg = &recvbuf[next_receive % RingN]; // msg ptr
    unsigned sequence = msg->sequence;
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
  unsigned my_send_index = atomic_next_send.fetch_add(1);
  msg->sequence = my_send_index;
  struct RingMessage *mp = &(sendbuf[my_send_index % RingN]);
  while ((my_send_index - LOAD_PEER_NEXT_RECV()) > (RingN - 10)) {
    //send_wait_count += 1;
    //Poll();  // deadlock prevention
    cpu_relax();
  }
  _movdir64b(mp, msg); //memcpy(mp, msg, sizeof(RingMessage));
}

void CPURing::ProcessMessage(struct RingMessage *msg)
{
  int msgtype = msg->header;
  /* do what the message says */
}


void CPURing::Drain()
{
  while (Poll());
}

/* does minimal work by waiting for and discarding messages up to seq */
void CPURing::Discard(unsigned seq)
{
  unsigned current = next_receive;
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
