#ifndef RINGLIB_H
#define RINGLIB_H

#define TRACE 1

constexpr int RingN = 1024;
constexpr int NOP_THRESHOLD = 128;   /* N/4 ? See analysis */

/* message types, all non-zero Non-zero acts as the ready flag*/
#define MSG_IDLE 0
#define MSG_NOP 1
#define MSG_PUT 2
#define MSG_GET 3
#define NUM_MESSAGE_TYPES 4

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
  int32_t total_sent[NUM_MESSAGE_TYPES];     // for accounting
  int32_t total_received[NUM_MESSAGE_TYPES];     // for accounting
  int32_t send_count;        // from command line
  int32_t recv_count;
  int32_t wait_in_send_nop;
  int32_t wait_in_send;
  int32_t wait_in_drain;
  const char *name;
};

#endif //! ifndef RINGLIB_CPP
