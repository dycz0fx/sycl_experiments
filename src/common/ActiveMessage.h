#ifndef ACTIVEMESSAGE_H
#define ACTIVEMESSAGE_H

#define _GNU_SOURCE             /* See feature_test_macros(7) */

#include <stdarg.h>
#include <pthread.h>
#include <stdint.h>


#define CACHELINE 64


/* command = 0 means wait
 * command = 1 means exit
 * other values are function pointers
 * the function sets command back to 0 when it is finished
 */



/* QUESTION
 *
 * would it be faster to have one cacheline for the command \
 * and a different one for the arguments?  If the am is sent with a single
 * 64 byte store, then no, but if it is built up word by word maybe yes
 */

struct AMSet;   // Forward declaration
typedef void *AMBarrier;   /* hide the underlying object */
/* the barrier is actually a SpinBarrier  and must be used in pairs */

typedef struct AMMsg {
  void (*volatile command) (struct AMMsg *);
  AMBarrier sync;             // a barrier object 
  uint64_t arg[4];
  uint64_t ret;
  long cycles;
} AMMsg __attribute__ ((aligned(CACHELINE)));

/* There should be a way to automatically pad a struct to be a multiple\
 * of a cacheline 
 */
typedef struct AMThread {
  int32_t tid;   // a 0 based index of the threads in this AMSet
  int32_t pad1;
  struct AMSet *ams;    // forward reference
  pthread_t pt_tid;
  pthread_attr_t pt_attributes;
  uint64_t pad2[6];
} AMThread  __attribute__ ((aligned(CACHELINE)));

typedef struct AMSet {
  AMMsg *am;    // vector of AMMsg objects
  int num_workers;
  AMThread *tp;        // vector of worker threads  
} AMSet;


#define AMCPU(amset, cpu) (&amset->am[(cpu)])

/* Waits for AM completion and returns function return value */
uint64_t AM_Wait(AMMsg *am);
/* Waits for AM completion and returns running time in cycles */
long AM_Wait_Time(AMMsg *am);

void AM_WaitForAll(AMSet *ams);

/* AM_Call is like AM_Send but it uses the calling thread to execute the command */
void AM_Call(AMMsg *am, void (*command) (AMMsg *p), AMBarrier sync, int count, ...);

/* AM_Send takes an AM_Msg*, typically &ams->am[cpu] in order to avoid
 * touching a shared AMSet object.  Clients should cache ams->am
 * Use the macro AMCPU(ams, cpu) for this
 */
/* WARNING make sure all variable length arguments are cast to uint64_t or
 * are 64 bit size types 
 */
void AM_Send(AMMsg *am, void (*command) (AMMsg *p), AMBarrier  sync, int count, ...);

/* or you can create a prototype message to use over and over again */
/* WARNING make sure all variable length arguments are cast to uint64_t or
 * are 64 bit size types 
 */
void AM_Compose(AMMsg *am, void (*command) (AMMsg *p), AMBarrier  sync, int count, ...);
/* and start it */
void AM_Start(AMMsg *to, AMMsg *msg);

/* send the given message to all cores in ams squentially */
void AM_Run_All_M(AMSet *ams, AMMsg *am);
void AM_Run_All(AMSet *am, void (*command) (AMMsg *p), AMBarrier  sync, int count, ...);

/* same, but waits for completion and returns time taken */
/* WARNING make sure all variable length arguments are cast to uint64_t or
 * are 64 bit size types 
 */
long AM_Send_Wait_Time(AMMsg *am, void (*command) (AMMsg *p), AMBarrier  sync, int count, ...);

// Get return value of most recent AM
uint64_t AM_GetReturn(AMMsg *m);
// Get cycles taken by most recent AM
long AM_GetTime(AMMsg *m);

// Create and destroy an active message set
AMSet *AM_Create(int n);
void AM_Destroy(AMSet *ams);

void AM_StartWorkers(AMSet *ams);
void AM_JoinWorkers(AMSet *ams);

AMBarrier AM_CreateBarrier(int n);
void AM_DestroyBarrier(AMBarrier b);
void AM_Barrier_Wait(AMBarrier b);
void AM_BarrierSetSize(AMBarrier b, uint64_t size);


// predefined commands

void AM_do_tsc_overhead(AMMsg *p);
void AM_do_rdtsc(AMMsg *p);

// AMMsgs to get and set CPU affinity
void AM_do_getcpu(AMMsg *p);
void AM_do_setcpu(AMMsg *p);

// predefined functions to call the above

uint64_t AM_fn_tsc_overhead(AMSet *ams, int cpu);
uint64_t AM_fn_rdtsc(AMSet *ams, int cpu);

int AM_fn_getcpu(AMSet *ams, int cpu);
void AM_fn_setcpu(AMSet *ams, int cpu, int affinity);


#endif
