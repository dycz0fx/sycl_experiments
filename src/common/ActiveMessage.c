/* ActiveMessage.c
 *
 */

#define _GNU_SOURCE             /* See feature_test_macros(7) */

#include "ActiveMessage.h"

// #include <sched.h>
#include "cpu_affinity.h"
#include <assert.h>
#include "rdtsc.h"
#include <stdio.h>
#include <string.h>
#include "SpinBarrier.h"
#include <immintrin.h>    /* _mm_malloc and _mm_free */
//#include <pmmintrin.h> 
// we could use MWAIT if it worked in user mode

#define LOAD(reg,p) \
  __asm__ volatile ("vmovaps (%1), %0\n": "=x"(reg): "r"(p));

#define USEMWAIT 0

#if USEMWAIT
#define WAITFORIDLE(am)              \
  while(am->command != NULL) {       \
    _mm_monitor(am, 0, 0);           \
    if (am->command == NULL) break;  \
    _mm_mwait(0,0);                  \
  }                                  \
  __sync_synchronize();
#else                                
#define WAITFORIDLE(am)              \
  while(am->command != NULL) {       \
    _mm_pause();                     \
  }                                  \
  __sync_synchronize();
#endif                                


void AM_Call(AMMsg *am, void (*command) (AMMsg *p), AMBarrier sync, int count, ...)
{
  long c_start, c_end;
  assert(count <= 4);
  am->sync = sync;
  am->ret = (uint64_t) -1;
  va_list ap;
  va_start(ap, count);
  for (int i = 0; i < count; i += 1) {
    am->arg[i] = va_arg(ap, uint64_t);
  }
  va_end(ap);
  __sync_synchronize();

  c_start = rdtsc();
  (command)((AMMsg *) am);
  c_end = rdtscp();
  am->cycles = c_end - c_start;

}

void AM_Send(AMMsg *am, void (*command) (AMMsg *p), AMBarrier sync, int count, ...)
{
  assert(count <= 4);
  WAITFORIDLE(am);
  am->sync = sync;
  am->ret = (uint64_t) -1;
  va_list ap;
  va_start(ap, count);
  for (int i = 0; i < count; i += 1) {
    am->arg[i] = va_arg(ap, uint64_t);
  }
  va_end(ap);
  __sync_synchronize();
  am->command = command;  // actually start the command 
}

void AM_Compose(AMMsg *am, void (*command) (AMMsg *p), AMBarrier sync, int count, ...)
{
  am->sync = sync;
  assert(count <= 4);
  va_list ap;
  va_start(ap, count);
  for (int i = 0; i < count; i += 1) {
    am->arg[i] = va_arg(ap, uint64_t);
  }
  va_end(ap);
  am->command = command;
}

void AM_Start(AMMsg *to, AMMsg *msg)
{
#if 0
  LOAD(r0, (__m512i *) msg);
  __m512i r0;
  WAITFORIDLE(to);
  _mm512_store_si512((__m512i *) to, r0);
  __sync_synchronize();
#endif
  WAITFORIDLE(to);
  memcpy(&to->sync, &msg->sync, 56);
  __sync_synchronize();
  to->command = msg->command;
}


long AM_Send_Wait_Time(AMMsg *am, void (*command) (AMMsg *p), AMBarrier sync, int count, ...)
{
  assert(count <= 4);
  WAITFORIDLE(am);
  am->sync = sync;
  va_list ap;
  va_start(ap, count);
  for (int i = 0; i < count; i += 1) {
    am->arg[i] = va_arg(ap, uint64_t);
  }
  va_end(ap);
  __sync_synchronize();
  am->command = command;  // actually start the command 
  return(AM_Wait_Time(am));
}

void AM_Run_All_M(AMSet *ams, AMMsg *am)
{
  for (int i = 0; i < ams->num_workers; i += 1) {
    AM_Start(AMCPU(ams, i), am);
    AM_Wait(AMCPU(ams, i));
  }
}

void AM_Run_All(AMSet *ams, void (*command) (AMMsg *p), AMBarrier sync, int count, ...)
{
  AMMsg m;
  m.sync = sync;
  assert(count <= 4);
  va_list ap;
  va_start(ap, count);
  for (int i = 0; i < count; i += 1) {
    m.arg[i] = va_arg(ap, uint64_t);
  }
  va_end(ap);
  m.command = command;
  AM_Run_All_M(ams, &m);
}


void *AM_Worker(void *ptr);
void *AM_Worker(void *ptr){
  assert(ptr);
  AMThread *tp = ptr;
  int tid = tp->tid;
  AMMsg *c = &tp->ams->am[tid];
  long c_start, c_end;
  // printf("tid %d ready\n", tid);
  for (;;) {
    // wait for command
#if USEMWAIT
    while(c->command == NULL) {
      _mm_monitor(c, 0, 0);
      if (c->command != NULL) break;
      _mm_mwait(0,0);
    }
#else
    while(c->command == NULL) {
      _mm_pause();
    }
#endif
    __sync_synchronize();
    if (c->sync) SpinBarrier_Wait((SpinBarrier_t *) c->sync);
    if (c->command == (void *) 1L) break;
    // printf("tid %d running\n", tid);
    c_start = rdtsc();
    (c->command)((AMMsg *) c);
    c_end = rdtscp();
    c->cycles = c_end - c_start;
    // printf("tid %d finished in %lu\n", tid, c->cycles);
    __sync_synchronize();
    c->command = NULL;
    __sync_synchronize();
    if (c->sync) SpinBarrier_Wait((SpinBarrier_t *) c->sync);
  }
  c->command = NULL;    // signal completion of exit function
  return NULL;
}

uint64_t AM_GetReturn(AMMsg *m)
{
  return(m->ret);
}

long AM_GetTime(AMMsg *m)
{
  return((long) m->cycles);
}

AMSet *AM_Create(int n)
{
  AMSet *ams = (AMSet *) _mm_malloc(sizeof(AMSet), CACHELINE);
  assert((sizeof(AMThread) % CACHELINE) == 0);
  assert((sizeof(AMMsg) % CACHELINE) == 0);
  ams->num_workers = n;
  ams->am = (AMMsg *) _mm_malloc(sizeof(AMMsg) * ams->num_workers, CACHELINE);
  ams->tp = NULL;
  for (int i = 1; i < ams->num_workers; i += 1) {
    ams->am[i].command = NULL;
    ams->am[i].sync = NULL;
  }
  return(ams);
}

void AM_Destroy(AMSet *ams)
{
  assert(ams);
  assert(ams->tp == NULL);
  _mm_free((AMMsg *) ams->am);
  ams->am = NULL;
  _mm_free(ams);
}

void AM_StartWorkers(AMSet *ams){
  assert(ams);
  assert(ams->tp == NULL);
  ams->tp = (AMThread *) _mm_malloc(sizeof(AMThread) * ams->num_workers, CACHELINE);
  assert(ams->tp);
  for(int i = 0; i < ams->num_workers; i += 1){
    ams->tp[i].tid = i;
    ams->tp[i].ams = ams;
    pthread_attr_init(&ams->tp[i].pt_attributes);
    ams->am[i].command = NULL;

    pthread_create(&ams->tp[i].pt_tid,
		   &ams->tp[i].pt_attributes,
		   AM_Worker, &ams->tp[i]);
    }
}

void AM_JoinWorkers(AMSet *ams)
{
  assert(ams);
  assert(ams->tp);
  for (int i = 0; i < ams->num_workers; i += 1) {
    AM_Send(&ams->am[i], (void (*) (AMMsg *p)) 1L, 0, 0);
    //printf("send exit to %d\n", i);
    //fflush(NULL);
    pthread_join(ams->tp[i].pt_tid, NULL);
    //printf("joined\n");
    //fflush(NULL);
    pthread_attr_destroy(&ams->tp[i].pt_attributes);
    ams->am[i].command = NULL;
  }
  _mm_free(ams->tp);
  ams->tp = NULL;
}

uint64_t AM_Wait(AMMsg *am)
{
  WAITFORIDLE(am);
  return (am->ret);
}

long AM_Wait_Time(AMMsg *am)
{
  WAITFORIDLE(am);
  return((long) am->cycles);
}

void AM_WaitForAll(AMSet *ams)
{
  for (int i = 0; i < ams->num_workers; i += 1) AM_Wait((AMMsg *) &ams->am[i]);
}



AMBarrier AM_CreateBarrier(int n)
{
  AMBarrier barr = (AMBarrier *) _mm_malloc(sizeof(SpinBarrier_t), CACHELINE);

  SpinBarrier_Init((SpinBarrier_t *) barr, n);
  return (barr);
}

void AM_BarrierSetSize(AMBarrier b, uint64_t size)
{
  SpinBarrier_SetSize((SpinBarrier_t *) b, size);
}

void AM_DestroyBarrier(AMBarrier b)
{
  _mm_free(b);
}

void AM_Barrier_Wait(AMBarrier b)
{
  SpinBarrier_Wait((SpinBarrier_t *) b);
}


void AM_do_rdtsc(AMMsg *p)
{
  p->ret = rdtsc();
}

void AM_do_tsc_overhead(AMMsg *p)
{
  p->ret = tsc_overhead();
}

void AM_do_getcpu(AMMsg *p)
{
  p->ret = get_cpu();
}

void AM_do_setcpu(AMMsg *p)
{
  set_cpu(p->arg[0]);
}

uint64_t AM_fn_tsc_overhead(AMSet *ams, int cpu)
{
  AM_Send(&ams->am[cpu], AM_do_tsc_overhead, 0, 0);
  return(AM_Wait((AMMsg *) &ams->am[cpu]));
}

uint64_t AM_fn_rdtsc(AMSet *ams, int cpu)
{
  AM_Send(&ams->am[cpu], AM_do_rdtsc, 0, 0);
  return(AM_Wait(&ams->am[cpu]));
}

int AM_fn_getcpu(AMSet *ams, int cpu)
{
  AM_Send(&ams->am[cpu], AM_do_getcpu, 0, 0);
  return(AM_Wait(&ams->am[cpu]));
}

void AM_fn_setcpu(AMSet *ams, int cpu, int affinity)
{
  //printf("set cpu %d on %d\n", cpu,affinity);
  //fflush(NULL);
  AM_Send(&ams->am[cpu], AM_do_setcpu, 0, 1, (long) affinity);
  AM_Wait(&ams->am[cpu]);
}
