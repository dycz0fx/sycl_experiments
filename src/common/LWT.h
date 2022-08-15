/* LWT.h 
 *
 * Lightweight Threads
 */

#ifndef LWT_H
#define LWT_H

#include "ActiveMessage.h"
#include "SpinBarrier.h"

#define CACHELINE 64
#define MAXCPU 64


struct LWTData {
  uint64_t cpuset;
  int64_t num_cpus;
}  __attribute__ ((aligned(CACHELINE)));

struct LWTStep {
  SpinBarrier_t sb;
  struct LWTData data;
  AMMsg am[MAXCPU];
  char *description;
  //long c_start[MAXCPU];
  //long c_end[MAXCPU];
}  __attribute__ ((aligned(CACHELINE)));

struct LWT {
  AMSet *ams;
  int64_t num_workers;
  int64_t num_steps;
  int64_t num_cpus;
  uint64_t cpuset;     // all cpus used anywhere
  struct LWTStep *steps;    // arrayof LWTStep
  int bind_cpu[MAXCPU];     // where each thread is pinned
}  __attribute__ ((aligned(CACHELINE)));

typedef struct LWT LWT_t;

/* Bit field operations */
#define ISSET(x, i)  (((x) >> (i)) & 1)
#define SETBIT(x, i)  (x) |= (1 << (i))
#define ISCLEAR(x, i) (!ISSET(x,i))

/* A test is a sequence of steps
 * A step is a set of CPUs each running an AM
 *
 * At the beginning, all the cpus are sent an AMMsg to run LWT, with 
 * a pointer to the Test and their ID
 * Before running a Step, each CPU spins on the Notifier
 * The last cpu to finish the previous step notifies all the waiters
 * by FFOing through the cpuset and Starting each cpu by writing to its notifier
 *
 * An improvement will be a notifier tree, signalled by a different something
 */



LWT_t *LWT_Create(int cpus, int steps);
void LWT_Reset(LWT_t *lwt);
void LWT_setcpu(LWT_t *lwt, int cpu, int bind);
void LWT_Start(LWT_t *lwt);
void LWT_Wait(LWT_t *lwt);
void LWT_Destroy(LWT_t *lwt);
void LWT_Set_Description(LWT_t *lwt, int step, char *description);
void LWT_Set_Action(LWT_t *lwt, int step, int cpu, 
		    void (*command) (AMMsg *p), int count, ...);
void LWT_Worker(AMMsg *m);

long LWT_Get_Cycles(LWT_t *lwt, int step, int cpu);
uint64_t LWT_Get_Return(LWT_t *lwt, int step, int cpu);
char *LWT_Get_Description(LWT_t *lwt, int step);

#endif
