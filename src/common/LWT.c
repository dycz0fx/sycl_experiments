/* Lightweight Thread library
 */


#include "LWT.h"
#include "rdtsc.h"
#include <assert.h>
#include <string.h>
#include "cpu_affinity.h"  // set_cpu

void LWT_Reset(LWT_t *lwt)
{
  memset(lwt->steps, 0, sizeof(struct LWTStep) * lwt->num_steps);

  for (unsigned i = 0; i < lwt->num_steps; i += 1) {
    SpinBarrier_Init(&lwt->steps[i].sb, lwt->num_workers);
  }
}

LWT_t *LWT_Create(int cpus, int steps)
{
  LWT_t *lwt = _mm_malloc (sizeof(LWT_t), CACHELINE);
  assert(lwt);
  //printf("lwt is %p\n", lwt);
  assert(cpus > 0);
  assert(cpus < MAXCPU);
  lwt->num_workers = cpus;
  if (cpus > 1){
    lwt->ams = AM_Create(cpus-1);  // hacky - LWT 0 will be the master)
    AM_StartWorkers(lwt->ams);
  }
  assert(steps > 0);
  lwt->steps = _mm_malloc (sizeof(struct LWTStep) * steps, CACHELINE);
  //printf("lwt->steps is %p\n", lwt->steps);
  assert(lwt->steps);
  lwt->num_steps = steps;
  LWT_Reset(lwt);
  //printf("LWT_Reset returns\n");
  return(lwt);
}

void LWT_setcpu(LWT_t *lwt, int cpu, int bind)
{
  lwt->bind_cpu[cpu] = bind;
  if (cpu != 0) AM_fn_setcpu(lwt->ams, cpu - 1, bind);
  else set_cpu(bind);
}


void LWT_Start(LWT_t *lwt)
{
  AMMsg p;
  if (lwt->ams) {
    for (int cpu = 0; cpu < lwt->ams->num_workers; cpu += 1) {
      AM_Send(AMCPU(lwt->ams, cpu), LWT_Worker, NULL, 2, 
	      (uint64_t) lwt, (uint64_t) cpu + 1);
    }
  }
  AM_Compose(&p, LWT_Worker, NULL, 2, (uint64_t) lwt, (uint64_t) 0);
  LWT_Worker(&p);
}

void LWT_Wait(LWT_t *lwt)
{
  if (lwt->ams) {
    AM_WaitForAll(lwt->ams);
  }
}

void LWT_Destroy(LWT_t *lwt)
{
  if (lwt->ams) {
    AM_JoinWorkers(lwt->ams);
    AM_Destroy(lwt->ams);
  }
  _mm_free(lwt->steps);
  _mm_free(lwt);
}



void LWT_Set_Action(LWT_t *lwt, int step, int cpu, 
		    void (*command) (AMMsg *p), int count, ...)
{
  assert(lwt);
  assert(step < lwt->num_steps);
  assert(cpu < lwt->num_workers);
  assert(cpu < MAXCPU);
  if (command == NULL) return;
  assert(count <= 4);
  va_list ap;
  va_start(ap, count);
  for (int i = 0; i < count; i += 1) {
    lwt->steps[step].am[cpu].arg[i] = va_arg(ap, uint64_t);
  }
  va_end(ap);
  lwt->steps[step].am[cpu].command = command;
  lwt->steps[step].am[cpu].sync = NULL;
  /* recompute num_cpus from bitfields, but its only one instruction! */
  SETBIT(lwt->steps[step].data.cpuset, cpu);
  lwt->steps[step].data.num_cpus = 
    _mm_countbits_64(lwt->steps[step].data.cpuset);
  //SpinBarrier_SetSize(&lwt->steps[step].sb, lwt->steps[step].data.num_cpus);
  SETBIT(lwt->cpuset, cpu);
  lwt->num_cpus = _mm_countbits_64(lwt->cpuset);
}

void LWT_Set_Description(LWT_t *lwt, int step, char *description)
{
  lwt->steps[step].description = description;
}


void LWT_Worker(AMMsg *m)
{
  assert(m);
  LWT_t *lwt = (LWT_t *) m->arg[0];
  int cpu = m->arg[1];
  assert (lwt);
  assert(cpu >= 0);
  assert(cpu < MAXCPU);
  long c_start, c_end;   // cycles
  // printf("cpu %d LWT ready\n", cpu);
  for (int step = 0; step < lwt->num_steps; step += 1) {
    // printf("Thread %d step %d \n", cpu, step);
    // fflush(NULL);
    SpinBarrier_Wait(&lwt->steps[step].sb);
    AMMsg *c = &lwt->steps[step].am[cpu];
    if (ISCLEAR(lwt->steps[step].data.cpuset, cpu)) {
      c->cycles = 0;
      continue;
    }
    c_start = rdtsc();
    //lwt->steps[step].c_start[cpu] = c_start;
    __sync_synchronize();
    (c->command)((AMMsg *) c);
    __sync_synchronize();
    c_end = rdtscp();
    //lwt->steps[step].c_end[cpu] = c_end;
    c->cycles = c_end - c_start;
    //printf("cpu %d finished step %d in %ld\n", cpu, step, c->cycles);
  }
}

long LWT_Get_Cycles(LWT_t *lwt, int step, int cpu)
{
  return(lwt->steps[step].am[cpu].cycles);
}

uint64_t LWT_Get_Return(LWT_t *lwt, int step, int cpu)
{
  return(lwt->steps[step].am[cpu].ret);
}

char *LWT_Get_Description(LWT_t *lwt, int step)
{
  return(lwt->steps[step].description);
}
