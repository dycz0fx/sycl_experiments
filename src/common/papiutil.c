/* papiutil.c
 *
 * encapsulating managing papi counters on a per-thread/core basis for 
 * code using the ActiveMessage framework
 */

#include "papiutil.h"
#include <stdio.h>
#include <immintrin.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int papiutil_num_counters = -1;

int papiutil_get_num_counters()
{
  return(papiutil_num_counters);
}

int *papiutil_alloc_events()
{
  assert(papiutil_num_counters != -1);
  int *events = _mm_malloc(sizeof(int) * papiutil_num_counters, 64);
  assert(events);
  return(events);
}


void papiutil_init()
{
  int result = PAPI_library_init(PAPI_VER_CURRENT);
  printf("PAPI_VER_CURRENT is %u\n", PAPI_VER_CURRENT);
  if (result != PAPI_VER_CURRENT) {
    if (result > 0)
      printf("PAPI_library_init version mismatch: %d %x\n", result, result);
    else
      printf("PAPI_library_init: %d %s\n", result, PAPI_strerror(result));
  }
  if ((result = PAPI_thread_init(pthread_self)) != PAPI_OK) {
    printf("PAPI_thread_init failed: %s\n", PAPI_strerror(result));
    exit(1);
  }
  unsigned long int tid;
  if ((tid = PAPI_thread_id()) == (unsigned long int) -1) {
    printf("PAPI_thread_id() returned %ld\n", (long int) tid);
    exit(1);
  }
  papiutil_num_counters = PAPI_num_counters();
  printf("PAPI has %d counters\n", papiutil_num_counters);
}


int papiutil_raw_start(int *events, int num_events)
{
  int result;
  int evsetid;
  evsetid = PAPI_NULL;
  result = PAPI_create_eventset(&evsetid);
  if (result != PAPI_OK) {
    printf("PAPI_create_event_set: %d %s\n",
	   result, PAPI_strerror(result));
    exit(1);
  }
  result = PAPI_add_events(evsetid, events, num_events);
  if (result != PAPI_OK) {
    printf("PAPI_addevents: %d %s\n",
	   result, PAPI_strerror(result));
    exit(1);
  }
  result = PAPI_start(evsetid);
  if (result != PAPI_OK) {
    printf("PAPI_start: %d %s\n",
	   result, PAPI_strerror(result));
    exit(1);
  }
  return(evsetid);
}

void papiutil_do_start(AMMsg *p)
{
  int *events = (int *) p->arg[0];
  int num_events = (int) p->arg[1];
  int evsetid = papiutil_raw_start(events, num_events);
  p->ret = evsetid;
}

void papiutil_raw_read(int evsetid, long long *counters)
{
  int rc;
  rc = PAPI_read(evsetid, counters) ;
  if (rc != PAPI_OK) {
    fprintf(stderr, "read PAPI_read_counters evsetid %d - FAILED %s \n", evsetid, PAPI_strerror(rc));
    exit(1);
  }
}

// first arg cpu, second arg counter array
void papiutil_do_read(AMMsg *p)
{
  int evsetid = (long int) p->arg[0];
  long long *counters = (long long *) p->arg[1];
  papiutil_raw_read(evsetid, counters);
}

void papiutil_raw_stop(int evsetid, long long *counters)
{
  int rc;
  if ((rc = PAPI_stop(evsetid, counters)) != PAPI_OK) {
    fprintf(stderr, "stop PAPI_read_counters evsetid %d - FAILED: %s\n", evsetid, PAPI_strerror(rc));
    exit(1);
  }
}

void papiutil_do_stop(AMMsg *p)
{
  int evsetid = (int) p->arg[0];
  long long *counters = (long long *) p->arg[1];
  papiutil_raw_stop(evsetid, counters);
}


int papiutil_get_env_events(char *prefix, int *events)
{

  int result;
  char envname[PAPI_MAX_STR_LEN];
  int i, ev;
  assert(strlen(prefix) < 15);
  // num_events is global 
  i = 0;
  ev = 0;
  assert(papiutil_num_counters != -1);
  for (i = 0; i < papiutil_num_counters; i += 1) {
    events[i] = 0;
    snprintf(envname, PAPI_MAX_STR_LEN, "%s%d", prefix, i);
    char *eval = getenv(envname);
    if (eval == NULL) continue;
    printf("papiutil found %s = %s\n", envname, eval);
    result = PAPI_event_name_to_code(eval, &events[ev]);
    if (result != PAPI_OK) {
      printf("PAPI_event_name_to_code(%s): %d %s\n", eval, result, PAPI_strerror(result));
      exit(1);
    }
    ev += 1;
  }
  return(ev);
}  

int papiutil_get_string_events(char *arg, int *events)
{
  char *token = strtok(arg, ",");
  int ev = 0;
  int result;
  assert(papiutil_num_counters != -1);
  while (token) {
    token= strtok(NULL, ",");
    printf("papiutil found %s\n", token);
    result = PAPI_event_name_to_code(token, &events[ev++]);
    if (result != PAPI_OK) {
      printf("PAPI_event_name_to_code(%s): %d %s\n", token, result, PAPI_strerror(result));
      exit(1);
    }
    if (ev == papiutil_num_counters) break;
  }
  return(ev);
}


long long *papiutil_alloc_counters()
{
  assert(papiutil_num_counters != -1);
  size_t size = sizeof(long long) * papiutil_num_counters;
  long long *counters = _mm_malloc(size, 64);
  assert(counters);
  return(counters);
}



void papiutil_printcounters(int *events, int num_events, long long *before, long long *after)
{  
  int result;
  for (int i = 0; i < num_events; i += 1) {
    char ename[128];
    result = PAPI_event_code_to_name(events[i], ename);
    if (result != PAPI_OK) strcpy(ename, "unknown");
    printf("%s: %lld\n", ename, after[i] - before[i]);
  }
}

void papiutil_printcounters_one(int *events, int num_events, long long *after)
{  
  int result;
  for (int i = 0; i < num_events; i += 1) {
    char ename[128];
    result = PAPI_event_code_to_name(events[i], ename);
    if (result != PAPI_OK) strcpy(ename, "unknown");
    printf("%s: %lld\n", ename, after[i]);
  }
}


    
void papiutil_free_events(int *events)
{
  _mm_free(events);
}

void papiutil_free_counters(long long *counters)
{
  _mm_free(counters);
}
