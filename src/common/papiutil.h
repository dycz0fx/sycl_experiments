/* papiutil.h
 *
 * encapsulating managing papi counters on a per-thread/core basis for 
 * code using the ActiveMessage framework
 */

#ifndef PAPIUTIL_H
#define PAPIUTIL_H

#include <pthread.h>
#include <ActiveMessage.h>
#include <papi.h>

int papiutil_get_num_counters();
int *papiutil_alloc_events();
void papiutil_free_events(int *events);

long long *papiutil_alloc_counters();
void papiutil_free_counters(long long *counters);


void papiutil_init();

int papiutil_raw_start(int *events, int num_events);
void papiutil_raw_read(int evsetid, long long *counters);
void papiutil_raw_stop(int evsetid, long long *counters);




void papiutil_do_start(AMMsg *p);
void papiutil_do_read(AMMsg *p);
void papiutil_do_stop(AMMsg *p);

/* Look for PAPI events in environment variables <prefix>#=xxx 
 * so if prefix is "C" then look for events
 * C0=xxx
 * C1=xxx
 * C2=xxx
 */

int papiutil_get_env_events(char *prefix, int *events);
int papiutil_get_string_events(char *arg, int *events);

void papiutil_start_papi(AMSet *ams);

void papiutil_printcounters_one(int *events, int num_events, long long *after);

void papiutil_printcounters(int *events, int num_events, long long *before, long long *after);

#endif
