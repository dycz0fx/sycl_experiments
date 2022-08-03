#ifndef CPU_AFFINITY__H
#define CPU_AFFINITY__H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE             /* See feature_test_macros(7) */
#endif
#include <sched.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static void set_cpu(int id){
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();

    CPU_ZERO(&cpuset);
    CPU_SET(id, &cpuset);

    int ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if(ret != 0){
      fprintf(stderr, "Unable to set affinity to %d\n", id);
        assert(0);
    }
}

static int get_cpu(){
    cpu_set_t cs;
    pthread_t thread;

    thread = pthread_self();

    CPU_ZERO(&cs);

    int ret = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cs);
    if(ret != 0){
        fprintf(stderr, "Unable to get affinity\n");
        assert(0);
    }
    for (int i = 0; i < (int) sizeof(cs)*8; i = i + 1) 
      {
	if (CPU_ISSET(i, &cs)) return(i);
      }
    return (0);
}

static int get_number_of_cpus()
{
    cpu_set_t cs;
    CPU_ZERO(&cs);
    sched_getaffinity(0, sizeof(cs), &cs);

    int count = 0;
    for (int i=0; i<(int)sizeof(cs)*8; i++)
    {
        if (CPU_ISSET(i, &cs))
            count++;
    }
    return count;
}

#endif
