#ifndef _PTHREAD_BARRIER_H_
#define _PTHREAD_BARRIER_H_

#include <pthread.h>

typedef struct barrier_s_ barrier_t;

struct barrier_s_ {
    unsigned count;
    unsigned total;
    pthread_mutex_t m;
    pthread_cond_t cv;
};

#define BARRIER_FLAG (1UL<<31)

extern void barrier_init(barrier_t *b, unsigned count);

extern int barrier_wait(barrier_t *b);

extern void barrier_destroy(barrier_t *b);

#endif /* _PTHREAD_BARRIER_H_ */
