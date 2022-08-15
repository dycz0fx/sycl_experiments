#include "barrier.h"

void barrier_destroy(barrier_t *b)
{
    pthread_mutex_lock(&b->m);

    while (b->total > BARRIER_FLAG)
    {
        /* 等待所有线程退出 barrier. */
        pthread_cond_wait(&b->cv, &b->m);
    }

    pthread_mutex_unlock(&b->m);

    pthread_cond_destroy(&b->cv);
    pthread_mutex_destroy(&b->m);
}

void barrier_init(barrier_t *b, unsigned count)
{
    pthread_mutex_init(&b->m, NULL);
    pthread_cond_init(&b->cv, NULL);
    b->count = count;
    b->total = BARRIER_FLAG;
}

int barrier_wait(barrier_t *b)
{
    pthread_mutex_lock(&b->m);

    while (b->total > BARRIER_FLAG)
    {
        pthread_cond_wait(&b->cv, &b->m);
    }

    /* 是否为第一个到达 barrier 的线程? */
    if (b->total == BARRIER_FLAG) b->total = 0;

    b->total++;

    if (b->total == b->count)
    {
        b->total += BARRIER_FLAG - 1;
        pthread_cond_broadcast(&b->cv);

        pthread_mutex_unlock(&b->m);

        return 1;// PTHREAD_BARRIER_SERIAL_THREAD;
    }
    else
    {
        while (b->total < BARRIER_FLAG)
        {
            pthread_cond_wait(&b->cv, &b->m);
        }

        b->total--;

        /* 唤醒所有进入 barrier 的线程. */
        if (b->total == BARRIER_FLAG) pthread_cond_broadcast(&b->cv);

        pthread_mutex_unlock(&b->m);

        return 0;
    }
}
