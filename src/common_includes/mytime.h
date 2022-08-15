#ifndef MY_TIME_HH
#define MY_TIME_HH

#include <time.h>
 
static inline struct timespec get_time(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t;
}

static inline struct timespec timespec_diff(struct timespec start, struct timespec end){
    struct timespec temp;
    if((end.tv_nsec - start.tv_nsec) < 0){
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }else{
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

static inline void print_time_diff(struct timespec start, struct timespec end){
    struct timespec diff = timespec_diff(start, end);
    printf("%lld.%.9ld", (long long)diff.tv_sec, diff.tv_nsec);
}

#endif
