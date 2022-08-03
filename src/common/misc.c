#include "misc.h"
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "magic_marker.h"

void ssc_mark_1(){
    MARKER(ROI_START_MARKER);
}
void ssc_mark_2(){
    MARKER(ROI_END_MARKER);
}

#if 0
void movget_memcpy(void* dst, void *src, int nb){
    __m512i x;
    char *dstc = dst;
    char *srcc = src;

#ifdef DEBUG_MODE
    if(nb%64 != 0){
        fprintf(stderr, "Bad movget memcpy size\n");
        exit(1);
    }
#endif

    nb /= 64;

    for(int i=0; i<nb; i++){
        x = _mm512_stream_load_si512(srcc);
        _mm512_store_si512(dstc, x);
        dstc += 64;
        srcc += 64;
    }
}
#endif

void print__m512i(__m512i x){
    printf("0x%08x-%08x-%08x-%08x-%08x-%08x-%08x-%08x-%08x-%08x-%08x-%08x-%08x-%08x-%08x-%08x\n",
            EXTRACT(x, 0),
            EXTRACT(x, 1),
            EXTRACT(x, 2),
            EXTRACT(x, 3),
            EXTRACT(x, 4),
            EXTRACT(x, 5),
            EXTRACT(x, 6),
            EXTRACT(x, 7),
            EXTRACT(x, 8),
            EXTRACT(x, 9),
            EXTRACT(x, 10),
            EXTRACT(x, 11),
            EXTRACT(x, 12),
            EXTRACT(x, 13),
            EXTRACT(x, 14),
            EXTRACT(x, 15)
          );
}


volatile int trace_flag;
void start_trace() 
{
  trace_flag = 1;
}

