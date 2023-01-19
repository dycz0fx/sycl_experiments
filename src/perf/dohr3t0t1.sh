#!/bin/bash
set -e

for threads in 1 2 4 8 16 ; do
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=500000 --b2acount=0
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=500000 --b2acount=0
done
for threads in 32 64 128 256; do
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=10000000 --b2acount=0
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=10000000 --b2acount=0
done

for threads in 1 2 4 8 16 ; do
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=500000 --b2acount=500000
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=500000 --b2acount=500000
done
for threads in 32 64 128 256; do
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=10000000 --b2acount=10000000
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=10000000 --b2acount=10000000
done
