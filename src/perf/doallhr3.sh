#!/bin/bash
set -e
LOCOUNT=1000000
HICOUNT=1000000
LOTHREADS="1 2 4 8 16"
HITHREADS="32 64 128 256"


for runs in 1 2 3 4 5; do
    srun src/perf/helperring3a --a=c --b=c --a2bbuf=c --b2abuf=c --athreads=1 --bthreads=1 --a2bcount=${LOCOUNT} --b2acount=0
done
for runs in 1 2 3 4 5; do
    srun src/perf/helperring3a --a=c --b=c --a2bbuf=c --b2abuf=c --athreads=1 --bthreads=1 --a2bcount=${LOCOUNT} --b2acount=${LOCOUNT}
done

for bthreads in ${LOTHREADS}; do
    srun src/perf/helperring3a --a=c --b=p0 --a2bbuf=p0 --b2abuf=c --athreads=1 --bthreads=${bthreads} --a2bcount=${LOCOUNT} --b2acount=0
done
for bthreads in ${HITHREADS}; do
    srun src/perf/helperring3a --a=c --b=p0 --a2bbuf=p0 --b2abuf=c --athreads=1 --bthreads=${bthreads} --a2bcount=${HICOUNT} --b2acount=0
done

for bthreads in ${LOTHREADS}; do
    srun src/perf/helperring3 --a=c --b=p0 --a2bbuf=p0 --b2abuf=c --athreads=1 --bthreads=${bthreads} --a2bcount=0 --b2acount=${LOCOUNT}

done

for bthreads in ${HITHREADS}; do
    srun src/perf/helperring3 --a=c --b=p0 --a2bbuf=p0 --b2abuf=c --athreads=1 --bthreads=${bthreads} --a2bcount=0 --b2acount=${HICOUNT}
done

for bthreads in ${LOTHREADS}; do
    srun src/perf/helperring3 --a=c --b=p0 --a2bbuf=p0 --b2abuf=c --athreads=1 --bthreads=${bthreads} --a2bcount=${LOCOUNT} --b2acount=${LOCOUNT}
done

for bthreads in ${HITHREADS}; do
    srun src/perf/helperring3 --a=c --b=p0 --a2bbuf=p0 --b2abuf=c --athreads=1 --bthreads=${bthreads} --a2bcount=${HICOUNT} --b2acount=${HICOUNT}
done

for threads in ${LOTHREADS} ; do
    srun src/perf/helperring3 --a=p0 --b=p1 --a2bbuf=p1 --b2abuf=p0 --athreads=${threads} --bthreads=${threads} --a2bcount=${LOCOUNT} --b2acount=0
done

for threads in ${HITHREADS}; do
    srun src/perf/helperring3 --a=p0 --b=p1 --a2bbuf=p1 --b2abuf=p0 --athreads=${threads} --bthreads=${threads} --a2bcount=${HICOUNT} --b2acount=0
done

for threads in ${LOTHREADS} 32; do
    srun src/perf/helperring3 --a=p0 --b=p1 --a2bbuf=p1 --b2abuf=p0 --athreads=${threads} --bthreads=${threads} --a2bcount=${LOCOUNT} --b2acount=${LOCOUNT}
done

for threads in ${HITHREADS}; do
    srun src/perf/helperring3 --a=p0 --b=p1 --a2bbuf=p1 --b2abuf=p0 --athreads=${threads} --bthreads=${threads} --a2bcount=${HICOUNT} --b2acount=${HICOUNT}
done

for threads in ${LOTHREADS} ; do
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=${LOCOUNT} --b2acount=0
done

for threads in ${HITHREADS}; do
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=${HICOUNT} --b2acount=0
done

for threads in ${LOTHREADS} ; do
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=${LOCOUNT} --b2acount=${LOCOUNT}
done

for threads in ${HITHREADS}; do
    srun src/perf/helperring3 --a=t0 --b=t1 --a2bbuf=t1 --b2abuf=t0 --athreads=${threads} --bthreads=${threads} --a2bcount=${HICOUNT} --b2acount=${HICOUNT}
done
