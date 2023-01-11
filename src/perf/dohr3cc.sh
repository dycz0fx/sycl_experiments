for runs in 1 2 3 4 5; do
    srun src/perf/helperring3 --a=c --b=c --a2bbuf=c --b2abuf=c --athreads=1 --bthreads=1 --a2bcount=1000000 --b2acount=0
done
for runs in 1 2 3 4 5; do
    srun src/perf/helperring3 --a=c --b=c --a2bbuf=c --b2abuf=c --athreads=1 --bthreads=1 --a2bcount=1000000 --b2acount=1000000
done
