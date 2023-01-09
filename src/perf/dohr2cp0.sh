for bthreads in 1 2 4 8 16 32; do
    srun src/perf/helperring2 --a=c --b=p0 --a2bbuf=p0 --b2abuf=t0 --athreads=1 --bthreads=${bthreads} --a2bcount=500000 --b2acount=0
done
for bthreads in 1 2 4 8 16 32; do
    srun src/perf/helperring2 --a=c --b=p0 --a2bbuf=p0 --b2abuf=t0 --athreads=1 --bthreads=${bthreads} --a2bcount=0 --b2acount=500000
done
for bthreads in 1 2 4 8 16 32; do
    srun src/perf/helperring2 --a=c --b=p0 --a2bbuf=p0 --b2abuf=t0 --athreads=1 --bthreads=${bthreads} --a2bcount=500000 --b2acount=500000
done
