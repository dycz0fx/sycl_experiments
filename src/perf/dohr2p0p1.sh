for threads in 1 2 4 8 16 32; do
    srun src/perf/helperring2 --a=p0 --b=p1 --a2bbuf=p1 --b2abuf=p0 --athreads=${threads} --bthreads=${threads} --a2bcount=500000 --b2acount=0
done
for threads in 1 2 4 8 16 32; do
    srun src/perf/helperring2 --a=p0 --b=p1 --a2bbuf=p1 --b2abuf=p0 --athreads=${threads} --bthreads=${threads} --a2bcount=500000 --b2acount=500000
done
