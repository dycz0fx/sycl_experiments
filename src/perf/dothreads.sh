# try different numbers of GPU threads filling or emptying communications
# buffer

for count in 1000; do
    for size in 262144; do
	for threads in 1 2 4 8 16 32 64; do
	    echo size ${threads}
	    srun -N1 ./pushpull --cputogpu --hostavx --deviceloop --devicedata --devicectl --count ${count}  --read ${size} --write ${size} --gputhreads=${threads}>> threads.log
	done
    done
done

for count in 1000; do
    for size in 262144; do
	for threads in 1 2 4 8 16 32 64; do
	    echo size ${threads}
	    srun -N1 ./pushpull --gputocpu --hostavx --deviceloop --devicedata --devicectl --count ${count}  --read ${size} --write ${size} --gputhreads=${threads}>> threads.log
	done
    done
done

for count in 1000; do
    for size in 262144; do
	for threads in 1 2 4 8 16 32 64; do
	    echo size ${threads}
	    srun -N1 ./pushpull --cputogpu --hostavx --deviceloop --hostdata --devicectl --count ${count}  --read ${size} --write ${size} --gputhreads=${threads}>> threads.log
	done
    done
done

for count in 1000; do
    for size in 262144; do
	for threads in 1 2 4 8 16 32 64; do
	    echo size ${threads}
	    srun -N1 ./pushpull --gputocpu --hostavx --deviceloop --hostdata --devicectl --count ${count}  --read ${size} --write ${size} --gputhreads=${threads}>> threads.log
	done
    done
done




