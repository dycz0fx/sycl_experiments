for count in 10; do
    for size in  4096 0; do
	for ctl in devicectl hostctl ; do
	    for data in hostdata devicedata; do
		for dir in cputogpu gputocpu; do
		    srun -N1 ./pushpull --${dir} --hostavx --deviceloop --${data} --${ctl} --gputhreads=64 --count ${count}  --read ${size} --write ${size} --atomicload=${load} --atomicstore=${st} >> flagdata.log
		done
	    done
	done
    done
done
