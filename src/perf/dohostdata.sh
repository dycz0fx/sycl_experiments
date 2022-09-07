for count in 1000; do
    for size in 0 4096; do
	for ctl in hostctl devicectl splitctl; do
	    for data in hostdata devicedata; do
		for dir in cputogpu gputocpu; do
		    srun -N1 ./pushpull --${dir} --hostavx --deviceloop --${data} --${ctl} --gputhreads=64 --count ${count}  --read ${size} --write ${size} --atomicload --atomicstore >> flagdata.log

		done
	    done
	done
    done
done
