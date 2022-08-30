for count in 1000; do

    for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
	echo size ${size}
	srun -N1 ./pushpull --cputogpu --hostavx --deviceloop --hostdata --devicectl --gputhreads=64 --count ${count}  --read ${size} --write ${size} >> hostdata.log
    done

    for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
	echo size ${size}
	srun -N1 ./pushpull --cputogpu --hostavx --deviceloop --hostdata --devicectl --gputhreads=64 --count ${count}  --read ${size} --write 0 >> hostdata.log
    done

    for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
	echo size ${size}
	srun -N1 ./pushpull --gputocpu --hostavx --deviceloop --hostdata --devicectl --gputhreads=64 --count ${count}  --read ${size} --write ${size} >> hostdata.log
    done

    for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
	echo size ${size}
	srun -N1 ./pushpull --gputocpu --hostavx --deviceloop --hostdata --devicectl --gputhreads=64 --count ${count}  --read 0 --write ${size} >> hostdata.log
    done





done
