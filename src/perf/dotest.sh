count=1000
size=32768
options="--hostavx --deviceloop --devicedata --devicectl --count ${count} "
srun -N1 ./pushpull --cputogpu ${options} --read ${size} --write ${size} 
srun -N1 ./pushpull --cputogpu ${options} --read ${size} --write 0
srun -N1 ./pushpull --cputogpu ${options} --read 0       --write ${size} 
options="--hostavx --deviceloop --hostdata --devicectl --count ${count} "
srun -N1 ./pushpull --gputocpu ${options} --read ${size} --write ${size} 
srun -N1 ./pushpull --gputocpu ${options} --read ${size} --write 0
srun -N1 ./pushpull --gputocpu ${options} --read 0       --write ${size} 

