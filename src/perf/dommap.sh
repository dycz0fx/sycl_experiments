for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
# for size in  32768 65536 131072 262144 524288; do
    echo size ${size}
    echo size ${size} >> log
    srun -N1 ./host_mmap cputogpu 10000 ${size} 0 >> log
    srun -N1 ./host_mmap cputogpu 10000 0 ${size} >> log
    srun -N1 ./host_mmap gputocpu 10000 ${size} 0 >> log
    srun -N1 ./host_mmap gputocpu 10000 0 ${size} >> log

done
