for count in 1000; do

for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
    echo size ${size}
    echo size ${size} >> avx.log
    srun -N1 ./host_mmap cputogpu avx device ${count}  ${size} ${size} >> avx.log
done

for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
    echo size ${size}
    echo size ${size} >> avx.log
    srun -N1 ./host_mmap gputocpu avx device ${count} ${size} ${size} >> avx.log
done

for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
    echo size ${size}
    echo size ${size} >> avx.log
    srun -N1 ./host_mmap cputogpu avx host ${count}  ${size} ${size} >> avx.log
done

for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
    echo size ${size}
    echo size ${size} >> avx.log
    srun -N1 ./host_mmap gputocpu avx host ${count} ${size} ${size} >> avx.log
done

done
