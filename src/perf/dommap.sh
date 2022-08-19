for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
# for size in  32768 65536 131072 262144 524288; do
    echo size ${size}
    echo size ${size} >> avx.log
    srun -N1 ./host_mmap cputogpu avx 10000 0 ${size} >> avx.log
done

for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
# for size in  32768 65536 131072 262144 524288; do
    echo size ${size}
    echo size ${size} >> avx.log
    srun -N1 ./host_mmap gputocpu avx 10000 ${size} 0 >> avx.log
done

for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
# for size in  32768 65536 131072 262144 524288; do
    echo size ${size}
    echo size ${size} >> loop.log
    srun -N1 ./host_mmap cputogpu loop 10000 0 ${size} >> loop.log
done

for size in 0 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do
# for size in  32768 65536 131072 262144 524288; do
    echo size ${size}
    echo size ${size} >> loop.log
    srun -N1 ./host_mmap gputocpu loop 10000 ${size} 0 >> loop.log
done
