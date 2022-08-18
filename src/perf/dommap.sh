for size in 0 32 64 128 256 512 1024 2048 4096 8192 16384; do
    echo size ${size}
    echo size ${size} >> log
    srun --reservation=stewartl_4143 ./host_mmap 10000 ${size} 0 >> log
    srun --reservation=stewartl_4143 ./host_mmap 10000 0 ${size} >> log
    srun --reservation=stewartl_4143 ./host_mmap 10000 ${size} ${size} >> log
done
