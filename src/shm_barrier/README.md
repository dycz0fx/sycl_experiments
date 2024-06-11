# sycl_experiments shm_barrier

create an intra node barrier using shm and mmap

### How to build

```
git clone git@github.com:srirajpaul/sycl_experiments.git
cd sycl_experiments
mkdir build
cd build
cmake .. -DDIR_shm_barrier=1
make # cmake --build .
mpirun -prepend-rank -n 4 ./src/shm_barrier/shm_barrier
```

