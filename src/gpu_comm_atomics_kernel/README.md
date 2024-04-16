# sycl_experiments gpu_comm_atomics_kernel

Experiment with SYCL, synchronize GPUs on the same node using Xelink

### How to build

```
git clone git@github.com:srirajpaul/sycl_experiments.git
cd sycl_experiments
mkdir build
cd build
cmake .. -DDIR_gpu_comm_atomics_kernel=1
make # cmake --build .
srun ./src/gpu_comm_atomics_kernel/gpu_comm_atomics_kernel
```

