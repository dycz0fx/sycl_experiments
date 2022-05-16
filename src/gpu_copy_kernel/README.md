# sycl_experiments gpu_copy_kernel

Experiment with SYCL, to copy data between GPUs on the same node using Xelink

### How to build

```
git clone git@github.com:srirajpaul/sycl_experiments.git
cd sycl_experiments
mkdir build
cd build
cmake .. -DDIR_gpu_copy_kernel=1
make # cmake --build .
srun ./src/gpu_copy_kernel/gpu_copy_kernel
```

