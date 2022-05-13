# sycl_experiments

Experiment with copying data between GPUs in the same node.

### How to build

```
git clone git@github.com:srirajpaul/sycl_experiments.git
cd sycl_experiments
mkdir build
cd build
cmake .. -DDIR_gpu_copy_kernel=1
make
srun ./src/gpu_copy_kernel/gpu_copy_kernel
```

