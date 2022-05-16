# sycl_experiments

Experiment with SYCL and Level Zero

### How to build

```
git clone git@github.com:srirajpaul/sycl_experiments.git
cd sycl_experiments
mkdir build
cd build
cmake ..
make # cmake --build .
srun ./src/gpu_copy_kernel/gpu_copy_kernel
#make clean # cmake --build . --target clean
```

