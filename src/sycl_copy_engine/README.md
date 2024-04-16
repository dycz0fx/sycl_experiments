# sycl_experiments sycl_copy_engine

Experiment with SYCL, using copy engine to do memcpy and measure invocation and completion time

### How to build

```
git clone git@github.com:srirajpaul/sycl_experiments.git
cd sycl_experiments
mkdir build
cd build
cmake .. -DDIR_sycl_copy_engine=1
make # cmake --build .
srun ./src/sycl_copy_engine/sycl_copy_engine
```

