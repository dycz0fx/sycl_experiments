# sycl_experiments sycl_l0_deadlock

Experiment with Level Zero and SYCL to show interleaving of command submission between SYCL and Level Zero can cause deadlock

### How to build

```
git clone git@github.com:srirajpaul/sycl_experiments.git
cd sycl_experiments
mkdir build
cd build
cmake .. -DDIR_sycl_l0_deadlock=1
make # cmake --build .
srun ./src/group_size_imbalance/sycl_l0_deadlock
```

