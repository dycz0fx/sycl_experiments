# sycl_experiments group_size_imbalance

Experiment with Level Zero zeKernelSuggestGroupSize API to check the group sizes returned for different global sizes.

### How to build

```
git clone git@github.com:srirajpaul/sycl_experiments.git
cd sycl_experiments
mkdir build
cd build
cmake .. -DDIR_group_size_imbalance=1
make # cmake --build .
srun ./src/group_size_imbalance/select_group_size
```

