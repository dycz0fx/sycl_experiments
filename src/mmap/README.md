basic_mmap

Uses default syck device.  Why does this get the GPU?  Is there an environment
variable to select the search order?

device selection is done by the Queue constructor according to the default_selector 

See https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-programming-model/device-selection.html

for an environment variable SYCL_DEVICE_FILTER

See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md

For example

SYCL_DEVICE_FILTER=host:host:0
SYCL_DEVICE_FILTER=level_zero:gpu:*


basic_mmap does not call zeInit, is that hidden inside sycl initialization?

allocates device mem size 2
allocates shared mem size 1
mmaps device mem
host inits mapped mem top 22,55

device code runs, copy the "22" to shared mem and writing 25, 74
host prints all 3 values, should be 25 74 22


cpu_gpu_mmap

last_device_mem device memory, size 2
shared_temp shared memory size N (10)
base = host mmapped copy of last_device_mem

host initializes base (last_device_mem) to 22, 55

actually this is a host_thread
   kernel runs, writing dev_loc[0] =  1 at one second intervals
Then a device kernel runs, copying last_device_mem[0] to shared_i at 50000000 count intervals


cpu_gpu_block is the same but the device kernel keeps trying until it gets the last value written by the cpu

gpu_cpu_mmap

the host thread reads and prints dev_loc[0] at 1 second intervals for 10 rounds
The device kernel stores i into dev_loc[0] at 50000000 count intervals
