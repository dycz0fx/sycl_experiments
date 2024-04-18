#!/bin/bash

export ZE_AFFINITY_MASK=0,2,4,6

is_write=1
if [ ! -z "$1" ]
then
    is_write=$1
fi

#SYCL program :
icpx -fsycl gpu_copy_kernel_sycl.cpp
#xelink 4 ranks
ZE_AFFINITY_MASK=0,2,4,6 ./a.out 33554432 $is_write
#mdfi 2 ranks
ZE_AFFINITY_MASK=0,1 ./a.out 33554432 $is_write 2

#OpenCL program :
ocloc compile -file kernel.cl -output kernel -output_no_suffix -device pvc -spv_only -q -options "-cl-std=CL2.0"
icpx gpu_copy_kernel_l0.cpp -lze_loader -lpthread
#xelink 4 ranks
ZE_AFFINITY_MASK=0,2,4,6 ./a.out 33554432 $is_write
#mdfi 2 ranks
ZE_AFFINITY_MASK=0,1 ./a.out 33554432 $is_write 2
