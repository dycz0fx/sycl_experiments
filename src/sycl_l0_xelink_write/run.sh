export ZE_AFFINITY_MASK=0,2,4,6

#SYCL program : 
icpx -fsycl gpu_copy_4ranks_sycl.cpp
./a.out 33554432 1
 
#OpenCL program: 
ocloc compile -file kernel.cl -output kernel -output_no_suffix -device pvc -spv_only -q -options "-cl-std=CL2.0"
icpx gpu_copy_4ranks_l0.cpp -lze_loader -lpthread
./a.out 4 33554432 1
