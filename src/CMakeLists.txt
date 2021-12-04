
link_libraries(pthread)

add_executable(gpu_cpu_atomics gpu_cpu_atomics.cpp)
add_executable(gpu_cpu_rd_wr gpu_cpu_rd_wr.cpp)
add_executable(gpu_cpu_memcpy gpu_cpu_memcpy.cpp)
add_executable(gpu_cpu_put gpu_cpu_put.cpp)
add_executable(gpu_cpu_flow gpu_cpu_flow.cpp)
add_executable(cpu_gpu_flow cpu_gpu_flow.cpp)

set(SRUN srun)
add_test(NAME gpu_shmem_put COMMAND ${SRUN} ./gpu_cpu_put)
