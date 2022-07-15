
link_libraries(pthread)
link_libraries(ze_loader)

add_executable(gpu_cpu_atomics gpu_cpu_atomics.cpp)
add_executable(gpu_cpu_rd_wr gpu_cpu_rd_wr.cpp)
add_executable(gpu_cpu_memcpy gpu_cpu_memcpy.cpp)
add_executable(gpu_cpu_put gpu_cpu_put.cpp)
add_executable(gpu_cpu_flow gpu_cpu_flow.cpp)
add_executable(cpu_gpu_flow cpu_gpu_flow.cpp)
add_executable(gpu_cpu_put_flags gpu_cpu_put_flags.cpp)

add_subdirectory(mmap)

set(SRUN srun)
add_test(NAME gpu_shmem_put_flags COMMAND ${SRUN} ./gpu_cpu_put_flags)
