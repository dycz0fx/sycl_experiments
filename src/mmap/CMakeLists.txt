
link_libraries(ze_loader)

add_executable(basic_mmap basic_mmap.cpp)
add_executable(cpu_gpu_mmap cpu_gpu_mmap.cpp)
add_executable(gpu_cpu_mmap gpu_cpu_mmap.cpp)

set(SRUN srun)
add_test(NAME basic_mmap_test COMMAND ${SRUN} ./basic_mmap)
add_test(NAME cpu_gpu_mmap_test COMMAND ${SRUN} ./cpu_gpu_mmap)
add_test(NAME gpu_cpu_mmap_test COMMAND ${SRUN} ./gpu_cpu_mmap)
