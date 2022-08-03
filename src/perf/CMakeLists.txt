
link_libraries(ze_loader)

add_executable(host host.cpp)
add_executable(host_mmap host_mmap.cpp)

set(SRUN srun)
add_test(NAME host_test COMMAND ${SRUN} ./host)
add_test(NAME host_mmap_test COMMAND ${SRUN} ./host_mmap)
