
link_libraries(ze_loader)

add_executable(host host.cpp)

set(SRUN srun)
add_test(NAME host_test COMMAND ${SRUN} ./host)
