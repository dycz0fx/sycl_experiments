
set(CMAKE_CXX_COMPILER mpicc)

link_libraries(rt)
link_libraries(pthread)

add_executable(shm_barrier shm_barrier.cpp)

