
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
link_libraries(ze_loader)

add_executable(sycl_copy_engine sycl_copy_engine.cpp)

