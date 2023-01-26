
cmake_minimum_required(VERSION 3.12)

project(GPU_CPU_Buffer VERSION 0.1)
#set(CMAKE_CXX_COMPILER dpcpp)
set(CMAKE_CXX_COMPILER icpx)
add_compile_options(-fsycl) 
add_link_options(-fsycl) 

enable_testing()

add_subdirectory(src)

