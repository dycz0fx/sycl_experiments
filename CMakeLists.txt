
cmake_minimum_required(VERSION 3.12)

project(GPU_CPU_Buffer VERSION 0.1)

set(CMAKE_CXX_COMPILER dpcpp)

enable_testing()

add_subdirectory(src)

