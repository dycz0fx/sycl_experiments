
set(CMAKE_CXX_COMPILER dpcpp)

link_libraries(ze_loader)
link_libraries(pthread)

add_executable(sycl_l0_deadlock sycl_l0_deadlock.cpp)

add_custom_command(
    TARGET sycl_l0_deadlock
    #OUTPUT kernel.spv 
    #COMMAND echo ocloc compile -file ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cl -output kernel -out_dir ${CMAKE_BINARY_DIR} -output_no_suffix -device pvc -spv_only -q -options "-cl-std=CL2.0"
    COMMAND ocloc compile -file ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cl -output kernel -out_dir ${CMAKE_BINARY_DIR} -output_no_suffix -device pvc -spv_only -q -options "-cl-std=CL2.0"
    )

