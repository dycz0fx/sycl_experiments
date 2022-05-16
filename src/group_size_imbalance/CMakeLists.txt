
#set(CMAKE_CXX_COMPILER dpcpp)

link_libraries(ze_loader)

add_executable(select_group_size select_group_size.cpp)

add_custom_command(
    TARGET select_group_size
    #OUTPUT kernel.spv 
    #COMMAND echo ocloc compile -file ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cl -output kernel -out_dir ${CMAKE_BINARY_DIR} -output_no_suffix -device pvc -spv_only -q -options "-cl-std=CL2.0"
    COMMAND ocloc compile -file ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cl -output kernel -out_dir ${CMAKE_BINARY_DIR} -output_no_suffix -device pvc -spv_only -q -options "-cl-std=CL2.0"
    )

