find_package(MKL CONFIG REQUIRED PATHS
/opt/intel/inteloneapi/tbb/latest/lib/cmake/tbb
/opt/intel/inteloneapi/mkl/latest/lib/cmake/mkl)
add_executable(fft fft.cpp) 
add_executable(fft2d fft2d.cpp) 
add_executable(dp_complex_2d dp_complex_2d.cpp) 

target_compile_options(fft PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(fft PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(fft PUBLIC $<LINK_ONLY:MKL::MKL>)

target_compile_options(fft2d PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(fft2d PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(fft2d PUBLIC $<LINK_ONLY:MKL::MKL>)

target_compile_options(dp_complex_2d PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(dp_complex_2d PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(dp_complex_2d PUBLIC $<LINK_ONLY:MKL::MKL>)
