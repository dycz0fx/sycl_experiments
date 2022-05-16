
if(DIR_gpu_copy_kernel)
    add_subdirectory(gpu_copy_kernel)
elseif(DIR_group_size_imbalance)
    add_subdirectory(group_size_imbalance)
else()
    add_subdirectory(gpu_copy_kernel)
    add_subdirectory(group_size_imbalance)
endif()

