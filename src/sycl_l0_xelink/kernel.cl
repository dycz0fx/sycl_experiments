/*
__kernel void reduce_scatter_kernel_2( \
    const __global int* in_buf_0,
    const __global int* in_buf_1,
    __global int* out_buf) {
    const size_t thread_id = get_global_id(0);

    int sum = in_buf_0[thread_id];
    sum += in_buf_1[thread_id];

    out_buf[thread_id] = sum;
}

__kernel void reduce_scatter_kernel_3( \
    const __global int* in_buf_0,
    const __global int* in_buf_1,
    const __global int* in_buf_2,
    __global int* out_buf) {
    const size_t thread_id = get_global_id(0);

    int sum = in_buf_0[thread_id];
    sum += in_buf_1[thread_id];
    sum += in_buf_2[thread_id];

    out_buf[thread_id] = sum;
}
*/

__kernel void reduce_scatter_kernel_4( \
    __global long* in_buf_0,
    __global long* in_buf_1,
    __global long* in_buf_2,
    __global long* in_buf_3,
    __global long* out_buf,
    const long count,
    const int is_write) {
    const size_t thread_id = get_global_id(0);

    /*
        const size_t work_group_size = get_global_size(0);
        const size_t idx = get_global_id(0);
        const size_t subgroup_size = get_sub_group_size();
        const size_t subgroup_idx = idx / subgroup_size * subgroup_size;

    intel_sub_group_block_write_ui(&in_buf_0[subgroup_idx], out_buf[0 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_2[subgroup_idx], out_buf[1 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_2[subgroup_idx], out_buf[2 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_3[subgroup_idx], out_buf[3 * count + thread_id]);
    */

    if (is_write) {
    in_buf_0[thread_id] = out_buf[0 * count + thread_id];
    in_buf_1[thread_id] = out_buf[1 * count + thread_id];
    in_buf_2[thread_id] = out_buf[2 * count + thread_id];
    in_buf_3[thread_id] = out_buf[3 * count + thread_id];
    }
    else {
    out_buf[0 * count + thread_id] = in_buf_0[thread_id];
    out_buf[1 * count + thread_id] = in_buf_1[thread_id];
    out_buf[2 * count + thread_id] = in_buf_2[thread_id];
    out_buf[3 * count + thread_id] = in_buf_3[thread_id];
    }
}

/*
__kernel void reduce_scatter_kernel_5( \
    const __global int* in_buf_0,
    const __global int* in_buf_1,
    const __global int* in_buf_2,
    const __global int* in_buf_3,
    const __global int* in_buf_4,
    __global int* out_buf) {
    const size_t thread_id = get_global_id(0);

    int sum = in_buf_0[thread_id];
    sum += in_buf_1[thread_id];
    sum += in_buf_2[thread_id];
    sum += in_buf_3[thread_id];
    sum += in_buf_4[thread_id];

    out_buf[thread_id] = sum;
}
*/

__kernel void reduce_scatter_kernel_6( \
    __global int* in_buf_0,
    __global int* in_buf_1,
    __global int* in_buf_2,
    __global int* in_buf_3,
    __global int* in_buf_4,
    __global int* in_buf_5,
    __global int* out_buf,
    const long count,
    const int is_write) {
    const size_t thread_id = get_global_id(0);

/*
    int sum = in_buf_0[thread_id];
    sum += in_buf_1[thread_id];
    sum += in_buf_2[thread_id];
    sum += in_buf_3[thread_id];
    sum += in_buf_4[thread_id];
    sum += in_buf_5[thread_id];

    out_buf[thread_id] = sum;
*/
    /*
        const size_t work_group_size = get_global_size(0);
        const size_t idx = get_global_id(0);
        const size_t subgroup_size = get_sub_group_size();
        const size_t subgroup_idx = idx / subgroup_size * subgroup_size;

    intel_sub_group_block_write_ui(&in_buf_0[subgroup_idx], out_buf[0 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_2[subgroup_idx], out_buf[1 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_2[subgroup_idx], out_buf[2 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_3[subgroup_idx], out_buf[3 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_4[subgroup_idx], out_buf[4 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_5[subgroup_idx], out_buf[5 * count + thread_id]);
    */

    if (is_write) {
    in_buf_0[thread_id] = out_buf[0 * count + thread_id];
    in_buf_1[thread_id] = out_buf[1 * count + thread_id];
    in_buf_2[thread_id] = out_buf[2 * count + thread_id];
    in_buf_3[thread_id] = out_buf[3 * count + thread_id];
    in_buf_4[thread_id] = out_buf[4 * count + thread_id];
    in_buf_5[thread_id] = out_buf[5 * count + thread_id];
    }
    else {
    out_buf[0 * count + thread_id] = in_buf_0[thread_id];
    out_buf[1 * count + thread_id] = in_buf_1[thread_id];
    out_buf[2 * count + thread_id] = in_buf_2[thread_id];
    out_buf[3 * count + thread_id] = in_buf_3[thread_id];
    out_buf[4 * count + thread_id] = in_buf_4[thread_id];
    out_buf[5 * count + thread_id] = in_buf_5[thread_id];
    }

}

__kernel void reduce_scatter_kernel_8( \
    __global int* in_buf_0,
    __global int* in_buf_1,
    __global int* in_buf_2,
    __global int* in_buf_3,
    __global int* in_buf_4,
    __global int* in_buf_5,
    __global int* in_buf_6,
    __global int* in_buf_7,
    __global int* out_buf,
    const long count,
    const int is_write) {
    const size_t thread_id = get_global_id(0);

    /*
        const size_t work_group_size = get_global_size(0);
        const size_t idx = get_global_id(0);
        const size_t subgroup_size = get_sub_group_size();
        const size_t subgroup_idx = idx / subgroup_size * subgroup_size;

    intel_sub_group_block_write_ui(&in_buf_0[subgroup_idx], out_buf[0 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_2[subgroup_idx], out_buf[1 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_2[subgroup_idx], out_buf[2 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_3[subgroup_idx], out_buf[3 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_4[subgroup_idx], out_buf[4 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_5[subgroup_idx], out_buf[5 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_6[subgroup_idx], out_buf[6 * count + thread_id]);
    intel_sub_group_block_write_ui(&in_buf_7[subgroup_idx], out_buf[7 * count + thread_id]);
    */

    if (is_write) {
    in_buf_0[thread_id] = out_buf[0 * count + thread_id];
    in_buf_1[thread_id] = out_buf[1 * count + thread_id];
    in_buf_2[thread_id] = out_buf[2 * count + thread_id];
    in_buf_3[thread_id] = out_buf[3 * count + thread_id];
    in_buf_4[thread_id] = out_buf[4 * count + thread_id];
    in_buf_5[thread_id] = out_buf[5 * count + thread_id];
    in_buf_6[thread_id] = out_buf[6 * count + thread_id];
    in_buf_7[thread_id] = out_buf[7 * count + thread_id];
    }
    else {
    out_buf[0 * count + thread_id] = in_buf_0[thread_id];
    out_buf[1 * count + thread_id] = in_buf_1[thread_id];
    out_buf[2 * count + thread_id] = in_buf_2[thread_id];
    out_buf[3 * count + thread_id] = in_buf_3[thread_id];
    out_buf[4 * count + thread_id] = in_buf_4[thread_id];
    out_buf[5 * count + thread_id] = in_buf_5[thread_id];
    out_buf[6 * count + thread_id] = in_buf_6[thread_id];
    out_buf[7 * count + thread_id] = in_buf_7[thread_id];
    }
}

