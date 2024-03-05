__kernel void test_kernel(__global int* buff, ulong count) {
    size_t thread_id = get_global_id(0);
    if(thread_id < count) {
        buff[thread_id] = thread_id;
    }
}

