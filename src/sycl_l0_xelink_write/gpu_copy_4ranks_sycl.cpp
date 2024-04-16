
/*
    On a multi GPU system, allocate data on each GPU and copy from one to other.
    This data transfer will make use of Xelink.
*/

#include<CL/sycl.hpp>
#include<iostream>
#include<chrono>

template<typename T, sycl::access::address_space Space = sycl::access::address_space::global_space, sycl::access::decorated IsDecorated = sycl::access::decorated::yes>
sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::yes> get_multi_ptr(T *ptr) {
    return sycl::address_space_cast<Space, IsDecorated>(ptr);
}

int main(int argc, char **argv) {

    //This will list the GPU device twice, i.e. for OpenCL and Level Zero
    //for (auto device : sycl::device::get_devices(sycl::info::device_type::gpu)) {
    //    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
    //              << std::endl;
    //}

    //find all the GPUS using level zero platform
    //https://sycl.readthedocs.io/en/latest/iface/platform.html#platform-example
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> devices;

    for (auto &platform : platforms) {
        auto backend = platform.get_backend();

        //if(backend == sycl::backend::ext_oneapi_level_zero) {
        //    std::cout << "Backend: " << b << std::endl;
        //}

        //std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>()
        //      << std::endl;

        if(backend == sycl::backend::ext_oneapi_level_zero) {
            auto devices_tmp = platform.get_devices();
            for (auto &device : devices_tmp) {
                //std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                //    << std::endl;
                devices.push_back(device);
            }
        }
    }

    constexpr int dev_count = 4;

    assert(devices.size() >= dev_count);
    std::vector<sycl::queue> q_vec;

    for (int i=0; i<dev_count; i++) {
        q_vec.emplace_back(sycl::queue(devices[i]));
    }

    size_t N = 32*1024*1024;
    int write = 1;

    if (argc >= 2) {
        N = atol(argv[1]);
    }

    if (argc >= 3) {
        write = atoi(argv[2]);
    }

    assert(sizeof(int) == 4);
    int *out_buffs[dev_count], *in_buffs[dev_count];
    for (int i=0; i<dev_count; i++) {
        out_buffs[i] = sycl::malloc_device<int>(N,q_vec[i]);
        in_buffs[i] = sycl::malloc_device<int>(N,q_vec[i]);
        q_vec[i].submit([&](sycl::handler &h) {
            h.parallel_for(N, [=](sycl::id<1> idx) {
                in_buffs[i][idx] = 23;
                out_buffs[i][idx] = 24;
            });
        }).wait();
    }

    int *ptr0 = in_buffs[0];
    int *ptr1 = in_buffs[1];
    int *ptr2 = in_buffs[2];
    int *ptr3 = in_buffs[3];
    int *ptr_0 = out_buffs[0];
    int *ptr_1 = out_buffs[1];
    int *ptr_2 = out_buffs[2];
    int *ptr_3 = out_buffs[3];

    const int rep = 10;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;

    for(int n=0;n<rep+5;n++) {

        if (n == 5) {
            start = std::chrono::high_resolution_clock::now();
        }

        std::vector<sycl::event> e_vec;
        constexpr int vec_size = 1;
        constexpr int work_group_size = 32;
        const size_t kernel_size = N;

        auto e = q_vec[0].submit([=](sycl::handler &h) {
            //h.parallel_for(N, [=](sycl::id<1> idx) {
            h.parallel_for(sycl::nd_range(sycl::range{kernel_size}, sycl::range{work_group_size}), [=](sycl::nd_item<1> it) {
                {
                    const size_t idx = it.get_global_linear_id();
                    sycl::sub_group sg = it.get_sub_group();
                    const size_t sgSize = sg.get_local_range()[0];
                    int base = (idx / sgSize) * sgSize * vec_size;

                    if (write) {
                    auto sum = sg.load<vec_size>(get_multi_ptr(ptr0 + idx));
                    sg.store<vec_size>(get_multi_ptr(ptr_0 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_1 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_2 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_3 + idx), sum);
                    }
                    else {
                    auto sum0 = sg.load<vec_size>(get_multi_ptr(ptr0 + idx));
                    auto sum1 = sg.load<vec_size>(get_multi_ptr(ptr1 + idx));
                    auto sum2 = sg.load<vec_size>(get_multi_ptr(ptr2 + idx));
                    auto sum3 = sg.load<vec_size>(get_multi_ptr(ptr3 + idx));
                    auto sum = sum0 + sum1 + sum2 + sum3;
                    sg.store<vec_size>(get_multi_ptr(ptr_0 + idx), sum);
                    }

/*
                    int sum = ptr0[idx];
                    ptr_0[idx] = sum;
                    ptr_1[idx] = sum;
                    ptr_2[idx] = sum;
                    ptr_3[idx] = sum;
                    */
                }
            });
        });
        e_vec.push_back(e);

        e = q_vec[1].submit([=](sycl::handler &h) {
            //h.parallel_for(N, [=](sycl::id<1> idx) {
            h.parallel_for(sycl::nd_range(sycl::range{kernel_size}, sycl::range{work_group_size}), [=](sycl::nd_item<1> it) {
                {
                    const size_t idx = it.get_global_linear_id();
                    sycl::sub_group sg = it.get_sub_group();
                    const size_t sgSize = sg.get_local_range()[0];
                    int base = (idx / sgSize) * sgSize * vec_size;

                    if (write) {
                    auto sum = sg.load<vec_size>(get_multi_ptr(ptr1 + idx));
                    sg.store<vec_size>(get_multi_ptr(ptr_0 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_1 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_2 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_3 + idx), sum);
                    }
                    else {
                    auto sum0 = sg.load<vec_size>(get_multi_ptr(ptr0 + idx));
                    auto sum1 = sg.load<vec_size>(get_multi_ptr(ptr1 + idx));
                    auto sum2 = sg.load<vec_size>(get_multi_ptr(ptr2 + idx));
                    auto sum3 = sg.load<vec_size>(get_multi_ptr(ptr3 + idx));
                    auto sum = sum0 + sum1 + sum2 + sum3;
                    sg.store<vec_size>(get_multi_ptr(ptr_1 + idx), sum);
                    }

/*
                    int sum = ptr1[idx];
                    ptr_0[idx] = sum;
                    ptr_1[idx] = sum;
                    ptr_2[idx] = sum;
                    ptr_3[idx] = sum;
                    */
                }
            });
        });
        e_vec.push_back(e);

        e = q_vec[2].submit([=](sycl::handler &h) {
            //h.parallel_for(N, [=](sycl::id<1> idx) {
            h.parallel_for(sycl::nd_range(sycl::range{kernel_size}, sycl::range{work_group_size}), [=](sycl::nd_item<1> it) {
                {
                    const size_t idx = it.get_global_linear_id();
                    sycl::sub_group sg = it.get_sub_group();
                    const size_t sgSize = sg.get_local_range()[0];
                    int base = (idx / sgSize) * sgSize * vec_size;

                    if (write) {
                    auto sum = sg.load<vec_size>(get_multi_ptr(ptr2 + idx));
                    sg.store<vec_size>(get_multi_ptr(ptr_0 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_1 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_2 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_3 + idx), sum);
                    }
                    else {
                    auto sum0 = sg.load<vec_size>(get_multi_ptr(ptr0 + idx));
                    auto sum1 = sg.load<vec_size>(get_multi_ptr(ptr1 + idx));
                    auto sum2 = sg.load<vec_size>(get_multi_ptr(ptr2 + idx));
                    auto sum3 = sg.load<vec_size>(get_multi_ptr(ptr3 + idx));
                    auto sum = sum0 + sum1 + sum2 + sum3;
                    sg.store<vec_size>(get_multi_ptr(ptr_2 + idx), sum);
                    }

/*
                    int sum = ptr2[idx];
                    ptr_0[idx] = sum;
                    ptr_1[idx] = sum;
                    ptr_2[idx] = sum;
                    ptr_3[idx] = sum;
                    */
                }
            });
        });
        e_vec.push_back(e);

        e = q_vec[3].submit([=](sycl::handler &h) {
            //h.parallel_for(N, [=](sycl::id<1> idx) {
            h.parallel_for(sycl::nd_range(sycl::range{kernel_size}, sycl::range{work_group_size}), [=](sycl::nd_item<1> it) {
                {
                    const size_t idx = it.get_global_linear_id();
                    sycl::sub_group sg = it.get_sub_group();
                    const size_t sgSize = sg.get_local_range()[0];
                    int base = (idx / sgSize) * sgSize * vec_size;

                    if (write) {
                    auto sum = sg.load<vec_size>(get_multi_ptr(ptr3 + idx));
                    sg.store<vec_size>(get_multi_ptr(ptr_0 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_1 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_2 + idx), sum);
                    sg.store<vec_size>(get_multi_ptr(ptr_3 + idx), sum);
                    }
                    else {
                    auto sum0 = sg.load<vec_size>(get_multi_ptr(ptr0 + idx));
                    auto sum1 = sg.load<vec_size>(get_multi_ptr(ptr1 + idx));
                    auto sum2 = sg.load<vec_size>(get_multi_ptr(ptr2 + idx));
                    auto sum3 = sg.load<vec_size>(get_multi_ptr(ptr3 + idx));
                    auto sum = sum0 + sum1 + sum2 + sum3;
                    sg.store<vec_size>(get_multi_ptr(ptr_3 + idx), sum);
                    }

                /*
                    int sum = ptr3[idx];
                    ptr_0[idx] = sum;
                    ptr_1[idx] = sum;
                    ptr_2[idx] = sum;
                    ptr_3[idx] = sum;
                    */
                }
            });
        });
        e_vec.push_back(e);

        for (int i=0; i<dev_count; i++) {
            e_vec[i].wait();
        }

    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "size : "<<sizeof(int) * N;
    std::cout << "  time : " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/rep<< " µs\n";

    int* host_buff = sycl::malloc_host<int>(N, q_vec[0]);
    /*
    q_vec[0].memcpy(host_buff, in_buffs[1], N*sizeof(int)).wait();
    */
    //int *ptr = in_buffs[1];
    q_vec[0].submit([=](sycl::handler &h) {
        h.parallel_for(N, [=](sycl::id<1> idx) {
            //host_buff[idx] = ptr[idx];
            host_buff[idx] = ptr_0[idx];
        });
    }).wait();

    for(int i=0;i<N;i++) {
        if(host_buff[i] != 25) {
            //std::cout<<"Error at index : "<<i<<" with value : "<<host_buff[i]<<std::endl;
            break;
        }
    }

    return 0;
}

