
/*
    On a multi GPU system, allocate data on each GPU and copy from one to other.
    This data transfer will make use of Xelink.
*/

#include<CL/sycl.hpp>
#include<iostream>
#include<chrono>

int main() {

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

        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>()
              << std::endl;

        if(backend == sycl::backend::ext_oneapi_level_zero) {
            auto devices_tmp = platform.get_devices();
            for (auto &device : devices_tmp) {
                std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                    << std::endl;
                devices.push_back(device);
            }
        }
    }

    //sycl::queue Q1(sycl::gpu_selector{});
    //sycl::queue Q2(sycl::gpu_selector{});
    assert(devices.size() >= 2);
    sycl::queue Q1(devices[0]);
    sycl::queue Q2(devices[1]);

    std::cout<<"selected device : "<<Q1.get_device().get_info<sycl::info::device::name>() <<"\n";
    std::cout<<"selected device : "<<Q2.get_device().get_info<sycl::info::device::name>() <<"\n";

    const int N = 1*1024*1024;

    assert(sizeof(int) == 4);
    int *buff1 = sycl::malloc_device<int>(N, Q1);
    int *buff2 = sycl::malloc_device<int>(N, Q2);
    Q1.memset(buff1, 0, N*sizeof(int));
    Q2.memset(buff2, 0, N*sizeof(int));
    Q1.wait();
    Q2.wait();

    //warmup
    for(int n=0;n<2;n++) {
    auto evt1 = Q1.submit( [&](sycl::handler &h) {
        h.parallel_for(N, [=](sycl::id<1> i) {
            buff2[i] = i;
        });
    });
    evt1.wait();
    }

    auto start = std::chrono::high_resolution_clock::now();
    //auto start = std::chrono::steady_clock::now();

    const int rep = 10;
    for(int n=0;n<rep;n++) {
    #if 1
    auto evt1 = Q1.submit( [&](sycl::handler &h) {
        h.parallel_for(N, [=](sycl::id<1> i) {
            buff1[i] = buff2[i];
            //buff2[i] = buff1[i]; //faster
        });
    });
    evt1.wait();
    #else
    Q1.memcpy(buff1, buff2, N*sizeof(int)).wait();
    //Q1.memcpy(buff2, buff1, N*sizeof(int)).wait(); //faster
    #endif
    }

    auto end = std::chrono::high_resolution_clock::now();
    //auto end = std::chrono::steady_clock::now();

    std::cout << "time : " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/rep<< " µs\n";

    int* host_buff = sycl::malloc_host<int>(N, Q1);
    Q1.memcpy(host_buff, buff1, N*sizeof(int)).wait();
    for(int i=0;i<N;i++) {
        if(host_buff[i] != i) {
            std::cout<<"Error at index : "<<i<<" with value : "<<host_buff[i]<<std::endl;
            break;
        }
    }

    sycl::free(buff1, Q1);
    sycl::free(buff2, Q2);
    sycl::free(host_buff, Q1);
    return 0;
}

