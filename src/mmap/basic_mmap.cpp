#include<CL/sycl.hpp>
#include<unistd.h>
#include<thread>
#include<iostream>
#include"level_zero/ze_api.h"
#include<CL/sycl/backend/level_zero.hpp>
#include<sys/mman.h>

constexpr size_t T = 1;

int main() {
    sycl::queue Q;
    std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() <<"\n";
    std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() <<"\n";

    size_t * last_device_mem = sycl::malloc_device<size_t>(2, Q);
    int64_t *shared_temp = sycl::malloc_shared<int64_t>(1, Q);

    //create mmap mapping of usm device memory on host
    sycl::context ctx = Q.get_context();
    ze_ipc_mem_handle_t ze_ipc_handle;
    ze_result_t ret = zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx), last_device_mem, &ze_ipc_handle);
    std::cout<<"zeMemGetIpcHandle return : "<<ret<<"\n";
    assert(ret == ZE_RESULT_SUCCESS);

    int fd;
    memcpy(&fd, &ze_ipc_handle, sizeof(fd));
    void *base = mmap(NULL, sizeof(size_t)*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(base != (void *) -1);
    //mapping created

    size_t *dev_loc = (size_t*)base;
    std::cout<<last_device_mem<<" "<<dev_loc<<"\n";
    dev_loc[0] = 22;
    dev_loc[1] = 55;

    auto e = Q.submit([&](sycl::handler &h) {
        //sycl::stream os(1024, 128, h);
        //h.single_task([=]() {
        h.parallel_for(sycl::nd_range<1>{{T}, {T}}, [=](sycl::nd_item<1> idx) {
            //os<<"kernel val: "<<last_device_mem[0]<<" "<<last_device_mem[1]<<"\n";
            shared_temp[0] = last_device_mem[0];
            last_device_mem[0] = 25;
            last_device_mem[1] = 74;
        });
    });
    e.wait_and_throw();
    std::cout<<"kernel over "<<dev_loc[0]<<" "<<dev_loc[1]<<" "<<shared_temp[0]<<"\n";
    sycl::free(last_device_mem, Q);
    return 0;
}

