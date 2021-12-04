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

    sycl::context ctx = Q.get_context();
    ze_ipc_mem_handle_t ze_ipc_handle;
    ze_result_t ret = zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx), last_device_mem, &ze_ipc_handle);
    std::cout<<"zeMemGetIpcHandle return : "<<ret<<"\n";
    assert(ret == ZE_RESULT_SUCCESS);

    int fd;
    memcpy(&fd, &ze_ipc_handle, sizeof(fd));
    void *base = mmap(NULL, sizeof(size_t)*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(base != (void *) -1);

    size_t *dev_loc = (size_t*)base;
    std::cout<<last_device_mem<<" "<<dev_loc<<"\n";
    dev_loc[0] = 22;
    dev_loc[1] = 55;

    std::thread host_thread = std::thread([=]() {
    //auto e_h = Q.submit([=](sycl::handler &h) {
    //    h.codeplay_host_task([=]() {
            for(int i=0;i<10;i++) {
                sleep(1);
                dev_loc[0] = i;
            }
    //    });
    //});
    });

    auto e = Q.submit([&](sycl::handler &h) {
        //sycl::stream os(1024, 128, h);
        //h.single_task([=]() {
        h.parallel_for(sycl::nd_range<1>{{T}, {T}}, [=](sycl::nd_item<1> idx) {
            //os<<"kernel val: "<<last_device_mem[0]<<" "<<last_device_mem[1]<<"\n";
            double delay = 0;
            sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> last_atomic(last_device_mem[0]);
            for(int i=0;i<10;i++) {
                //shared_temp[i] = last_device_mem[0];
                shared_temp[i] = last_atomic.load();

                //add a delay to interleave gpu kernel and cpu thread
                constexpr long D = 50000000;
                for(int j=0;j<D;j++) delay+=j;
            }
            shared_temp[0] = delay;
            //os<<"kernel sum: "<<delay<<"\n";
        });
    });
    e.wait_and_throw();
    std::cout<<"kernel over\n";
    for(int i=0;i<10;i++)
        std::cout<<shared_temp[i]<<"\n";
    //e_h.wait_and_throw();
    host_thread.join();
    sycl::free(last_device_mem, Q);
    sycl::free(shared_temp, Q);
    return 0;
}

