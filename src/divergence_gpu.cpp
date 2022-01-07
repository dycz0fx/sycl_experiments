//dpcpp -g divergence_gpu.cpp -lpthread -lze_loader

#include<CL/sycl.hpp>
#include<iostream>
#include<thread>
#include"level_zero/ze_api.h"
#include<CL/sycl/backend/level_zero.hpp>
#include<sys/mman.h>

#define cpu_relax() asm volatile("rep; nop")

constexpr size_t T = 2;

template<typename T>
T *get_mmap_address(T * device_ptr, int size, sycl::queue Q) {
    sycl::context ctx = Q.get_context();
    ze_ipc_mem_handle_t ze_ipc_handle;
    ze_result_t ret = zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx), device_ptr, &ze_ipc_handle);
    std::cout<<"zeMemGetIpcHandle return : "<<ret<<"\n";
    assert(ret == ZE_RESULT_SUCCESS);

    int fd;
    memcpy(&fd, &ze_ipc_handle, sizeof(fd));
    void *base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(base != (void *) -1);

    return (T*)base;
}

int main() {
    //queue Q(default_selector{});
    sycl::queue Q(sycl::gpu_selector{});

#if 1
    int64_t* dev_mem = sycl::malloc_device<int64_t>(3, Q);
    int64_t* dev_print = dev_mem + 1;

    Q.memset(dev_mem, 0, sizeof(int64_t)*3);
    Q.wait();

    int64_t *dev_print_mmap = get_mmap_address(dev_mem, sizeof(int64_t)*3, Q) + 1;

#else

    int64_t* dev_mem = sycl::malloc_device<int64_t>(1, Q);
    int64_t* dev_print = sycl::malloc_device<int64_t>(2, Q);

    Q.memset(dev_mem, 0, sizeof(int64_t));
    Q.memset(dev_print, 0, sizeof(int64_t)*2);
    Q.wait();

    int64_t *dev_print_mmap = get_mmap_address(dev_print, sizeof(int64_t)*2, Q);
#endif

    bool *end = new bool(false);
    std::thread host_thread = std::thread([=]() {
        std::cout<<"thread start\n";
        fflush(stdout);
        while( end[0] == false) {
            if(dev_print_mmap[1] % 10000 == 1) {
                std::cout<<dev_print_mmap[0]<<" "<<dev_print_mmap[1]<<"\n";
                fflush(stdout);
            }
            cpu_relax();
        }
        std::cout<<"thread end\n";
        fflush(stdout);
    });


    auto e = Q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{{T}, {T}}, [=](sycl::nd_item<1> idx) {
            int index = idx.get_global_id(0);
            
            //sycl::ext::oneapi::atomic_ref<int64_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> dev_print_atomic(dev_print[index]);
            //dev_print_atomic.fetch_add(1);

            //remove dev_print[index]<999999 to get infinte loop
            while(dev_mem[0] < index && dev_print[index]<999999) {
                //dev_print_atomic.fetch_add(1);
                dev_print[index] = dev_print[index]+1;
            }

            dev_mem[0] = index+1;
        });
    });


    std::cout<<"chkpt 1\n";
    fflush(stdout);
    e.wait();
    std::cout<<"chkpt 2\n";
    fflush(stdout);
    end[0] = true;

    host_thread.join();
    std::cout<<"dev_print "<<dev_print_mmap[0]<<" "<<dev_print_mmap[1]<<"\n";
    fflush(stdout);
    int val;
    Q.memcpy(&val, dev_mem, sizeof(int)).wait();
    std::cout<<"dev_mem :"<<val<<"\n";
    
    sycl::free(dev_mem, Q);
    sycl::free(dev_print, Q);
    delete end;
    return 0;
}

