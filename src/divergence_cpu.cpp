//dpcpp -g divergence_cpu.cpp

#include<CL/sycl.hpp>
#include<iostream>

constexpr size_t T = 2;

int main() {
    //queue Q(default_selector{});
    sycl::queue Q(sycl::cpu_selector{});

    int64_t* dev_mem = sycl::malloc_device<int64_t>(1, Q);

    Q.memset(dev_mem, 0, sizeof(int64_t));
    Q.wait();

    auto e = Q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>{{T}, {T}}, [=](sycl::nd_item<1> idx) {
            int index = idx.get_global_id(0);
            
            while(dev_mem[0] < index);

            dev_mem[0] = index+1;
        });
    });


    std::cout<<"chkpt 1\n";
    fflush(stdout);
    e.wait();
    std::cout<<"chkpt 2\n";
    fflush(stdout);

    int val;
    Q.memcpy(&val, dev_mem, sizeof(int)).wait();
    std::cout<<"dev_mem :"<<val<<"\n";
    
    sycl::free(dev_mem, Q);
    return 0;
}

