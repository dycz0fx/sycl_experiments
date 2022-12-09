#include<CL/sycl.hpp>
#include<unistd.h>
#include<thread>
#include<iostream>

//TODO: gets stuck after when N>70. Dave's suggestion is to try with bigger sycl::stream
constexpr size_t N = 20;
constexpr size_t T = 10;

struct packet_t {
    int64_t dest; // to be a pointer
    int64_t val;
};

struct info_t {
    size_t * last_device;
    size_t * last;
    packet_t * buff;
};

void memcpy_helper(size_t *dest, size_t *src, int count, sycl::queue Q) {
    Q.memcpy(dest, src, count).wait();
}

void oshmem_int64_t_put_nbi(int64_t dest, int64_t val, info_t info, sycl::nd_item<1> idx) {
    packet_t pkt {dest, val};
    
    //TODO: For now, assume buffer is infinte and so never full

    //TODO: Understand more about memory_order, memory_scope and address_space

    //find an index to put the packet
    sycl::atomic_ref<size_t, sycl::memory_order::seq_cst, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> last_device_atomic(info.last_device[0]);
    size_t index = last_device_atomic.fetch_add(1, sycl::memory_order::seq_cst);
   
    info.buff[index] = pkt;

    //TODO: get idx using dpcpp free function extensions rather than as argument
    idx.barrier(sycl::access::fence_space::global_space);
    //idx.barrier();

    //wait until the all data upto index is added so that there are no holes in the buffer
    sycl::atomic_ref<size_t, sycl::memory_order::seq_cst, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> last_atomic(info.last[0]);
    size_t expected = index;
    while(!last_atomic.compare_exchange_strong(expected, index+1, sycl::memory_order::seq_cst, sycl::memory_order::seq_cst)) {
        expected = index;
    }
}

int main() {

    sycl::queue Q;
    std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() <<"\n";
    std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() <<"\n";

    //TODO: I think using two separate malloc_device for last and last_device is causing some stuck issues
    size_t * last_device_mem = sycl::malloc_device<size_t>(2, Q);
    packet_t * buff = sycl::malloc_host<packet_t>(N*T, Q); // This can also be a usm device memory
    Q.memset(last_device_mem, 0, sizeof(size_t)*2);
    Q.wait();
    info_t info{last_device_mem, last_device_mem+1, buff};

    std::thread host_thread = std::thread([=]() {
    //auto e_h = Q.submit([=](sycl::handler &h) {
    //    h.codeplay_host_task([=]() {
            size_t i = 0;
            size_t val_prev = 0, val_curr = 0;
            while(i<N*T) {
                memcpy_helper(&val_curr, info.last, sizeof(size_t), Q);
                //val_curr = info.last[0]; // directly read instead of memcpy while using usm shared or host
                assert(val_curr >= val_prev);
                while(val_prev < val_curr) {
                    std::cout<<"inside host "<<val_prev<<" "<<i<<" buffer tid:val "<<info.buff[val_prev].dest<<":"<<info.buff[val_prev].val<<"\n";
                    val_prev ++;
                    i++;
                }
                std::cout<<"inside host empty "<<i<<"\n";
                sleep(1);
            }
    //    });
    //});
    });

    std::cout<<"kernel going to launch\n";

    auto e = Q.submit([&](sycl::handler &h) {
        sycl::stream os(1024, 128, h);
        //h.single_task([=]() {
    	h.parallel_for(sycl::nd_range<1>{{T}, {T}}, [=](sycl::nd_item<1> idx) {
            double delay = 0;
            constexpr long D = 10000000;
            //constexpr long D = 0;
            for(int i=0;i<N;i++) {
                oshmem_int64_t_put_nbi(idx.get_global_id(0) ,i, info, idx);

                //add a delay to interleave gpu kernel and cpu thread
                size_t val = info.last[0];
                for(int j=0;j<D;j++) delay+=val;
            }
            os<<"kernel sum: "<<delay<<"\n";
        });
    });
    std::cout<<"kernel launched\n";

    e.wait_and_throw();
    std::cout<<"kernel over\n";
    //e_h.wait_and_throw();
    host_thread.join();
    sycl::free(last_device_mem, Q);
    sycl::free(buff, Q);
    return 0;
}

