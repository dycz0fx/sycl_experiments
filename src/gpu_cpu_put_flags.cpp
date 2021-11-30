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
    int flag;
};

struct info_t {
    size_t * capacity;
    size_t * last_device;
    size_t * first;
    packet_t * buff;
};

void memcpy_helper(size_t *dest, size_t *src, int count, sycl::queue Q) {
    Q.memcpy(dest, src, count).wait();
}

void oshmem_int64_t_put(int64_t dest, int64_t val, info_t info, sycl::nd_item<1> idx) {
    packet_t pkt {dest, val, 1};
    size_t capacity = info.capacity[0];
    
    //find an index to put the packet
    sycl::ONEAPI::atomic_ref<size_t, sycl::ONEAPI::memory_order::acq_rel, sycl::ONEAPI::memory_scope::device, sycl::access::address_space::global_device_space> last_device_atomic(info.last_device[0]);
    size_t index = last_device_atomic.fetch_add(1);

    //make sure there is space
    //if there is no space, wait until space becomes available
    int64_t first_required = index - capacity;
    while(info.first[0] <= first_required); // uncached read

    //write data to the buffer
    info.buff[index%capacity] = pkt; //uncached write
   }

int main() {

    sycl::queue Q;
    std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() <<"\n";
    std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() <<"\n";

    //TODO: I think using two separate malloc_device for last and last_device is causing some stuck issues
    size_t * last_device_mem = sycl::malloc_device<size_t>(2, Q);
    packet_t * buff = sycl::malloc_host<packet_t>(N*T, Q); // This can also be a usm device memory
    Q.memset(last_device_mem, 0, sizeof(size_t)*2);
    //memset buffer to 0;
    Q.wait();
    //info_t info{last_device_mem, last_device_mem+1, buff};
    info_t info;

    std::thread host_thread = std::thread([=]() {
    //auto e_h = Q.submit([=](sycl::handler &h) {
    //    h.codeplay_host_task([=]() {
            size_t capacity = info.capacity[0];
            while(true) {
                //wait for data to be added by GPU
                int64_t first_index = info.first[0];
                while(info.buff[first_index].flag == 0);

                for(int i=info.first[0]; info.buff[i].flag == 1; i++)
                {
                    //read the data and release the location
                    auto data = info.buff[i%capacity];

                    info.buff[i%capacity].flag = 0;

                    info.first[0] = i+1; // uncached write
                }
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
                oshmem_int64_t_put(idx.get_global_id(0) ,i, info, idx);

                //add a delay to interleave gpu kernel and cpu thread
                //size_t val = info.last[0];
                //for(int j=0;j<D;j++) delay+=val;
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

