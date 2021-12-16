//dpcpp -g -O0 gpu_cpu_put_flags.cpp -lpthread -lze_loader
//srun -n 1 ./a.out

#include<CL/sycl.hpp>
#include<unistd.h>
#include<thread>
#include<iostream>
#include"level_zero/ze_api.h"
#include<CL/sycl/backend/level_zero.hpp>
#include<sys/mman.h>

constexpr size_t N = 5;
constexpr size_t T = 1;
constexpr size_t capacity = 20;

struct packet_t {
    int64_t dest; // to be a pointer
    int64_t val;
    int64_t flag;
};

struct info_t {
    size_t * last;
    size_t * first;
    packet_t * buff;
    size_t * flag;
    size_t * flag_tmp1;
    size_t * flag_tmp2;
    size_t * flag_tmp3;
};

#define cpu_relax() asm volatile("rep; nop")

void oshmem_int64_t_put(int64_t dest, int64_t val, info_t info) {
    packet_t pkt {dest, val, 1};
    
    //find an index to put the packet
    sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> last_atomic(info.last[0]);
    size_t index = last_atomic.fetch_add(1);

    //make sure there is space
    //if there is no space, wait until space becomes available
    //int64_t first_required = index - capacity;
    //sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> first_atomic(info.first[0]);
    //while((int64_t)(first_atomic.load()) <= first_required);

    //write data to the buffer
    //sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> buff_atomic(info.buff[index%capacity]);
    //buff_atomic.store(pkt);
    //info.buff[index%capacity] = pkt; //uncached write. can we use atomic write on complex struct? we might have to split this into multiple atomic operations or use a fence

    //int num_parts = sizeof(packet_t)/sizeof(long);
    //long *buff_ptr = (long*) &(info.buff[index%capacity]);
    //long *pkt_ptr = (long*) &pkt;
    //for(int i=0; i<num_parts;i++){
    //    sycl::ext::oneapi::atomic_ref<long, sycl::memory_order::seq_cst, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> buff_atomic(buff_ptr[i]);
    //    buff_atomic.store(pkt_ptr[i]);
    //}

    //sycl::ext::oneapi::atomic_ref<int64_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> buff_atomic(info.buff[0].flag);
    //buff_atomic.store(1);
    
    sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> flag_atomic(info.flag[index%capacity]);
    flag_atomic.store(1);
    info.flag_tmp1[index%capacity] = flag_atomic.load();
    info.flag_tmp2[index%capacity] = info.flag[index%capacity];
    info.flag_tmp3[index%capacity] = index%capacity;
}

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

    sycl::queue Q;
    std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() <<"\n";
    std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() <<"\n";

    size_t * last = sycl::malloc_device<size_t>(1, Q);
    size_t * first = sycl::malloc_device<size_t>(1, Q);
    packet_t * buff = sycl::malloc_device<packet_t>(capacity, Q);
    size_t * flag =  sycl::malloc_device<size_t>(capacity, Q);
    size_t * flag_tmp1 =  sycl::malloc_shared<size_t>(capacity, Q);
    size_t * flag_tmp2 =  sycl::malloc_shared<size_t>(capacity, Q);
    size_t * flag_tmp3 =  sycl::malloc_shared<size_t>(capacity, Q);

    Q.memset(last, 0, sizeof(size_t));
    Q.memset(first, 0, sizeof(size_t));
    Q.memset(buff, 0, sizeof(packet_t)*capacity);
    Q.memset(flag, 0, sizeof(size_t)*capacity);
    Q.memset(flag_tmp1, 0, sizeof(size_t)*capacity);
    Q.memset(flag_tmp2, 0, sizeof(size_t)*capacity);
    Q.memset(flag_tmp3, 0, sizeof(size_t)*capacity);
    Q.wait();

    info_t info{last, first, buff, flag, flag_tmp1, flag_tmp2, flag_tmp3};

    //create mmap for first and buff
    //volatile 
    size_t *last_mmap = get_mmap_address(last, sizeof(size_t)*1, Q);
    //volatile 
    size_t *first_mmap = get_mmap_address(first, sizeof(size_t)*1, Q);
    //last_mmap[0] = 5;
    first_mmap[0] = 10;
    packet_t *buff_mmap = get_mmap_address(buff, sizeof(packet_t)*capacity, Q);
    //volatile 
    size_t *flag_mmap = get_mmap_address(flag, sizeof(size_t)*capacity, Q);

    std::cout<<"("<<first<<" "<<first_mmap<<") ("<<last<<" "<<last_mmap<<")\n";

    std::thread host_thread = std::thread([=]() {
    //auto e_h = Q.submit([=](sycl::handler &h) {
    //    h.codeplay_host_task([=]() {
            std::cout<<"host thread: starting\n";
            fflush(stdout);

#if 0
            size_t first_loc; 
            //while(true) {
            for(int n=0;n<N;) {
                first_loc = first_mmap[0];
                //wait for data to be added by GPU
                std::cout<<"host thread: first_loc "<<first_loc<<"\n";
                fflush(stdout);
                //while(buff_mmap[first_loc].flag == 0) { cpu_relax(); }
                while(flag_mmap[first_loc] == 0) { cpu_relax(); }
                std::cout<<"host thread: data found\n";
                fflush(stdout);

                //for(int i=first_mmap[0]; buff_mmap[i].flag == 1; i++)
                for(int i=first_loc; flag_mmap[i%capacity] == 1; i++, n++)
                {
                    std::cout<<"host thread: data reading at "<<first_loc<<" "<<i<<" "<<n<<"\n";
                    fflush(stdout);

                    //read the data and release the location
                    //auto data = buff_mmap[i%capacity];
                    //buff_mmap[i%capacity].flag = 0;

                    //flag_mmap[i%capacity] = 0;

                    //first_mmap[0]= i+1;
                }
            }
            std::cout<<"Exiting host thread\n";
            fflush(stdout);
#endif
    //    });
    //});
    });

    std::cout<<"kernel going to launch\n";

    auto e = Q.submit([&](sycl::handler &h) {
        //sycl::stream os(1024, 128, h);
        //h.single_task([=]() {
    	h.parallel_for(sycl::nd_range<1>{{T}, {T}}, [=](sycl::nd_item<1> idx) {
            for(int i=0;i<N;i++) {
                oshmem_int64_t_put(idx.get_global_id(0) ,i, info);
            }
        });
    });
    std::cout<<"kernel launched\n";

    e.wait_and_throw();
    std::cout<<"kernel over\n";

    for(int i=0;i<capacity;i++)
        std::cout<<flag_mmap[i]<<" ";
    std::cout<<"\n";
    for(int i=0;i<capacity;i++)
        std::cout<<info.flag_tmp1[i]<<" ";
    std::cout<<"\n";
    for(int i=0;i<capacity;i++)
        std::cout<<info.flag_tmp2[i]<<" ";
    std::cout<<"\n";
    for(int i=0;i<capacity;i++)
        std::cout<<info.flag_tmp3[i]<<" ";
    std::cout<<"\n";
    fflush(stdout);
    //e_h.wait_and_throw();
    host_thread.join();
    sycl::free(last, Q);
    sycl::free(first, Q);
    sycl::free(buff, Q);

    //munmap(first_mmap, sizeof(size_t)*1);
    //munmap(buff_mmap, sizeof(packet_t)*capacity);
    //munmap(flag_mmap, sizeof(size_t)*capacity);
    return 0;
}

