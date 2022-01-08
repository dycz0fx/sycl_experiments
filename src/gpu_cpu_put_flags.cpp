//dpcpp -g gpu_cpu_put_flags.cpp -lpthread -lze_loader
//srun -n 1 ./a.out

#include<CL/sycl.hpp>
#include<unistd.h>
#include<thread>
#include<iostream>
#include"level_zero/ze_api.h"
#include<CL/sycl/backend/level_zero.hpp>
#include<sys/mman.h>

#define VLT volatile

constexpr size_t N = 32;
constexpr size_t T = 1024;
constexpr size_t capacity = 64;

struct packet_t {
    int64_t dest; // to be a pointer
    int64_t val;
    int64_t flag; // delete. using a separate flag array
};

struct info_t {
    size_t * last;
    size_t * first;
    packet_t * buff;
    size_t * flag;

    //DEBUG START
    size_t * tmp_counter;
    size_t * tmp_index;
    size_t * tmp_if;
    size_t * tmp_while;
    size_t * flag_tmp1;
    size_t * flag_tmp2;
    size_t * flag_tmp3;
    //DEBUG END
};

#define cpu_relax() asm volatile("rep; nop")


void add_data_to_buffer(packet_t pkt, size_t index, info_t info, sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> tmp_counter_atomic) {

    //write data to the buffer
    //sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> buff_atomic(info.buff[index%capacity]);
    //buff_atomic.store(pkt);
    //info.buff[index%capacity] = pkt; //uncached write. can we use atomic write on complex struct? we might have to split this into multiple atomic operations or use a fence

    //write data in parts for now
    int num_parts = sizeof(packet_t)/sizeof(long);
    long *buff_ptr = (long*) &(info.buff[index%capacity]);
    long *pkt_ptr = (long*) &pkt;
    for(int i=0; i<num_parts;i++){
        sycl::ext::oneapi::atomic_ref<long, sycl::memory_order::seq_cst, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> buff_atomic(buff_ptr[i]);
        buff_atomic.store(pkt_ptr[i]);
    }

    //DEBUG
    tmp_counter_atomic++;

    //sycl::ext::oneapi::atomic_ref<int64_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> buff_atomic(info.buff[0].flag);
    //buff_atomic.store(1);
    
    //set flag to 1 to denote the data is available
    sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> flag_atomic(info.flag[index%capacity]);
    flag_atomic.store(1);

    //DEBUG
    tmp_counter_atomic++;
}

void oshmem_int64_t_put(int64_t dest, int64_t val, info_t info) {
    packet_t pkt {dest, val, 1};
    
    //find an index to put the packet
    sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> last_atomic(info.last[0]);
    size_t index = last_atomic.fetch_add(1);

    //DEBUG START
    auto idx = sycl::ext::oneapi::experimental::this_nd_item<1>();
    size_t g_idx = idx.get_global_id(0);
    sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> tmp_index_atomic(info.tmp_index[g_idx]);
    tmp_index_atomic.store(index);
    //info.tmp_index[g_idx] = index;

    sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> tmp_counter_atomic(info.tmp_counter[g_idx]);
    tmp_counter_atomic.store(index);
    //DEBUG END

    //make sure there is space
    //if there is no space, wait until space becomes available
    sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::ext_intel_global_device_space> first_atomic(info.first[0]);

    size_t capacity_req = index - first_atomic.load();
    if(capacity_req < capacity) {
        add_data_to_buffer(pkt, index, info, tmp_counter_atomic);
    }

    //DEBUG
    info.tmp_if[g_idx] = g_idx;

    assert(index >= first_atomic.load());
    while(capacity_req >= capacity) { tmp_counter_atomic++;
        capacity_req = index - first_atomic.load();
        if(capacity_req < capacity) {
            add_data_to_buffer(pkt, index, info, tmp_counter_atomic);
        }
    }

    //DEBUG START
    info.tmp_while[g_idx] = 1;
    tmp_counter_atomic++;

    //info.flag_tmp1[index%capacity] = flag_atomic.load();
    //info.flag_tmp2[index%capacity] = info.flag[index%capacity];
    //info.flag_tmp3[index%capacity] = index%capacity;
    //DEBUG END
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

    size_t size = sizeof(size_t)*(1+1) + sizeof(packet_t)*capacity + sizeof(size_t)*capacity;
    size_t size_tmp = sizeof(size_t)*T*4;
    size_t size_full = size + size_tmp + sizeof(size_t)*capacity*3;
    VLT size_t * last = sycl::malloc_device<size_t>(size_full, Q);
    VLT size_t * first = last+1;
    packet_t * buff = (packet_t *)(first+1);
    VLT size_t * flag = (size_t *)(buff + capacity);
    VLT size_t * tmp_counter = (size_t *)(flag + capacity);
    VLT size_t * tmp_index = (size_t *)(tmp_counter + T);
    VLT size_t * tmp_if = (size_t *)(tmp_index + T);
    VLT size_t * tmp_while = (size_t *)(tmp_if + T);
    size_t * flag_tmp1 = (size_t*)(tmp_while + T);
    size_t * flag_tmp2 = flag_tmp1 + capacity;
    size_t * flag_tmp3 = flag_tmp2 + capacity;

    //size_t * flag_tmp1 = sycl::malloc_host<size_t>(capacity, Q);
    //size_t * flag_tmp2 = sycl::malloc_host<size_t>(capacity, Q);
    //size_t * flag_tmp3 = sycl::malloc_host<size_t>(capacity, Q);

    Q.memset((void*)last, 0, size);

    #if 0
    size_t * last = sycl::malloc_device<size_t>(2, Q);
    //size_t * first = sycl::malloc_device<size_t>(1, Q);
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
    #endif
    Q.wait();

    info_t info{(size_t*)last, (size_t*)first, buff, (size_t*)flag, (size_t*)tmp_counter, (size_t*)tmp_index, (size_t*)tmp_if, (size_t*)tmp_while, flag_tmp1, flag_tmp2, flag_tmp3};

    //create mmap for first and buff
    //size_t *last_mmap = get_mmap_address(last, sizeof(size_t)*1, Q);
    VLT size_t *last_mmap = get_mmap_address((size_t*)last, size_full, Q);
    //size_t *first_mmap = get_mmap_address(first, sizeof(size_t)*1, Q);
    VLT size_t *first_mmap = last_mmap+1;

    //packet_t *buff_mmap = get_mmap_address(buff, sizeof(packet_t)*capacity, Q);
    packet_t *buff_mmap = (packet_t *)(first_mmap+1);
    //size_t *flag_mmap = get_mmap_address(flag, sizeof(size_t)*capacity, Q);
    VLT size_t *flag_mmap = (size_t *)(buff_mmap+capacity);
    VLT size_t *tmp_counter_mmap = (size_t *)(flag_mmap+capacity);
    VLT size_t *tmp_index_mmap = (size_t *)(tmp_counter_mmap+T);
    VLT size_t *tmp_if_mmap = (size_t *)(tmp_index_mmap+T);
    VLT size_t *tmp_while_mmap = (size_t *)(tmp_if_mmap+T);

    //std::cout<<"("<<first_mmap<<" "<<first_mmap<<") ("<<last<<" "<<last_mmap<<")\n";

    std::thread host_thread = std::thread([=]() {
    //auto e_h = Q.submit([=](sycl::handler &h) {
    //    h.codeplay_host_task([=]() {
            std::cout<<"host thread: starting\n";
            fflush(stdout);

            size_t first_loc; 
            //while(true) {
            for(int n=0;n<N*T;) {
                first_loc = first_mmap[0];
                //wait for data to be added by GPU
                std::cout<<"host thread: first_loc "<<first_loc<<"\n";
                fflush(stdout);
                //while(buff_mmap[first_loc%capacity].flag == 0) { cpu_relax(); }
                size_t count_tmp = 0;
                while(flag_mmap[first_loc%capacity] == 0) { cpu_relax();
                    //DEBUG START
                    count_tmp++;
                    if(count_tmp%10000 == 0) {
                        count_tmp = 0;
                        std::string print_str = "host thread mid: ";
                        for(int i=0;i<T;i++) {
                            print_str.append("("+std::to_string(tmp_index_mmap[i])+",");
                            print_str.append(std::to_string(tmp_counter_mmap[i])+",");
                            print_str.append(std::to_string(tmp_if_mmap[i])+",");
                            print_str.append(std::to_string(tmp_while_mmap[i])+")");
                            //print_str.append(" ");
                        }
                        print_str.append("\n");
                        std::cout<<print_str;
                        fflush(stdout);
                    }
                    //DEBUG END
                }
                std::cout<<"host thread: data found\n";
                fflush(stdout);

                //for(int i=first_mmap[0]; buff_mmap[i].flag == 1; i++)
                for(int i=first_loc; flag_mmap[i%capacity] == 1; i++, n++)
                {
                    //std::cout<<"host thread: data reading at "<<first_loc<<" "<<i<<" "<<n<<"\n";
                    //fflush(stdout);

                    //read the data and release the location
                    auto data = buff_mmap[i%capacity];
                    std::cout<<"host thread: data reading at "<<first_loc<<" "<<i<<" "<<n<<" ("<<data.dest<<" , "<<data.val<<")\n";
                    fflush(stdout);
                    buff_mmap[i%capacity].flag = 0; //not needed since we have separate flag array

                    //reset flag to denote data is read
                    flag_mmap[i%capacity] = 0;

                    //increment first to denote data is read
                    first_mmap[0]= i+1;
                }
            }
            std::cout<<"Exiting host thread\n";
            fflush(stdout);
    //    });
    //});
    });

    std::cout<<"kernel going to launch\n";

    auto e = Q.submit([&](sycl::handler &h) {
        //sycl::stream os(1024, 128, h);
        //h.single_task([=]() {
    	h.parallel_for(sycl::nd_range<1>{{T}, {T}}, [=](sycl::nd_item<1> idx) {
            for(int64_t i=0;i<N;i++) {
                oshmem_int64_t_put(idx.get_global_id(0) ,i, info);
            }
        });
    });
    std::cout<<"kernel launched\n";
    e.wait_and_throw();
    std::cout<<"kernel over\n";
    fflush(stdout);

    //DEBUG START
    std::string print_str = "(index,counter,if,while): ";
    for(int i=0;i<T;i++) {
        print_str.append("("+std::to_string(tmp_index_mmap[i])+",");
        print_str.append(std::to_string(tmp_counter_mmap[i])+",");
        print_str.append(std::to_string(tmp_if_mmap[i])+",");
        print_str.append(std::to_string(tmp_while_mmap[i])+")");
        print_str.append(" ");
    }
    print_str.append("\n");
    std::cout<<print_str;
    fflush(stdout);

    #if 0
    for(int i=0;i<capacity;i++)
        std::cout<<flag_mmap[i]<<" ";
    std::cout<<"\n";
    fflush(stdout);
    for(int i=0;i<capacity;i++)
        std::cout<<info.flag_tmp1[i]<<" ";
    std::cout<<"\n";
    fflush(stdout);
    for(int i=0;i<capacity;i++)
        std::cout<<info.flag_tmp2[i]<<" ";
    std::cout<<"\n";
    fflush(stdout);
    for(int i=0;i<capacity;i++)
        std::cout<<info.flag_tmp3[i]<<" ";
    std::cout<<"\n";
    fflush(stdout);
    #endif
    //DEBUG END

    //e_h.wait_and_throw();
    host_thread.join();
    sycl::free((size_t*)last, Q);
    //sycl::free(first, Q);
    //sycl::free(buff, Q);

    //munmap(first_mmap, sizeof(size_t)*1);
    //munmap(buff_mmap, sizeof(packet_t)*capacity);
    //munmap(flag_mmap, sizeof(size_t)*capacity);
    return 0;
}

