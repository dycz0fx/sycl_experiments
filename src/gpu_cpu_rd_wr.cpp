#include<CL/sycl.hpp>
#include<unistd.h>
#include<thread>
#include<iostream>

constexpr size_t N = 20;
constexpr size_t T = 1;
int main() {

    sycl::queue Q;
    std::cout<<"selected device : "<<Q.get_device().get_info<sycl::info::device::name>() <<"\n";
    std::cout<<"device vendor : "<<Q.get_device().get_info<sycl::info::device::vendor>() <<"\n";

    size_t * last = sycl::malloc_host<size_t>(1, Q);
    last[0] = 0;

    auto host_thread_func = [=]() {
    //auto e_h = Q.submit([=](sycl::handler &h) {
    //    h.codeplay_host_task([=]() {
            size_t i = 0;
            size_t val_prev = 0, val_curr = 0;
            while(i<N*T) {
                //sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> last_atomic(last[0]);
                //int val_curr = last_atomic.load(sycl::memory_order::seq_cst);
                val_curr = last[0];
                assert(val_curr >= val_prev);
                while(val_prev < val_curr) {
                    std::cout<<"inside host "<<val_prev<<" "<<i<<"\n";
                    val_prev ++;
                    i++;
                }
                std::cout<<"inside host empty "<<i<<"\n";
                sleep(1);
            }
    //    });
    //});
    };

    std::thread host_thread = std::thread(host_thread_func);

    std::cout<<"kernel going to launch\n";

    auto e = Q.submit([&](sycl::handler &h) {
        sycl::stream os(1024, 128, h);
        //h.single_task([=]() {
	    h.parallel_for(sycl::nd_range<1>{{T}, {T}}, [=](sycl::nd_item<1> idx) {
            float delay = 0;
            constexpr long D = 10000000;
            //constexpr long D = 0;
            for(int i=0;i<N;i++) {
                //sycl::ext::oneapi::atomic_ref<size_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> last_atomic(last[0]);
                //size_t val = last_atomic.fetch_add(1,sycl::memory_order::seq_cst);
                size_t val = last[0];
                last[0] = val+1;

		        idx.barrier(sycl::access::fence_space::global_space);
		        //idx.barrier();

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
    sycl::free(last, Q);
    return 0;
}

