// icpx -fsycl sycl_copy_engine.cpp -lze_loader

#include<level_zero/ze_api.h>
#include<sycl.hpp>
#include<iostream>
#include<chrono>

sycl::queue create_sycl_queue(sycl::queue q, int ordinal, int index) {
    // TODO: should we use the parameter q or a new queue?
    sycl::device dev = q.get_device();
    sycl::context ctx = q.get_context();
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    ze_context_handle_t ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

    // Create Command Queue
    ze_command_queue_desc_t Qdescriptor = {};
    Qdescriptor.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    Qdescriptor.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    Qdescriptor.ordinal = ordinal;
    Qdescriptor.index = index;
    Qdescriptor.flags = 0;
    Qdescriptor.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    //ze_command_queue_handle_t ze_cmd_queue = nullptr;
    //ze_result_t result = zeCommandQueueCreate(ze_ctx, ze_dev, &Qdescriptor, &ze_cmd_queue);

    ze_command_list_handle_t ze_imm_cmd_list = nullptr;
    ze_result_t result = zeCommandListCreateImmediate(ze_ctx, ze_dev, &Qdescriptor, &ze_imm_cmd_list);
    if (result != ZE_RESULT_SUCCESS) {
      std::cerr << "zeCommandQueueCreate failed\n";
      return q;
    }

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::device> InteropDeviceInput{ze_dev};
    sycl::device InteropDevice =
        sycl::make_device<sycl::backend::ext_oneapi_level_zero>(InteropDeviceInput);

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context> InteropContextInput{
        ze_ctx, std::vector<sycl::device>(1, InteropDevice), sycl::ext::oneapi::level_zero::ownership::keep};
    sycl::context InteropContext =
      sycl::make_context<sycl::backend::ext_oneapi_level_zero>(InteropContextInput);

    //sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> InteropQueueInputCQ{
    //  ze_cmd_queue, InteropDevice, sycl::ext::oneapi::level_zero::ownership::keep};

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> InteropQueueInputCL{
      ze_imm_cmd_list, InteropDevice, sycl::ext::oneapi::level_zero::ownership::keep};

    //return sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(InteropQueueInputCQ, InteropContext);
    return sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(InteropQueueInputCL, InteropContext);
}

int main(int argc, char *argv[]) {
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

    const bool use_ce = 1;
    bool use_inorder = 1;
    if (argc > 1) {
        use_inorder = atoi(argv[1]);
    }

    assert(devices.size() >= 2);
    sycl::queue Q = use_inorder ? sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order{}) : sycl::queue(sycl::gpu_selector_v);
    sycl::queue Q1(devices[0]);
    sycl::queue Q2(devices[1]);
    const int N = 1*1024*1024*1024;

    assert(sizeof(int) == 4);
    int *buff1 = sycl::malloc_device<int>(N, Q1);
    int *buff2 = sycl::malloc_device<int>(N, Q2);
    Q1.memset(buff1, 0, N*sizeof(int));
    Q2.memset(buff2, 0, N*sizeof(int));
    Q1.wait();
    Q2.wait();

    if (use_ce) {
        Q1 = create_sycl_queue(Q1, 2, 0);
    }

    //warmup
    for(int n=0;n<2;n++) {
        Q1.memcpy(buff2, buff1, sizeof(int) * N).wait();
    }

    auto start = std::chrono::high_resolution_clock::now();

    const int rep = 10;
    sycl::event e;
    for(int n=0;n<rep;n++) {
        e = Q.submit([&](sycl::handler &h) {
            h.depends_on(e);
            h.parallel_for(1, [=](sycl::id<1> i) {
                buff1[i] = 0;
            });
        });

        e = Q1.submit( [=](sycl::handler &h) {
            h.depends_on(e);
            h.memcpy(buff1, buff2, N*sizeof(int));
        });

        e = Q.submit([&](sycl::handler &h) {
            h.depends_on(e);
            h.parallel_for(1, [=](sycl::id<1> i) {
                buff1[i] = 0;
            });
        });
    }

    auto end_invoke = std::chrono::high_resolution_clock::now();

    Q.wait();

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "time invoke: " << std::chrono::duration_cast<std::chrono::microseconds>(end_invoke-start).count()/rep<< " µs\n";
    std::cout << "time full  : " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/rep<< " µs\n";

    sycl::free(buff1, Q1);
    sycl::free(buff2, Q2);
    return 0;
}

