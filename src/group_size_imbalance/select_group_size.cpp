// ocloc compile -file kernel.cl -output kernel -output_no_suffix -device pvc -spv_only -q -options "-cl-std=CL2.0"
// dpcpp select_group_size.cpp -lze_loader

#include<iostream>
#include<fstream>
#include<vector>
#include<chrono>
#include<assert.h>
#include<level_zero/ze_api.h>

#define ZE_CALL(ze_name, ze_args) do { ze_call(ze_name ze_args, #ze_name); } while(0)
/*
#define ZE_CALL(ze_name, ze_args) do {  \
    ze_result_t ze_result = ze_name ze_args; \
     if(ZE_RESULT_SUCCESS != ze_result)  { \
       std::cerr<<"Got Error at "<<ze_name<<" with return value "<<ze_result<<"\n"; \
       std::terminate(); \
    }\
    ze_result;\
}while(0)
*/
ze_result_t ze_call(ze_result_t ze_result, const char* ze_name) {
    if(ZE_RESULT_SUCCESS != ze_result)  {
       std::cerr<<"Got Error at "<<ze_name<<" with return value "<<ze_result<<"\n";
       std::terminate();
    }
    return ze_result;
}

std::vector<ze_device_handle_t> devices;
ze_context_handle_t context;
std::vector<ze_command_list_handle_t> cmd_lists;
std::vector<ze_command_queue_handle_t> cmd_queues;
std::vector<ze_module_handle_t> modules;
std::vector<ze_kernel_handle_t> kernels;
ze_event_pool_handle_t event_pool;
std::vector<ze_event_handle_t> events;
uint32_t event_index = 0;

ze_module_handle_t loadModule(std::string filepath, ze_device_handle_t device, ze_context_handle_t context) {

    std::ifstream file(filepath, std::ios_base::in | std::ios_base::binary);
    assert(file.good());

    file.seekg(0, file.end);
    size_t filesize = file.tellg();
    file.seekg(0, file.beg);

    std::vector<uint8_t> module_data(filesize);
    file.read(reinterpret_cast<char*>(module_data.data()), filesize);
    file.close();

    ze_module_desc_t module_desc {
        .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
        .pNext = nullptr,
        .format = ZE_MODULE_FORMAT_IL_SPIRV,
        .inputSize = module_data.size(),
        .pInputModule = reinterpret_cast<const uint8_t*>(module_data.data()),
        .pBuildFlags = nullptr,
        .pConstants = nullptr
    };
    ze_module_build_log_handle_t build_log;
    ze_module_handle_t module;
    ze_result_t result = zeModuleCreate(context, device, &module_desc, &module, &build_log);
    if(ZE_RESULT_SUCCESS != result) {
        size_t log_size;
        ZE_CALL(zeModuleBuildLogGetString, (build_log, &log_size, nullptr));
        std::string log(log_size, '\0');
        ZE_CALL(zeModuleBuildLogGetString, (build_log, &log_size, const_cast<char*>(log.data())));
        std::cerr<<"Failed to load Module with error : "<<log<<"\n";
        std::terminate();
    }

    return module;
}

void init_ze() {
    // Initialize the driver
    ze_init_flag_t flags = ZE_INIT_FLAG_GPU_ONLY;
    zeInit(flags);

    // Discover all the driver instances
    uint32_t driver_count = 0;
    ZE_CALL(zeDriverGet, (&driver_count, nullptr));
    std::vector<ze_driver_handle_t> drivers(driver_count);
    ZE_CALL(zeDriverGet, (&driver_count, drivers.data()));
    ze_driver_handle_t driver = drivers[0];

    // Find driver instances with a GPU device
    uint32_t device_count = 0;
    zeDeviceGet(driver, &device_count, nullptr);
    devices.resize(device_count);
    zeDeviceGet(driver, &device_count, devices.data());

    ze_device_properties_t device_properties;
    zeDeviceGetProperties(devices[0], &device_properties);
    assert(ZE_DEVICE_TYPE_GPU == device_properties.type);

    // Create the context
    ze_context_desc_t context_desc = {
        .stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC,
        .pNext = nullptr,
        .flags = 0
    };
    zeContextCreate(driver, &context_desc, &context);

    //create command queues and command lists
    for(ze_device_handle_t& device: devices) {

        //create compute command queue i.e ordinal=0
        ze_command_queue_desc_t queue_desc = {
            .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
            .pNext = NULL,
            .ordinal = 0, //use ordinal zero for compute queue
            .index = 0,
            .flags = 0,
            .mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
            .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
        };
        ze_command_queue_handle_t queue;
        zeCommandQueueCreate(context, device, &queue_desc, &queue);
        cmd_queues.push_back(queue);

        //create command list
        ze_command_list_desc_t list_desc = {
            .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
            .pNext = nullptr,
            .commandQueueGroupOrdinal = 0, //use zero for the compute command queue
            .flags = 0
        };
        ze_command_list_handle_t list;
        zeCommandListCreate(context, device, &list_desc, &list);
        cmd_lists.push_back(list);

        //load the opencl kernel from file
        ze_module_handle_t module = loadModule("kernel.spv", device, context);
        modules.push_back(module);
    
        ze_kernel_desc_t kernel_desc{
            .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
            .pNext = nullptr,
            .flags = 0,
            .pKernelName = "test_kernel" 
        };
        ze_kernel_handle_t kernel;
        zeKernelCreate(module, &kernel_desc, &kernel);
        kernels.push_back(kernel);
    }

    //create event pool
    ze_event_pool_desc_t pool_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        .pNext = nullptr,
        .flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP,
        .count = 10
    };
    zeEventPoolCreate(context, &pool_desc, 0, nullptr, &event_pool);
}

void finalize_ze() {
    for(ze_event_handle_t &event: events)
        zeEventDestroy(event);
    zeEventPoolDestroy(event_pool);
    for(int i=0;i<devices.size();i++) {
        zeKernelDestroy(kernels[i]);
        zeModuleDestroy(modules[i]);
        zeCommandListDestroy(cmd_lists[i]);
        zeCommandQueueDestroy(cmd_queues[i]);
       
    }
    zeContextDestroy(context);
}

using namespace std::chrono;

duration<long double, std::micro> run_kernel(ze_kernel_handle_t kernel, int idx, void* buff, size_t count, bool print) {

    uint32_t groupSizeX = 1u;
    uint32_t groupSizeY = 1u;
    uint32_t groupSizeZ = 1u;
    ze_group_count_t groupCount{1u, 1u, 1u};

    zeKernelSuggestGroupSize(kernel, count, 1u, 1u, &groupSizeX, &groupSizeY, &groupSizeZ);
    groupCount.groupCountX = count / groupSizeX;

    if(print)
        std::cout<<"idx: "<<idx<<" count: "<<count<<" group size: "<<groupSizeX<<" group count: "<<groupCount.groupCountX<<"\n";

    //setup arguments for the opencl kernel
    zeKernelSetArgumentValue(kernel, 0, sizeof(buff), &buff);
    zeKernelSetArgumentValue(kernel, 1, sizeof(count), &count);

    //create an output event for the kernel so that host can synchronize for completion
    ze_event_desc_t event_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
        .pNext = nullptr,
        .index = event_index++,
        .signal = ZE_EVENT_SCOPE_FLAG_DEVICE,
        .wait = ZE_EVENT_SCOPE_FLAG_DEVICE
    };
    ze_event_handle_t event;
    zeEventCreate(event_pool, &event_desc, &event);

    //Add the kernel to the command list
    zeCommandListAppendLaunchKernel(cmd_lists[0], kernel, &groupCount, event, 0, nullptr);

    //Close the command list for submission
    //Multiple operations/kernels can be added to the command list before closing it
    zeCommandListClose(cmd_lists[0]);

    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    //submit the kernel for execution on the device
    zeCommandQueueExecuteCommandLists(cmd_queues[0], 1, &(cmd_lists[0]), nullptr);

    //wait for the kernel to finish
    zeEventHostSynchronize(event, std::numeric_limits<uint64_t>::max());

    high_resolution_clock::time_point end_time = high_resolution_clock::now();

    //reset the event back to not signaled state
    zeEventHostReset(event);
    zeCommandListReset(cmd_lists[0]);
    
    //collect events to be destroyed later
    events.push_back(event);

    duration<long double, std::micro> total_time = end_time - start_time;
    return total_time;
}

int main() {
    init_ze();

    constexpr int ranks = 2;
    constexpr size_t buff_count = 11184815;
    constexpr size_t buff_bytes = buff_count * sizeof(int);
    void* dev_buff = nullptr, * host_buff = nullptr;

    //using only GPU 0
    ze_device_handle_t device = devices[0];
    ze_kernel_handle_t kernel = kernels[0];

    //allocate device memory
    ze_device_mem_alloc_desc_t dev_buff_alloc_desc {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        .pNext = nullptr,
        .flags = 0, 
        .ordinal = 0
    };
    //ZE_CALL(zeMemAllocDevice, (context, &dev_buff_alloc_desc, buff_bytes, 0, device, &dev_buff));
    zeMemAllocDevice(context, &dev_buff_alloc_desc, buff_bytes, 0, device, &dev_buff);

    //allocate host memory
    ze_host_mem_alloc_desc_t host_buff_alloc_desc {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        .pNext = nullptr,
        .flags = 0
    };
    zeMemAllocHost(context, &host_buff_alloc_desc, buff_bytes, 0, &host_buff);

    //calcuate buffer count in each rank
    size_t buff_count_rank_1 = buff_count/ranks;
    size_t buff_count_rank_2 = buff_count - buff_count_rank_1;

    //warmup
    run_kernel(kernel, 1, dev_buff, buff_count_rank_1, false);
    run_kernel(kernel, 2, (int*)dev_buff+buff_count_rank_1, buff_count_rank_2, false);

    //timing run
    auto time1 = run_kernel(kernel, 1, dev_buff, buff_count_rank_1, true);
    auto time2 = run_kernel(kernel, 2, (int*)dev_buff+buff_count_rank_1, buff_count_rank_2, true);

    //copy data back to host for validation
    zeCommandListAppendMemoryCopy(cmd_lists[0], host_buff, dev_buff, buff_bytes, nullptr, 0, nullptr);
    zeCommandListClose(cmd_lists[0]);
    zeCommandQueueExecuteCommandLists(cmd_queues[0], 1, &(cmd_lists[0]), nullptr);
    zeCommandQueueSynchronize(cmd_queues[0], std::numeric_limits<uint64_t>::max());
    
    //validate_buffer
    for(size_t i=0; i<buff_count; i++) {
        int expected = i;
        //2nd kernel restarts the written value from zero onwards
        if(i>=buff_count_rank_1)
            expected -= buff_count_rank_1;
        if(((int*)host_buff)[i] != expected) {
            std::cerr<<"Error at location "<<i<<", expected "<<expected<<", actual "<<((int*)host_buff)[i]<<"\n";
            break;
        }
    }

    std::cout<<"Execution 1 time : "<<time1.count()<<"\n";
    std::cout<<"Execution 2 time : "<<time2.count()<<"\n";

    zeMemFree(context, host_buff);
    zeMemFree(context, dev_buff);
    finalize_ze();

    return 0;
}
