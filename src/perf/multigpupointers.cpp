#include <CL/sycl.hpp>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

#define BUFSIZE (1L << 20)

// ############################

int main(int argc, char *argv[]) {
  struct timespec ts_start, ts_end;
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

  // modelled after Jeff Hammond code in PRK repo
  std::vector<sycl::queue> qs;
  
  auto platforms = sycl::platform::get_platforms();
  for (auto & p : platforms) {
    auto pname = p.get_info<sycl::info::platform::name>();
    std::cout << "*Platform: " << pname << std::endl;
    if ( pname.find("Level-Zero") == std::string::npos) {
        std::cout << "non Level Zero GPU skipped" << std::endl;
        continue;
    }
    auto devices = p.get_devices();
    for (auto & d : devices ) {
        std::cout << "**Device: " << d.get_info<sycl::info::device::name>() << std::endl;
        if ( d.is_gpu() ) {
            std::cout << "**Device is GPU - adding to vector of queues" << std::endl;
            qs.push_back(sycl::queue(d, prop_list));
        }
    }
  }


  int ngpu = qs.size();
  std::cout << "Number of GPUs found  = " << ngpu << std::endl;
    

  std::vector<uint64_t *> device_src(ngpu);
  for (int i = 0; i < ngpu; i += 1) {
    device_src[i] = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, qs[i]);
    std::cout << " device_src[" << i << "] " << device_src[i] << std::endl;
  }
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0 ; j < ngpu; j += 1) {
      uint64_t *p = device_src[j];
      qs[i].submit([&](sycl::handler &h) {
	  auto out = sycl::stream(1024, 768, h);
	  auto task = 
	    [=]() {
	    out << "gpu " << i << " device_src[" << j << "] = " << p << "\n";
	  };
	h.single_task(task);
      }).wait();
    }
  }


  return (0);
    

}


