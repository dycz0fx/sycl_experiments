// queuetest.cpp
#include <CL/sycl.hpp>

// for mmap
#include <sys/mman.h>
#include "level_zero/ze_api.h"
#include <sys/stat.h>

// in order to use placement new
#include <new>   

// general utility
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h>

#include "ringlib3.cpp"

constexpr double NSEC_IN_SEC = 1000000000.0;

void printduration(const char* name, sycl::event e)
  {
    uint64_t start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
    std::cout << name << " execution time: " << duration << " sec" << std::endl;
  }


template<typename T>
T *get_mmap_address(T * device_ptr, size_t size, sycl::queue Q) {
    sycl::context ctx = Q.get_context();
    ze_ipc_mem_handle_t ze_ipc_handle;
    ze_result_t ret = zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx), device_ptr, &ze_ipc_handle);
    std::cout<<"zeMemGetIpcHandle return : " << ret << std::endl;
    assert(ret == ZE_RESULT_SUCCESS);
    int fd;
    memcpy(&fd, &ze_ipc_handle, sizeof(fd));
    std::cout << " fd " << fd << std::endl;
    struct stat statbuf;
    fstat(fd, &statbuf);
    std::cout << "requested size " << size << std::endl;
    std::cout << "fd size " << statbuf.st_size << std::endl;
    void *base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == (void *) -1) {
      std::cout << "mmap returned -1" << std::endl;
      std::cout << strerror(errno) << std::endl;  
    }
    assert(base != (void *) -1);
    return (T*)base;
}

void printdatastructures(const char *name, GPURing *r, RingMessage *send, RingMessage *receive)
{
  printf("%s GPURing at %p\n", name, r);
  printf("%s next_peer_receive %d\n", name, receive[RingN].sequence);
  r->Print(name);
  int low = r->receive_count - 10;
  int high = r->receive_count + 10;
  if (low < 0) low = 0;
  for (int i = low; i <= high; i += 1) {
    printf("%s recvbuf[%d] = %d, %ld, %ld\n", name, i, receive[i%RingN].sequence, receive[i%RingN].data[1], receive[i%RingN].data[2]);
  }
  low = r->next_send - 10;
  high = r->next_send + 10;
  if (low < 0) low = 0;
  for (int i = low; i <= high; i += 1) {
    printf("%s sendbuf[%d] = %d, %ld, %ld\n", name, i, send[i%RingN].sequence, send[i%RingN].data[1], send[i%RingN].data[2]);
  }
}

int main(int argc, char *argv[]) {
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

  //sycl::queue qa1 = sycl::queue(sycl::gpu_selector_v, prop_list);
  //sycl::queue qb1 = sycl::queue(sycl::gpu_selector_v, prop_list);
  std::vector<sycl::queue> pvcqa;
  std::vector<sycl::queue> tileqa;
  std::vector<sycl::queue> pvcqb;
  std::vector<sycl::queue> tileqb;
  int qcount = 0;
  auto platforms = sycl::platform::get_platforms();
  for (auto & p : platforms) {
    auto pname = p.get_info<sycl::info::platform::name>();
    //std::cout << "*Platform: " << pname << std::endl;
    if ( pname.find("Level-Zero") == std::string::npos) {
      //std::cout << "non Level Zero GPU skipped" << std::endl;
        continue;
    }
    auto devices = p.get_devices();
    //std::cout << "number of devices: " << devices.size() << std::endl;
    for (auto & d : devices ) {
      //std::cout << "**Device: " << d.get_info<sycl::info::device::name>() << std::endl;
      if (d.is_gpu()) {
	if (qcount < 2) {
	  pvcqa.push_back(sycl::queue(d, prop_list));
	  pvcqb.push_back(sycl::queue(d, prop_list));
	  //std::cout << "create pvcq[" << pvcq.size() - 1 << "]" << std::endl;
	  //std::cout << "**max wg: " << d.get_info<cl::sycl::info::device::max_work_group_size>()  << std::endl;
	  qcount += 1;
	} else {
	  std::vector<sycl::device> sd = d.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::next_partitionable);
	  //std::cout << " subdevices " << sd.size() << std::endl;
	  for (auto &subd: sd) {
	    tileqa.push_back(sycl::queue(subd, prop_list));
	    tileqb.push_back(sycl::queue(subd, prop_list));
	    //std::cout << "create tileq[" << tileq.size() - 1 << "]" << std::endl;
	    //std::cout << "**max wg: " << subd.get_info<cl::sycl::info::device::max_work_group_size>()  << std::endl;
	    qcount += 1;
	  }
	}
	if (qcount >= 4) break;
      }
      if (qcount >= 4) break;
    }
  }
  sycl::queue qa1 = pvcqa[0];
  sycl::queue qb1 = pvcqb[0];
  
  std::cout<<"selected device : "<<qa1.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<qa1.get_device().get_info<sycl::info::device::vendor>() << std::endl;
  std::cout<<"selected device : "<<qb1.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<qb1.get_device().get_info<sycl::info::device::vendor>() << std::endl;

  GPURing *ringa = sycl::aligned_alloc_device<GPURing>(4096, 2, qa1); 
  GPURing *ringb = sycl::aligned_alloc_device<GPURing>(4096, 2, qb1);
  std::cout << "ringa " << ringa << std::endl;
  std::cout << "ringb " << ringb << std::endl;

  GPURing *hostringa = (GPURing *) get_mmap_address(ringa, sizeof(GPURing), qa1);
  GPURing *hostringb = (GPURing *) get_mmap_address(ringb, sizeof(GPURing), qb1);
  std::cout << "hostringa " << hostringa << std::endl;
  std::cout << "hostringb " << hostringb << std::endl;

  RingMessage *buffera = sycl::aligned_alloc_device<RingMessage>(4096, 16384, qa1);
  RingMessage *bufferb = sycl::aligned_alloc_device<RingMessage>(4096, 16384, qb1);
  std::cout << "buffera " << buffera << std::endl;
  std::cout << "bufferb " << bufferb << std::endl;

  RingMessage *hostbuffera = (RingMessage *) get_mmap_address(buffera, sizeof(RingMessage) * 8192, qa1);
  RingMessage *hostbufferb = (RingMessage *) get_mmap_address(bufferb, sizeof(RingMessage) * 8192, qb1);
  std::cout << "hostbuffera " << hostbuffera << std::endl;
  std::cout << "hostbufferb " << hostbufferb << std::endl;

  qa1.memset(buffera, 0, sizeof(RingMessage) * 8192).wait();
  qb1.memset(bufferb, 0, sizeof(RingMessage) * 8192).wait();

  unsigned a2bcount = 900000;
  unsigned b2acount = 1000000;
  unsigned athreads = 32;
  unsigned bthreads = 32;

  std::cout << "ready to start kernels" << std::endl;
  sycl::event ea1;
  sycl::event eb1;
  {
    unsigned send_count = a2bcount;
    unsigned recv_count = b2acount;
    std::cout << "gpu a send_count " << send_count << " recv_count " << recv_count << std::endl;
    ea1 = qa1.submit([&](sycl::handler &h) {
	h.parallel_for_work_group(sycl::range(1), sycl::range(athreads), [=](sycl::group<1> grp) {
	    GPURing *gpua = new(ringa) GPURing(23, bufferb, buffera);
	    grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		struct RingMessage msg;
		msg.header = MSG_PUT;
		int si = it.get_global_id()[0];  // send index
		while (si < send_count) {
		  msg.data[1] = si;
		  msg.data[2] = 23;
		  gpua->Send(&msg);
		  si += athreads;
		  while ((gpua->receive_count < recv_count) && gpua->Poll(1));
		}
		while (gpua->receive_count < recv_count) gpua->Poll(1);
	      });
	  });
      });
  }
  {
    int send_count = b2acount;
    int recv_count = a2bcount;
    std::cout << "gpu b send_count " << send_count << " recv_count " << recv_count << std::endl;
    eb1 = qb1.submit([&](sycl::handler &h) {
	h.parallel_for_work_group(sycl::range(1), sycl::range(bthreads), [=](sycl::group<1> grp) {
	    GPURing *gpub = new(ringb) GPURing(43, buffera, bufferb);
	    grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		struct RingMessage msg;
		msg.header = MSG_PUT;
		int si = it.get_global_id()[0];  // send index
		while (si < send_count) {
		  msg.data[1] = si;
		  msg.data[2] = 43;
		  gpub->Send(&msg);
		  si += bthreads;
		  while ((gpub->receive_count < recv_count) && gpub->Poll(1));
		}
		while (gpub->receive_count < recv_count) gpub->Poll(1);
	      });
	  });
      });
  }
    std::cout << "Waiting for kernels" << std::endl;

    sleep(4);
    
    printdatastructures("GPUA", hostringa, hostbufferb, hostbuffera);
    printdatastructures("GPUB", hostringb, hostbuffera, hostbufferb);

    ea1.wait();
    eb1.wait();
    printduration("gpua", ea1);
    printduration("gpub", eb1);
}
