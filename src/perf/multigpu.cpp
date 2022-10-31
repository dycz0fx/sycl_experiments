#include <CL/sycl.hpp>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <getopt.h>
#include "level_zero/ze_api.h"
#include <sys/mman.h>
#include"../common_includes/rdtsc.h"
#include <time.h>
#include <sys/stat.h>
#include <immintrin.h>

constexpr size_t BUFSIZE = (1<<27);
constexpr size_t CHECKSIZE = 4096;
constexpr double NSEC_IN_SEC = 1000000000.0;


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

// get a pointer to src_ptr on device src_q which is usable on device use_q
template<typename T>
T *get_ipc_address(T *src_ptr, size_t size, sycl::queue src_q, sycl::queue use_q) {
  // get the ipc handle from the source context
    sycl::context ctx = src_q.get_context();
    ze_ipc_mem_handle_t ze_ipc_handle;
    ze_result_t ret = zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx), src_ptr, &ze_ipc_handle);
    // std::cout<<"zeMemGetIpcHandle return : " << ret << std::endl;
    assert(ret == ZE_RESULT_SUCCESS);
    // get the ze_context_handle_t from the use_q
    ctx = use_q.get_context();
    ze_context_handle_t ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    // get the sycl::device in use_q
    sycl::device sdev = use_q.get_device();
    // get the ze device corresponding to the sdev
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sdev);
    void *ptr;
    ret = zeMemOpenIpcHandle(ze_ctx, ze_dev, ze_ipc_handle, 0, &ptr);
    assert(ret == ZE_RESULT_SUCCESS);
    return((T *) ptr);
}

/* command line mode arguments:
 *
 * storeflush iter setflagloc pollflagloc tocpumode fromcpumode
 *   tocpuloc 0 = device, 1 = host
 *   togpuloc 0 = device, 1 = host
 *   mode  0 = pointer, 1 = atomic, 2 = uncached
 */



void printduration(const char* name, sycl::event e)
  {
    uint64_t start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
    std::cout << name << " execution time: " << duration << " sec" << std::endl;
  }

char checkbuf[CHECKSIZE];

void memcheck(const char *msg, int src, int dest, int val, size_t size) {
  unsigned errors = 0;
  for (int i = 0; i < size; i += 1) {
    if (checkbuf[i] != val) errors += 1;
    if (errors == 1) printf ("    offset %d expected %d got %d\n", i, val, checkbuf[i]);
  }
  if (errors != 0) printf("    %s %d to %d val %d errors %d\n", msg,  src, dest, val, errors);
}

void setandcheck(const char *msg, int src, int dest, sycl::queue q, uint64_t *p, int val, size_t size) {
  q.memset(p, val, size);
  q.wait_and_throw();
  q.memcpy(checkbuf, p, size);
  q.wait_and_throw();
  memcheck(msg, src, dest, val, size);
}

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

  int haz_ngpu = qs.size();
  std::cout << "Number of GPUs found  = " << haz_ngpu << std::endl;

  int ngpu = 2;   // make a command line argument
  
  std::vector<uint64_t *> device_src(ngpu);
  std::vector<uint64_t *> device_dest(ngpu);

  // vector of a vector of pointers
  std::vector<uint64_t **> device_src_map(ngpu);
  std::vector<uint64_t **> device_dest_map(ngpu);

  for (int i = 0; i < ngpu; i += 1) {
    device_src[i] = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, qs[i]);
    device_dest[i] = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, qs[i]);
    device_src_map[i] = (uint64_t **) sycl::aligned_alloc_host(4096, ngpu * sizeof(uint64_t *), qs[i]);
    
    device_dest_map[i] = (uint64_t **) sycl::aligned_alloc_host(4096, ngpu * sizeof(uint64_t *), qs[i]);

    std::cout << " device_src[" << i << "] " << device_src[i] << std::endl;
    std::cout << " device_dest[" << i << "] " << device_dest[i] << std::endl;
  }

  #if 0
  // initialize IPC addresses for each GPU
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      if (i == j) {
	device_src_map[j][i] = device_src[i];
	device_dest_map[j][i] = device_dest[i];
      } else {
	device_src_map[j][i] = get_ipc_address(device_src[i], BUFSIZE, qs[i],qs[j]);
	device_dest_map[j][i] = get_ipc_address(device_dest[i], BUFSIZE, qs[i],qs[j]);
      }
    }
  }
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      std::cout << "ipc src view from " << i << " of " << j << ": " << device_src_map[i][j] << std::endl; 
    }
    for (int j = 0; j < ngpu; j += 1) {
      std::cout << "ipc dest view from " << i << " of " << j << ": " << device_dest_map[i][j] << std::endl; 
    }
  }
  
  
  // initialize device_src
  std::cout << " memory initialized " << std::endl;

  // test printf
  for (int i = 0; i < ngpu; i += 1) {
    qs[i].submit([&](sycl::handler &h) {
	auto out = sycl::stream(1024, 768, h);
	auto task = 
	  [=]() {
	  out << "In task " << i << "\n";
	  };
	h.single_task(task);
      });
  }

    std::cout<<"store with ipc maps" << std::endl;

  // test cross-device stores
  // from i to j
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      std::cout << "store from " << i << " to " << j << std::endl;
      setandcheck("  store source", i, j, qs[i], &device_src_map[i][i][0], i, CHECKSIZE);
      setandcheck("  store dest", i, j, qs[j], &device_dest_map[j][j][0], 100 + j, CHECKSIZE);
      qs[i].memcpy(&device_dest_map[i][j][0], &device_src_map[i][i][0], CHECKSIZE);
      qs[i].wait_and_throw();
      qs[j].memcpy(checkbuf, &device_dest_map[j][j][0], CHECKSIZE);
      qs[j].wait_and_throw();
      memcheck("  store", i, j, i, CHECKSIZE);
    }
  }

  // test cross-device loads
  std::cout<<"load with ipc maps" << std::endl;
  
  // from i to j
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      std::cout << "load from " << i << " to " << j << std::endl;
      setandcheck("  load src", i, j, qs[i], &device_src_map[i][i][0], i, CHECKSIZE);
      setandcheck("  load dest", i, j, qs[j], &device_dest_map[j][j][0], 100 + j, CHECKSIZE);
      qs[j].memcpy(&device_dest_map[j][j][0], &device_src_map[j][i][0], CHECKSIZE);
      qs[j].wait_and_throw();
      qs[j].memcpy(checkbuf, &device_dest_map[j][j][0], CHECKSIZE);
      qs[j].wait_and_throw();
      memcheck("  load ", i, j, i, CHECKSIZE);
    }
  }

  std::cout<<"test readback map" << std::endl;
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      std::cout << "readback map on " << i << " to " << j << std::endl;
      setandcheck(" readback map ", i, j, qs[i], device_src_map[i][j], (i << 4) + j, CHECKSIZE);
    }
  }
  
#endif
  std::cout << "test readback no ipc" << std::endl;
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      std::cout << "readback on " << i << " to " << j << std::endl;
      setandcheck(" readback ", i, j, qs[i], device_src[j], (i << 4) + j, CHECKSIZE);

    }
  }
  std::cout << "test readback dest no ipc" << std::endl;
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      std::cout << "readback dest on " << i << " to " << j << std::endl;
      setandcheck(" readback dest ", i, j, qs[i], device_dest[j], (i << 4) + j, CHECKSIZE);

    }
  }
  

  // try without the maps
#define TRYNOMAPS 1
#if TRYNOMAPS
  std::cout<<"store without ipc maps" << std::endl;

    // test cross-device stores
  // from i to j
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      std::cout << "store from " << i << " to " << j << std::endl;
      
      //assert(device_src[i] == device_src_map[i][i]);
      //assert(device_dest[j] == device_dest_map[j][j]);
      setandcheck("  store source", i, j, qs[i], &device_src[i][0], i, CHECKSIZE);
      setandcheck("  store dest", i, j, qs[j], &device_dest[j][0], j + 100, CHECKSIZE);
      qs[i].memcpy(&device_dest[j][0], &device_src[i][0], CHECKSIZE);
      qs[i].wait_and_throw();
      qs[j].memcpy(checkbuf, &device_dest[j][0], CHECKSIZE);
      qs[j].wait_and_throw();
      memcheck("  store", i, j, i, CHECKSIZE);
    }
  }

  std::cout<<"load without ipc maps" << std::endl;
  
  // from i to j
  for (int i = 0; i < ngpu; i += 1) {
    for (int j = 0; j < ngpu; j += 1) {
      std::cout << "load from " << i << " to " << j << std::endl;
      //assert(device_src[i] == device_src_map[i][i]);
      //assert(device_dest[j] == device_dest_map[j][j]);
      setandcheck("  load src", i, j, qs[i], &device_src[i][0], i, CHECKSIZE);
      setandcheck("  load dest", i, j, qs[j], &device_dest[j][0], 100 + j, CHECKSIZE);
      qs[j].memcpy(&device_dest[j][0], &device_src[i][0], CHECKSIZE);
      qs[j].wait_and_throw();
      qs[j].memcpy(checkbuf, &device_dest[j][0], CHECKSIZE);
      qs[j].wait_and_throw();
      memcheck("  load ", i, j, i, CHECKSIZE);
    }
  }
#endif
  for (int mode = 0; mode < 3; mode += 1) {
    uint64_t *s, *d;
    switch (mode) {
    case 0:  // push
      s = device_src[0];
      d = device_dest[1];
      break;
    case 1:  // pull
      s = device_src[1];
      d = device_dest[0];
      break;
    case 2:  // same
      s = device_src[0];
      d = device_dest[0];
      break;
    default:
      assert(0);
    }
    printf("csv, mode, size, threads,  count, duration, bandwidth MB/s\n");
    // size is in bytes
    for (size_t size = 8; size < BUFSIZE; size <<= 1) {
      int maxthreads = size/8;
      if (maxthreads > 1024) maxthreads = 1024;
      for (int threads = 1; threads <= maxthreads; threads <<= 1) {
	// locloop is in uint64_t
	uint64_t loc_loop = (size/8) / threads;
	double duration;
	int count;
	// run for more and more counts until it takes more than 0.1 sec
	for (count = 1; count < (1 << 20); count <<= 1) {
	  clock_gettime(CLOCK_REALTIME, &ts_start);
	  auto e = qs[0].submit([&](sycl::handler &h) {
	      h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
		  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		      int j = it.get_global_id()[0];
		      for (int iter = 0; iter < count; iter += 1) {
			for (int k = j * loc_loop; k < (j+1) * loc_loop; k += 1) {
			  d[k] = s[k];
			}
			sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
		      }
		    });
		});
	    });
	  e.wait_and_throw();
	  clock_gettime(CLOCK_REALTIME, &ts_end);
	  duration = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	    ((double) (ts_end.tv_nsec - ts_start.tv_nsec));
	  duration /= 1000000000.0;
	  if (duration > 0.1) {
	    break;
	  }
	}
	double per_iter = duration / count;
	double bw = size / (per_iter);
	double bw_mb = bw / 1000000.0;
	printf("csv, %d, %ld, %d, %d, %f, %f\n", mode, size, threads, count, per_iter, bw_mb);
      }
    }
  }
      
  std::cout<<"kernel returned" << std::endl;
  return 0;

}



#if 0
	uint64_t start =
	  e.get_profiling_info<sycl::info::event_profiling::command_start>();
	uint64_t end =
	  e.get_profiling_info<sycl::info::event_profiling::command_end>();
	duration = static_cast<double>(end - start) / NSEC_IN_SEC;
	if (duration > 0.1) {
	  duration /= count;
	  break;
	}
#endif

