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
#include <chrono>
#include "uncached.cpp"
#include <sys/stat.h>

constexpr size_t BUFSIZE = (1L<<20);
constexpr double NSEC_IN_SEC = 1000000000.0;

#ifdef __SYCL_DEVICE_ONLY__
extern "C" {
  SYCL_EXTERNAL ulong intel_get_cycle_counter() __attribute__((overloadable));
}

ulong get_cycle_counter() {
  return intel_get_cycle_counter();
}
#else
ulong get_cycle_counter() {
  return 0xDEADBEEF;
}
#endif // __SYCL_DEVICE_ONLY__

void printduration(const char* name, sycl::event e)
{
  uint64_t start =
    e.get_profiling_info<sycl::info::event_profiling::command_start>();
  uint64_t end =
    e.get_profiling_info<sycl::info::event_profiling::command_end>();
  double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
  std::cout << name << " execution time: " << duration << " sec" << std::endl;
}

int check_ulong(const char *test, ulong *d, size_t count, ulong pat)
{
  int errors = 0;
  for (int idx = 0; idx < count; idx += 1) {
    ulong val;
    val = pat + idx;
    if (d[idx] != val) {
      errors += 1;
      printf("ulong2 %s error idx[%d] == %ld should be %ld\n", test, idx, d[idx], val);
    }
  }
  return(errors);
}

int check_ulong2(const char *test, ulong2 *d, size_t count, ulong pat)
{
  int errors = 0;
  for (int idx = 0; idx < count; idx += 1) {
    ulong2 val;
    for (int i = 0; i < 1; i += 1) {
      val[i] = pat + idx + (i*1000000);
      if (d[idx][i] != val[i]) {
	errors += 1;
	printf("ulong2 %s error idx[%d][%d] == %ld should be %ld\n", test, idx, i, d[idx][i], val[i]);
      }
    }
  }
  return(errors);
}

int check_ulong4(const char *test, ulong4 *d, size_t count, ulong pat)
{
  int errors = 0;
  for (int idx = 0; idx < count; idx += 1) {
    ulong4 val;
    for (int i = 0; i < 1; i += 1) {
      val[i] = pat + idx + (i*1000000);
      if (d[idx][i] != val[i]) {
	errors += 1;
	printf("ulong4 %s error idx[%d][%d] == %ld should be %ld\n", test, idx, i, d[idx][i], val[i]);
      }
    }
  }
  return(errors);
}

int check_ulong8(const char *test, ulong8 *d, size_t count, ulong pat)
{
  int errors = 0;
  for (int idx = 0; idx < count; idx += 1) {
    ulong8 val;
    for (int i = 0; i < 1; i += 1) {
      val[i] = pat + idx + (i*1000000);
      if (d[idx][i] != val[i]) {
	errors += 1;
	printf("ulong8 %s error idx[%d][%d] == %ld should be %ld\n", test, idx, i, d[idx][i], val[i]);
      }
    }
  }
  return(errors);
}

void kernel_fill_ulong(sycl::queue q, ulong *d, size_t count, ulong pat)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong val = pat + idx;
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}


void kernel_fill_ulong2(sycl::queue q, ulong2 *d, size_t count, ulong pat)
{
  //printf("kernel_fill_ulong2 d %p count %ld pat %ld\n", d, count, pat);
  //fflush(stdout);
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong2 val;
	  for (int i = 0; i < 2; i += 1) val[i] = pat + idx + (i*1000000);
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}


void kernel_fill_ulong4(sycl::queue q, ulong4 *d, size_t count, ulong pat)
{
  //printf("kernel_fill_ulong4 d %p count %ld pat %ld\n", d, count, pat);
  //fflush(stdout);
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong4 val;
	  for (int i = 0; i < 4; i += 1) val[i] = pat + idx + (i*1000000);
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}

void kernel_fill_ulong8(sycl::queue q, ulong8 *d, size_t count, ulong pat)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong8 val;
	  for (int i = 0; i < 8; i += 1) val[i] = pat + idx + (i*1000000);
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}


void kernel_store_ulong(sycl::queue q, ulong *d, size_t count, ulong pat)
{			
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong val = pat + idx;
	  ucs_ulong(&d[idx], val);
	});
    });
  e.wait_and_throw();
}

void kernel_store_ulong2(sycl::queue q, ulong2 *d, size_t count, ulong pat)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong2 val;
	  for (int i = 0; i < 2; i += 1) val[i] = pat + idx + (i*1000000);
	  ucs_ulong2(&d[idx],val);
	});
    });
  e.wait_and_throw();
}

void kernel_store_ulong4(sycl::queue q, ulong4 *d, size_t count, ulong pat)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong4 val;
	  for (int i = 0; i < 4; i += 1) val[i] = pat + idx + (i*1000000);
	  ucs_ulong4(&d[idx],val);
	});
    });
  e.wait_and_throw();
}

void kernel_store_ulong8(sycl::queue q, ulong8 *d, size_t count, ulong pat)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong8 val;
	  for (int i = 0; i < 8; i += 1) val[i] = pat + idx + (i*1000000);
	  ucs_ulong8(&d[idx],val);
	});
    });
  e.wait_and_throw();
}


void kernel_load_ulong(sycl::queue q, ulong *d, ulong *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = ucl_ulong(&s[idx]);
	});
    });
  e.wait_and_throw();
}


void kernel_load_ulong2(sycl::queue q, ulong2 *d, ulong2 *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = ucl_ulong2(&s[idx]);
	});
    });
  e.wait_and_throw();
}

void kernel_load_ulong4(sycl::queue q, ulong4 *d, ulong4 *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = ucl_ulong4(&s[idx]);
	});
    });
  e.wait_and_throw();
}

void kernel_load_ulong8(sycl::queue q, ulong8 *d, ulong8 *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = ucl_ulong8(&s[idx]);
	});
    });
  e.wait_and_throw();
}

void kernel_copy_ulong(sycl::queue q, ulong *d, ulong *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = s[idx];
	});
    });
  e.wait_and_throw();
}


void kernel_copy_ulong2(sycl::queue q, ulong2 *d, ulong2 *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = s[idx];
	});
    });
  e.wait_and_throw();
}

void kernel_copy_ulong4(sycl::queue q, ulong4 *d, ulong4 *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = s[idx];
	});
    });
  e.wait_and_throw();
}

void kernel_copy_ulong8(sycl::queue q, ulong8 *d, ulong8 *s, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  d[idx] = s[idx];
	});
    });
  e.wait_and_throw();
}


int runtest(int mode, size_t size, int test_host, sycl::queue testq, int check_host, sycl::queue checkq, uint64_t *store_test_mem, uint64_t *load_test_mem, uint64_t *check_mem)
{
  int errors = 0;
  if (mode == 1) {
    size_t count = size / sizeof(ulong);
    ulong *store_test_area = (ulong *) store_test_mem;
    ulong *load_test_area = (ulong *) load_test_mem;
    ulong *check_area = (ulong *) check_mem;
    memset(check_area, 0xff, size);
    kernel_fill_ulong(testq, store_test_area, count, 1000000000);
    kernel_copy_ulong(checkq, check_area, load_test_area, count);
    errors += check_ulong("store", check_area, count, 1000000000);
    memset(check_area, 0xff, size);
    if (test_host) kernel_fill_ulong(testq, store_test_area, count, 2000000000);
    else kernel_store_ulong(testq, store_test_area, count, 2000000000);
    if (check_host) kernel_copy_ulong(checkq, check_area, load_test_area, count);
    else kernel_load_ulong(checkq, check_area, load_test_area, count);
    errors += check_ulong("load", check_area, count, 2000000000);
  } else if (mode == 2) {
    size_t count = size / sizeof(ulong2);
    ulong2 *store_test_area = (ulong2 *) store_test_mem;
    ulong2 *load_test_area = (ulong2 *) load_test_mem;
    ulong2 *check_area = (ulong2 *) check_mem;
    memset(check_area, 0xff, size);
    kernel_fill_ulong2(testq, store_test_area, count, 1000000000);
    kernel_copy_ulong2(checkq, check_area, load_test_area, count);
    errors += check_ulong2("store", check_area, count, 1000000000);
    memset(check_area, 0xff, size);
    if (test_host) kernel_fill_ulong2(testq, store_test_area, count, 2000000000);
    else kernel_store_ulong2(testq, store_test_area, count, 2000000000);
    if (check_host) kernel_copy_ulong2(checkq, check_area, load_test_area, count);
    else kernel_load_ulong2(checkq, check_area, load_test_area, count);
    errors += check_ulong2("load", check_area, count, 2000000000);
  } else if (mode == 4) {
    size_t count = size / sizeof(ulong4);
    ulong4 *store_test_area = (ulong4 *) store_test_mem;
    ulong4 *load_test_area = (ulong4 *) load_test_mem;
    ulong4 *check_area = (ulong4 *) check_mem;
    memset(check_area, 0xff, size);
    kernel_fill_ulong4(testq, store_test_area, count, 1000000000);
    kernel_copy_ulong4(checkq, check_area, load_test_area, count);
    errors += check_ulong4("store", check_area, count, 1000000000);
    memset(check_area, 0xff, size);
    if (test_host) kernel_fill_ulong4(testq, store_test_area, count, 2000000000);
    else kernel_store_ulong4(testq, store_test_area, count, 2000000000);
    if (check_host) kernel_copy_ulong4(checkq, check_area, load_test_area, count);
    else kernel_load_ulong4(checkq, check_area, load_test_area, count);
    errors += check_ulong4("load", check_area, count, 2000000000);
  } else if (mode == 8) {
    size_t count = size / sizeof(ulong4);
    ulong4 *store_test_area = (ulong4 *) store_test_mem;
    ulong4 *load_test_area = (ulong4 *) load_test_mem;
    ulong4 *check_area = (ulong4 *) check_mem;
    memset(check_area, 0xff, size);
    kernel_fill_ulong4(testq, store_test_area, count, 1000000000);
    kernel_copy_ulong4(checkq, check_area, load_test_area, count);
    errors += check_ulong4("store", check_area, count, 1000000000);
    memset(check_area, 0xff, size);
    if (test_host) kernel_fill_ulong4(testq, store_test_area, count, 2000000000);
    else kernel_store_ulong4(testq, store_test_area, count, 2000000000);
    if (check_host) kernel_copy_ulong4(checkq, check_area, load_test_area, count);
    else kernel_load_ulong4(checkq, check_area, load_test_area, count);
    errors += check_ulong4("load", check_area, count, 2000000000);
  }
  return (errors);
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



// ############################

int main(int argc, char *argv[]) {
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

  // modelled after Jeff Hammond code in PRK repo
  /* The objective of this code is to have two complete PVCs with one queue each, and 
   * a third PVC with one queue for each o the two tiles.
   */
  std::vector<sycl::queue> pvcq;
  std::vector<sycl::queue> tileq;

  int qcount = 0;	   
  
  auto platforms = sycl::platform::get_platforms();
  for (auto & p : platforms) {
    auto pname = p.get_info<sycl::info::platform::name>();
    std::cout << "*Platform: " << pname << std::endl;
    if ( pname.find("Level-Zero") == std::string::npos) {
        std::cout << "non Level Zero GPU skipped" << std::endl;
        continue;
    }
    auto devices = p.get_devices();
    std::cout << "number of devices: " << devices.size() << std::endl;
    for (auto & d : devices ) {
      std::cout << "**Device: " << d.get_info<sycl::info::device::name>() << std::endl;
      if (d.is_gpu()) {
	if (qcount < 2) {
	  pvcq.push_back(sycl::queue(d, prop_list));
	  std::cout << "create pvcq[" << qcount << "]" << std::endl;
	  std::cout << "**max wg: " << d.get_info<cl::sycl::info::device::max_work_group_size>()  << std::endl;
	  qcount += 1;
	} else {
	  std::vector<sycl::device> sd = d.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::next_partitionable);
	  std::cout << " subdevices " << sd.size() << std::endl;
	  for (auto &subd: sd) {
	    tileq.push_back(sycl::queue(subd, prop_list));
	    std::cout << "create tileq[" << qcount << "]" << std::endl;
	    std::cout << "**max wg: " << subd.get_info<cl::sycl::info::device::max_work_group_size>()  << std::endl;
	    qcount += 1;
	  }
	}
	if (qcount >= 4) break;
      }
      if (qcount >= 4) break;
    }
  }
  sycl::queue hostq = sycl::queue(sycl::cpu_selector_v, prop_list);
  std::cout << "create host queue" << std::endl;
  
  std::cout << "Number of GPUs = " << pvcq.size() << std::endl;
  std::cout << "Number of tiles = " << tileq.size() << std::endl;
  
  uint64_t * host_mem = (uint64_t *) sycl::aligned_alloc_host(4096, BUFSIZE, pvcq[0]);
  uint64_t * check_mem = (uint64_t *) sycl::aligned_alloc_host(4096, BUFSIZE, pvcq[0]);
  std::vector<uint64_t *> pvc_mem;
  std::vector<uint64_t *> tile_mem;
  for (int i = 0; i < 2; i += 1) {
    pvc_mem.push_back((uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE*2, pvcq[i]));
    tile_mem.push_back((uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE*2, tileq[i]));
  }
  std::vector<uint64_t *> pvc_mem_hostmap;
  std::vector<uint64_t *> tile_mem_hostmap;
  for (int i = 0; i < 2; i += 1) {
    pvc_mem_hostmap.push_back(get_mmap_address(pvc_mem[i], BUFSIZE, pvcq[i]));
    tile_mem_hostmap.push_back(get_mmap_address(tile_mem[i], BUFSIZE, tileq[i]));
  }

  size_t size_lo = 64;
  size_t size_hi = BUFSIZE;
  int mode_lo = 1;
  int mode_hi = 8;
  int writeq_lo = 0;
  int writeq_hi = 4;
  int readq_lo = 0;
  int readq_hi = 4;
  int mem_lo = 0;
  int mem_hi = 4;
  if (argc == 6) {
    mode_lo = mode_hi = atol(argv[1]);
    size_lo = size_hi = atol(argv[2]);
    writeq_lo = writeq_hi = atol(argv[3]);
    readq_lo = readq_hi = atol(argv[4]);
    mem_lo = mem_hi = atol(argv[5]);
  }
  
  /* mode
   * 1    ulong
   * 2    ulong2
   * 4    ulong4
   * 8    ulong8
   */
  /* cmd
   * 0       host
   * 1, 2    pvc     use intrinsics
   * 3, 4    tile    use intrinsics
   */
  /* mem
   * 0      host
   * 1, 2   pvc
   * 3, 4   tile
   */
  printf("csv,mode, size, writeq, readq, mem, errors\n");
  fflush(stdout);
  for (size_t size = size_lo; size <= size_hi; size <<= 1) {
    for (int mode = mode_lo; mode <= mode_hi; mode <<=1) {
      for (int writeq = writeq_lo; writeq <=writeq_hi; writeq += 1) {
	for (int readq = readq_lo; readq <= readq_hi; readq += 1) {
	  for (int mem = mem_lo; mem <= mem_hi; mem += 1) {
	    uint64_t *store_test_mem;
	    uint64_t *load_test_mem;
	    sycl::queue testq;
	    sycl::queue checkq;

	    assert((size >= 64) && (size <= BUFSIZE));
	    assert((mode == 1) || (mode == 2) || (mode == 4) || (mode == 8));
	    assert((writeq == 0) || (writeq == 1) || (writeq == 2) || (writeq == 3) || (writeq == 4));
	    assert((readq == 0) || (readq == 1) || (readq == 2) || (readq == 3) || (readq == 4));
	    assert((mem == 0) || (mem == 1) || (mem == 2) || (mem == 3) || (mem == 4));
	    if (writeq == 0) {
	     testq = hostq;
	    } else if (writeq == 1) {
	      testq = pvcq[0];
	    } else if (writeq == 2) {
	      testq = pvcq[1];
	    } else if (writeq == 3) {
	      testq = tileq[0];
	    } else if (writeq == 4) {
	      testq = tileq[1];
	    }
	    if (readq == 0) {
	       checkq = hostq;
	    } else if (readq == 1) {
	      checkq = pvcq[0];
	    } else if (readq == 2) {
	      checkq = pvcq[1];
	    } else if (readq == 3) {
	      checkq = tileq[0];
	    } else if (readq == 4) {
	      checkq = tileq[1];
	    }
	    if (writeq == 0) {
	      if (mem == 0) {
		store_test_mem = host_mem;
	      } else if (mem == 1) {
		store_test_mem = pvc_mem_hostmap[0];
	      } else if (mem == 2) {
		store_test_mem = pvc_mem_hostmap[1];
	      } else if (mem == 3) {
		store_test_mem = tile_mem_hostmap[0];
	      } else if (mem == 4) {
		store_test_mem = tile_mem_hostmap[1];
	      }
	    } else {
	      if (mem == 0) {
		store_test_mem = host_mem;
	      } else if (mem == 1) {
		store_test_mem = pvc_mem[0];
	      } else if (mem == 2) {
		store_test_mem = pvc_mem[1];
	      } else if (mem == 3) {
		store_test_mem = tile_mem[0];
	      } else if (mem == 4) {
		store_test_mem = tile_mem[1];
	      }
	    }
	    if (readq == 0) {
	      if (mem == 0) {
		load_test_mem = host_mem;
	      } else if (mem == 1) {
		load_test_mem = pvc_mem_hostmap[0];
	      } else if (mem == 2) {
		load_test_mem = pvc_mem_hostmap[1];
	      } else if (mem == 3) {
		load_test_mem = tile_mem_hostmap[0];
	      } else if (mem == 4) {
		load_test_mem = tile_mem_hostmap[1];
	      }
	    } else {
	      if (mem == 0) {
		load_test_mem = host_mem;
	      } else if (mem == 1) {
		load_test_mem = pvc_mem[0];
	      } else if (mem == 2) {
		load_test_mem = pvc_mem[1];
	      } else if (mem == 3) {
		load_test_mem = tile_mem[0];
	      } else if (mem == 4) {
		load_test_mem = tile_mem[1];
	      }
	    }
	    printf("next %d,%ld,%d,%d,%d\n", mode, size, writeq, readq, mem);
	    fflush(stdout);
	    int errors = runtest(mode, size, writeq==0, testq, readq==0, checkq, store_test_mem, load_test_mem, check_mem);
	    printf("csv,%d,%ld,%d,%d,%d,%d\n", mode, size, writeq, readq, mem, errors);
	    fflush(stdout);
	  }
	}
      }
    }
  }
  
  
    return 0;
}

