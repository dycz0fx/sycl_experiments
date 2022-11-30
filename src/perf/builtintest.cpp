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

void check_ulong(const char *test, ulong *d, size_t count)
{
  for (int idx = 0; idx < count; idx += 1) {
    ulong val;
    val = idx;
    if (d[idx] != val) {
      printf("ulong2 %s error idx[%d] == %ld should be %ld\n", test, idx, d[idx], val);
    }
  }
  printf("ulong2 %s\n", test);
}

void check_ulong2(const char *test, ulong2 *d, size_t count)
{
  for (int idx = 0; idx < count; idx += 1) {
    ulong2 val;
    for (int i = 0; i < 1; i += 1) {
      val[i] = idx + (i*1000000);
      if (d[idx][i] != val[i]) {
	printf("ulong2 %s error idx[%d][%d] == %ld should be %ld\n", test, idx, i, d[idx][i], val[i]);
      }
    }
  }
  printf("ulong2 %s\n", test);
}

void check_ulong4(const char *test, ulong4 *d, size_t count)
{
  for (int idx = 0; idx < count; idx += 1) {
    ulong4 val;
    for (int i = 0; i < 1; i += 1) {
      val[i] = idx + (i*1000000);
      if (d[idx][i] != val[i]) {
	printf("ulong4 %s error idx[%d][%d] == %ld should be %ld\n", test, idx, i, d[idx][i], val[i]);
      }
    }
  }
  printf("ulong4 %s\n", test);
}

void check_ulong8(const char *test, ulong8 *d, size_t count)
{
  for (int idx = 0; idx < count; idx += 1) {
    ulong8 val;
    for (int i = 0; i < 1; i += 1) {
      val[i] = idx + (i*1000000);
      if (d[idx][i] != val[i]) {
	printf("ulong8 %s error idx[%d][%d] == %ld should be %ld\n", test, idx, i, d[idx][i], val[i]);
      }
    }
  }
  printf("ulong8 %s\n", test);
}

void kernel_fill_ulong(sycl::queue q, ulong *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong val = idx;
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}


void kernel_fill_ulong2(sycl::queue q, ulong2 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong2 val;
	  for (int i = 0; i < 2; i += 1) val[i] = idx + (i*1000000);
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}


void kernel_fill_ulong4(sycl::queue q, ulong4 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong4 val;
	  for (int i = 0; i < 4; i += 1) val[i] = idx + (i*1000000);
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}

void kernel_fill_ulong8(sycl::queue q, ulong8 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong8 val;
	  for (int i = 0; i < 8; i += 1) val[i] = idx + (i*1000000);
	  d[idx] = val;
	});
    });
  e.wait_and_throw();
}


void kernel_store_ulong(sycl::queue q, ulong *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong val = idx;
	  ucs_ulong(&d[idx], val);
	});
    });
  e.wait_and_throw();
}

void kernel_store_ulong2(sycl::queue q, ulong2 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong2 val;
	  for (int i = 0; i < 2; i += 1) val[i] = idx + (i*1000000);
	  ucs_ulong2(&d[idx],val);
	});
    });
  e.wait_and_throw();
}

void kernel_store_ulong4(sycl::queue q, ulong4 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong4 val;
	  for (int i = 0; i < 4; i += 1) val[i] = idx + (i*1000000);
	  ucs_ulong4(&d[idx],val);
	});
    });
  e.wait_and_throw();
}

void kernel_store_ulong8(sycl::queue q, ulong8 *d, size_t count)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range(count), [=](sycl::id<1> idx) {
	  ulong8 val;
	  for (int i = 0; i < 8; i += 1) val[i] = idx + (i*1000000);
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

uint64_t *test_mem;
sycl::queue testq;
sycl::queue checkq;


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
	  qcount += 1;
	} else {
	  std::vector<sycl::device> sd = d.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::next_partitionable);
	  std::cout << " subdevices " << sd.size() << std::endl;
	  std::cout << "**max wg: " << d.get_info<cl::sycl::info::device::max_work_group_size>()  << std::endl;
	  for (auto &subd: sd) {
	    tileq.push_back(sycl::queue(subd, prop_list));
	    std::cout << "create tileq[" << qcount << "]" << std::endl;
	    qcount += 1;
	  }
	}
	if (qcount >= 4) break;
      }
      if (qcount >= 4) break;
    }
  }

  std::cout << "Number of GPUs = " << pvcq.size() << std::endl;
  std::cout << "Number of tiles = " << tileq.size() << std::endl;
  
  uint64_t * host_mem = (uint64_t *) sycl::aligned_alloc_host(4096, BUFSIZE, pvcq[0]);
  uint64_t * check_mem = (uint64_t *) sycl::aligned_alloc_host(4096, BUFSIZE, pvcq[0]);
  std::vector<uint64_t *> pvc_mem;
  std::vector<uint64_t *> tile_mem;
  for (int i = 0; i < 2; i += 1) {
    pvc_mem.push_back((uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, pvcq[i]));
    tile_mem.push_back((uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE, tileq[i]));
  }
  int size = 0;
  int mode = 0;
  int writeq = 0;
  int readq = 0;
  int mem = 0;
  if (argc < 6) exit(1);
  size = atoi(argv[1]);
  mode = atoi(argv[2]);
  writeq = atoi(argv[3]);
  readq = atoi(argv[4]);
  mem = atoi(argv[5]);
  
  std::cout << " size " << size << std::endl;
  std::cout << " mode " << mode << std::endl;
  std::cout << " writeq " << writeq << std::endl;
  std::cout << " readq " << readq << std::endl;
  std::cout << " mem " << mem << std::endl;
  assert(size <= BUFSIZE);
  
  void * test_mem = NULL;
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

  assert((mode == 1) || (mode == 2) || (mode == 4) || (mode = 8));
  assert((writeq == 0) || (writeq == 1) || (writeq == 2) || (writeq == 3) || (writeq == 4));
  assert((readq == 0) || (readq == 1) || (readq == 2) || (readq == 3) || (readq == 4));
  assert((mem == 0) || (mem == 1) || (mem == 2) || (mem == 3) || (mem == 4));
  if (writeq == 0) {
    assert(0);//    testq = hostq;
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
    assert(0);//    testq = hostq;
  } else if (readq == 1) {
    checkq = pvcq[0];
  } else if (readq == 2) {
    checkq = pvcq[1];
  } else if (readq == 3) {
    checkq = tileq[0];
  } else if (readq == 4) {
    checkq = tileq[1];
  }
  if (mem == 0) {
    test_mem = host_mem;
  } else if (mem == 1) {
    test_mem = pvc_mem[0];
  } else if (mem == 2) {
    test_mem = pvc_mem[1];
  } else if (mem == 3) {
    test_mem = tile_mem[0];
  } else if (mem == 4) {
    test_mem = tile_mem[1];
  }

  if (mode == 1) {
    size_t count = size / sizeof(ulong);
    ulong *test_area = (ulong *) test_mem;
    ulong *check_area = (ulong *) check_mem;
    testq.memset(test_area, 0xff, size);
    testq.wait_and_throw();
    kernel_fill_ulong(testq, test_area, count);
    memset(check_area, 0xff, size);
    checkq.memcpy(check_area, test_area, size);
    checkq.wait_and_throw();
    check_ulong("fill", check_area, count);
    testq.memset(test_area, 0xff, size);
    testq.wait_and_throw();
    kernel_store_ulong(testq, test_area, size);
    memset(check_area, 0xff, size);
    checkq.memcpy(check_area, test_area, size);
    checkq.wait_and_throw();
    check_ulong("store", check_area, count);
    testq.memset(test_area, 0xff, size);
    kernel_fill_ulong(testq, test_area, count);
    memset(check_area, 0xff, size);
    kernel_load_ulong(checkq, check_area, test_area, count);
    check_ulong("load", check_area, count);
  } else if (mode == 2) {
    size_t count = size / sizeof(ulong2);
    ulong2 *test_area = (ulong2 *) test_mem;
    ulong2 *check_area = (ulong2 *) check_mem;
    testq.memset(test_area, 0xff, size);
    testq.wait_and_throw();
    kernel_fill_ulong2(testq, test_area, count);
    memset(check_area, 0xff, size);
    checkq.memcpy(check_area, test_area, size);
    checkq.wait_and_throw();
    check_ulong2("fill", check_area, count);
    testq.memset(test_area, 0xff, size);
    testq.wait_and_throw();
    kernel_store_ulong2(testq, test_area, size);
    memset(check_area, 0xff, size);
    checkq.memcpy(check_area, test_area, size);
    checkq.wait_and_throw();
    check_ulong2("store", check_area, count);
    testq.memset(test_area, 0xff, size);
    kernel_fill_ulong2(testq, test_area, count);
    memset(check_area, 0xff, size);
    kernel_load_ulong2(checkq, check_area, test_area, count);
    check_ulong2("load", check_area, count);
  } else if (mode == 4) {
    size_t count = size / sizeof(ulong4);
    ulong4 *test_area = (ulong4 *) test_mem;
    ulong4 *check_area = (ulong4 *) check_mem;
    testq.memset(test_area, 0xff, size);
    testq.wait_and_throw();
    kernel_fill_ulong4(testq, test_area, count);
    memset(check_area, 0xff, size);
    checkq.memcpy(check_area, test_area, size);
    checkq.wait_and_throw();
    check_ulong4("fill", check_area, count);
    testq.memset(test_area, 0xff, size);
    testq.wait_and_throw();
    kernel_store_ulong4(testq, test_area, size);
    memset(check_area, 0xff, size);
    checkq.memcpy(check_area, test_area, size);
    checkq.wait_and_throw();
    check_ulong4("store", check_area, count);
    testq.memset(test_area, 0xff, size);
    kernel_fill_ulong4(testq, test_area, count);
    memset(check_area, 0xff, size);
    kernel_load_ulong4(checkq, check_area, test_area, count);
    check_ulong4("load", check_area, count);
  } else if (mode == 8) {
    size_t count = size / sizeof(ulong8);
    ulong8 *test_area = (ulong8 *) test_mem;
    ulong8 *check_area = (ulong8 *) check_mem;
    testq.memset(test_area, 0xff, size);
    testq.wait_and_throw();
    kernel_fill_ulong8(testq, test_area, count);
    memset(check_area, 0xff, size);
    checkq.memcpy(check_area, test_area, size);
    checkq.wait_and_throw();
    check_ulong8("fill", check_area, count);
    testq.memset(test_area, 0xff, size);
    testq.wait_and_throw();
    kernel_store_ulong8(testq, test_area, size);
    memset(check_area, 0xff, size);
    checkq.memcpy(check_area, test_area, size);
    checkq.wait_and_throw();
    check_ulong8("store", check_area, count);
    testq.memset(test_area, 0xff, size);
    kernel_fill_ulong8(testq, test_area, count);
    memset(check_area, 0xff, size);
    kernel_load_ulong8(checkq, check_area, test_area, count);
    check_ulong8("load", check_area, count);
  }
  return 0;
}

