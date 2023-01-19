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

#include "uncached.cpp"
#include "syclutilities.cpp"
#include <immintrin.h>
#include <omp.h>
    
#include "../common_includes/rdtsc.h"

void writecyclecounter(const char *name, sycl::queue q, uint64_t *buffer, size_t count, int threads)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
	  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
	      int si = it.get_global_id()[0];  // send index
	      while (si < count) {
		buffer[si] = get_cycle_counter();
		si += threads;
	      }
	    });
	});
    });
  e.wait();
  printduration("writecyclecounter", e);
}

void readcyclecounter(const char *name, sycl::queue q, uint64_t *buffer, size_t count, int threads)
{
  uint64_t start, end;
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
	  uint64_t start = get_cycle_counter();
	  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
	      int si = it.get_global_id()[0];  // send index
	      while (si < count) {
		volatile uint64_t v = buffer[si];
		si += threads;
	      }
	    });
	  uint64_t end = get_cycle_counter();
	  buffer[0] = end-start;
	});
    });
  e.wait();
  printduration(name, e);
}

void writecyclecounter_uncached(const char *name, sycl::queue q, uint64_t *buffer, size_t count, int threads)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
	  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
	      int si = it.get_global_id()[0];  // send index
	      while (si < count) {
		ucs_ulong(&buffer[si], get_cycle_counter());
		si += threads;
	      }
	    });
	});
    });
  e.wait();
  printduration("writecyclecounter", e);
}

void readcyclecounter_uncached(const char *name, sycl::queue q, uint64_t *buffer, size_t count, int threads)
{
  uint64_t start, end;
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
	  uint64_t start = get_cycle_counter();
	  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
	      int si = it.get_global_id()[0];  // send index
	      while (si < count) {
		volatile uint64_t v = ucl_ulong(&buffer[si]);
		si += threads;
	      }
	    });
	  uint64_t end = get_cycle_counter();
	  buffer[0] = end-start;
	});
    });
  e.wait();
  printduration(name, e);
}

void pingpong(sycl::queue q, uint64_t *gr, uint64_t *gw, uint64_t *hr, uint64_t *hw, size_t count)
{
  volatile uint64_t *bw = hw;
  volatile uint64_t *br = hr;

  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(1), [=](sycl::group<1> grp) {
	  gw[16] = get_cycle_counter();
	  ulong i;
	  for (i = 1; i <= count; i += 1) {
	    while (ucl_ulong(gr) != i) {
	      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
	    }
	    ucs_ulong(gw, i);
	    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
	  }
	  gw[24] = get_cycle_counter();
	});
    });
  ulong start = rdtsc();
  for (ulong i = 1; i <= count; i += 1) {
    *bw = i;
    while (*br != i) cpu_relax();
  }
  ulong end = rdtsc();
  e.wait();
  ulong kernel_cycles = kernelcycles(e);
  ulong clock = max_frequency(q);
  printf("rdtsc elapsed %ld\n", end - start);
  printf("kernel nsec %ld\n", kernel_cycles);
  ulong cstart = hr[16];
  ulong cend = hr[24];
  printf("kernel cycles %ld clock %ld, sec %f\n", cend-cstart, clock, (double) (cend-cstart) / (double) (clock * 1000000));
  
  printduration("pingpong", e);
}


constexpr size_t BUFSIZE = 1L << 24;

int main(int argc, char *argv[]) {
  std::cout << "Number of available threads: " << omp_get_num_threads() << std::endl;
  if (argc < 2) exit(1);
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

  sycl::queue qa1 = sycl::queue(sycl::gpu_selector_v, prop_list);
  
  std::cout<<"selected device : "<<qa1.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout<<"device vendor : "<<qa1.get_device().get_info<sycl::info::device::vendor>() << std::endl;

  uint64_t *host_buffer = (uint64_t *) sycl::aligned_alloc_host(4096, BUFSIZE * 2, qa1);
  uint64_t *device_buffer = (uint64_t *) sycl::aligned_alloc_device(4096, BUFSIZE * 2, qa1);
  std::cout << "host_buffer " << host_buffer << std::endl;
  std::cout << "device_buffer " << device_buffer << std::endl;

  uint64_t *host_access_device_buffer = (uint64_t *) get_mmap_address(device_buffer, BUFSIZE, qa1);
  std::cout << "host_access_device_buffer " << host_access_device_buffer << std::endl;

  qa1.memset(device_buffer, 0, BUFSIZE).wait();
  memset(host_buffer, 0, BUFSIZE);

  if (strcmp(argv[1], "gpureaddevice") == 0) {
    assert(argc == 4);
    unsigned count = atol(argv[2]);
    assert(count < (BUFSIZE/sizeof(uint64_t)));
    unsigned athreads = atol(argv[3]);
    readcyclecounter("gpureaddevice", qa1, device_buffer, count, athreads);
    printf("%s count %u cycles %ld\n", "gpureaddevice", count, host_access_device_buffer[0]);
  }
  if (strcmp(argv[1], "gpureadhost") == 0) {
    assert(argc == 4);
    unsigned count = atol(argv[2]);
    assert(count < (BUFSIZE/sizeof(uint64_t)));
    unsigned athreads = atol(argv[3]);
    readcyclecounter_uncached("gpureadhost", qa1, host_buffer, count, athreads);
    printf("%s count %u cycles %ld\n", "gpureadhost", count, host_buffer[0]);
  }
  if (strcmp(argv[1], "gpureaddeviceucl") == 0) {
    assert(argc == 4);
    unsigned count = atol(argv[2]);
    assert(count < (BUFSIZE/sizeof(uint64_t)));
    unsigned athreads = atol(argv[3]);
    readcyclecounter_uncached("gpureaddeviceucl", qa1, device_buffer, count, athreads);
    printf("%s count %u cycles %ld\n", "gpureaddeviceucl", count, host_access_device_buffer[0]);
  }
  if (strcmp(argv[1], "gpureadhostucl") == 0) {
    assert(argc == 4);
    unsigned count = atol(argv[2]);
    assert(count < (BUFSIZE/sizeof(uint64_t)));
    unsigned athreads = atol(argv[3]);
    readcyclecounter("gpureadhostucl", qa1, host_buffer, count, athreads);
    printf("%s count %u cycles %ld\n", "gpureadhostucl", count, host_buffer[0]);
  }
  if (strcmp(argv[1], "gpuwritedevice") == 0) {
    assert(argc == 4);
    unsigned count = atol(argv[2]);
    assert(count < (BUFSIZE/sizeof(uint64_t)));
    unsigned athreads = atol(argv[3]);
    writecyclecounter("gpuwritedevice", qa1, device_buffer, count, athreads);
    qa1.memcpy(host_buffer, device_buffer, count * sizeof(uint64_t)).wait();
    uint64_t old = 0;
    for (size_t i = 0; i < count; i += 1) {
      uint64_t v = host_buffer[i];
      printf("%8ld %ld, diff %ld\n", i, v, v-old);
      old = v;
    }
  }
  if (strcmp(argv[1], "gpuwritehost") == 0) {
    assert(argc == 4);
    unsigned count = atol(argv[2]);
    assert(count < (BUFSIZE/sizeof(uint64_t)));
    unsigned athreads = atol(argv[3]);
    writecyclecounter("gpuwritehost", qa1, host_buffer, count, athreads);
    uint64_t old = 0;
    for (size_t i = 0; i < count; i += 1) {
      uint64_t v = host_buffer[i];
      printf("%8ld %ld, diff %ld\n", i, v, v-old);
      old = v;
    }
  }
  if (strcmp(argv[1], "gpuwritedeviceucs") == 0) {
    assert(argc == 4);
    unsigned count = atol(argv[2]);
    assert(count < (BUFSIZE/sizeof(uint64_t)));
    unsigned athreads = atol(argv[3]);
    writecyclecounter_uncached("gpuwritedeviceucs", qa1, device_buffer, count, athreads);
    qa1.memcpy(host_buffer, device_buffer, count * sizeof(uint64_t)).wait();
    uint64_t old = 0;
    for (size_t i = 0; i < count; i += 1) {
      uint64_t v = host_buffer[i];
      printf("%8ld %ld, diff %ld\n", i, v, v-old);
      old = v;
    }
  }
  if (strcmp(argv[1], "gpuwritehostucs") == 0) {
    assert(argc == 4);
    unsigned count = atol(argv[2]);
    assert(count < (BUFSIZE/sizeof(uint64_t)));
    unsigned athreads = atol(argv[3]);
    uint64_t old = 0;
    writecyclecounter_uncached("gpuwritehostucs", qa1, host_buffer, count, athreads);
    for (size_t i = 0; i < count; i += 1) {
      uint64_t v = host_buffer[i];
      printf("%8ld %ld, diff %ld\n", i, v, v-old);
      old = v;
    }
  }
  if (strcmp(argv[1], "hostwritegpu") == 0) {
    assert(argc == 3);
    unsigned count = atol(argv[2]);
    
    assert(count < (BUFSIZE/sizeof(uint64_t)));
    uint64_t start_time = rdtsc();
    #pragma omp parallel for
    for (size_t i = 0; i < count; i += 1) {
      uint64_t v[8];
      v[0] = rdtsc();
      v[1] = i;
      _movdir64b(&host_access_device_buffer[i << 3], v);
    }
    uint64_t end_time = rdtsc();
    sleep(1);
    //qa1.memcpy(host_buffer, device_buffer, count * sizeof(uint64_t)).wait();
    //sleep(1);
    uint64_t last = 0;
    for (size_t i = 0; i < count; i += 1) {
      uint64_t v = host_access_device_buffer[i<<3];
      printf("%8ld c %ld diff %ld\n", i, v, v - last);
      last = v;
    }
    printf("csv,hostwritegpu,count,threads,time\n");
    printf("csv,hostwritegpu,%u,%d,%ld\n", count,  omp_get_num_threads() , end_time-start_time);
  }
  if (strcmp(argv[1], "hostwritehost") == 0) {
    assert(argc == 3);
    unsigned count = atol(argv[2]);
    
    assert(count < (BUFSIZE/sizeof(uint64_t)));
    uint64_t start_time = rdtsc();
    #pragma omp parallel for
    for (size_t i = 0; i < count; i += 1) {
      uint64_t v[8];
      v[0] = rdtsc();
      v[1] = i;
      host_buffer[0+(i << 3)] = v[0];
      host_buffer[0+(i << 3)] = v[1];
    }
    uint64_t end_time = rdtsc();
    sleep(1);
    //qa1.memcpy(host_buffer, device_buffer, count * sizeof(uint64_t)).wait();
    //sleep(1);
    uint64_t last = 0;
    for (size_t i = 0; i < count; i += 1) {
      uint64_t v = host_buffer[i<<3];
      printf("%8ld c %ld diff %ld\n", i, v, v - last);
      last = v;
    }
    printf("csv,hostwritehost,count,threads,time\n");
    printf("csv,hostwritehost,%u,%d,%ld\n", count,  omp_get_num_threads() , end_time-start_time);
  }

  if (strcmp(argv[1], "pingpong") == 0) {
    assert(argc == 5);
    unsigned count = atol(argv[2]);
    assert(count < (BUFSIZE/sizeof(uint64_t)));
    uint64_t *hr = NULL;
    uint64_t *hw = NULL;
    uint64_t *gr = NULL;
    uint64_t *gw = NULL;
    if (strcmp(argv[3], "cpu")) {
      hw = &host_buffer[0];
      gr = &host_buffer[0];
    }
    if (strcmp(argv[3], "device")) { 
      hw = &host_access_device_buffer[0];
      gr = &device_buffer[0];
   }
    if (strcmp(argv[4], "cpu")) {
      hr = &host_buffer[8];
      gw = &host_buffer[8];
    }
    if (strcmp(argv[4], "device")) {
      hr = &host_access_device_buffer[8];
      gw = &device_buffer[8];
    }
    assert(hr != NULL);
    assert(hw != NULL);
    assert(gr != NULL);
    assert(gw != NULL);
    pingpong(qa1, gr, gw, hr, hw, count);
  }
  
  if (strcmp(argv[1], "comparetimer") == 0) {
    assert(argc == 5);
    unsigned gcount = atol(argv[2]);
    unsigned ccount = atol(argv[3]);
    unsigned athreads = atol(argv[4]);
    sycl::event ea1;
    {
      ea1 = qa1.submit([&](sycl::handler &h) {
	  h.parallel_for_work_group(sycl::range(1), sycl::range(athreads), [=](sycl::group<1> grp) {
	      grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		  int si = it.get_global_id()[0];  // send index
		  while (device_buffer[0] == 0)
		    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
		  while (si < gcount) {
		    ucs_ulong(&device_buffer[8], get_cycle_counter());
		    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
		    si += athreads;
		  }
		  ucs_ulong(&device_buffer[0],0);
		});
	    });
	});
    }
    uint64_t start[8] = {1,1,1,1,1,1,1,1};
    uint64_t stop[8] = {0,0,0,0,0,0,0,0};

    volatile uint64_t *flag = &host_access_device_buffer[0];
    volatile uint64_t *value = &host_access_device_buffer[8];
    _movdir64b((void *) flag, start);
    uint64_t old = *value;
    for (size_t i = 0; i < ccount; i += 1) {
      uint64_t v = *value;
      if (v != old) {
	uint64_t ts = rdtsc();
	host_buffer[0+(i*2)] = ts;
	host_buffer[1+(i*2)] = v;
	old = v;
      }
      if (*flag == 0) break;
    }
    ea1.wait();
    printduration("timer read", ea1);
    uint64_t oldh = 0;
    uint64_t oldd = 0;
    for (size_t i = 0; i < ccount*2; i += 1) {
      uint64_t hv = host_buffer[0+(i*2)];
      uint64_t dv = host_buffer[1+(i*2)];
	
      printf("h %8ld d %16ld %ld dh %ld dd %ld\n", i, hv, dv, hv - oldh, dv - oldd);
      oldh = hv;
      oldd = dv;
    }
  }
}
