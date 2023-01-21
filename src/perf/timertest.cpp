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
#include <getopt.h>


#include "uncached.cpp"
#include "syclutilities.cpp"
#include <immintrin.h>
#include <omp.h>
    
#include "../common_includes/rdtsc.h"


enum { cmd_count=1, cmd_stride, cmd_threads, cmd_bufloc, cmd_help, cmd_op, cmd_from, cmd_end=-1};
const char *cmd_str[] =  {"nil", "count", "stride", "threads", "bufloc", "help", "op", "from", 0};

constexpr int buflocN = 5;
constexpr int fromN = 5;
enum { loc_c=1, loc_p0, loc_p1, loc_t0, loc_t1, loc_end = -1};
const char *loc_str[] = {"nil", "c", "p0", "p1", "t0", "t1", 0};

enum { op_load=1, op_store, op_ucs, op_ucl, op_pingpong, op_comparetimer, op_timeread, op_end = -1};
const char *op_str[] = {"nil", "load", "store", "ucs", "ucl", "pingpong", "comparetimer", "timeread", 0};

int str_to_code(const char **table, const char *s)
{
  int code = 0;
  while (table[code] != 0) {
    if (strcmp(table[code], s) == 0) return(code);
    code += 1;
  }
  printf ("unknown value %s\n", s);
  printf ("expecting one of ");
  code = 0;
  while (table[code] != 0) {
    printf(" %s", table[code]);
    code += 1;
  }
  printf("\n");
  exit(1);
  return -1;
}

const char *code_to_str(const char **table, int code)
{
  int len = 0;
  while (table[len]) len += 1;
  if (code < 0) return ("unknown");
  if (code >= len) return ("unknown");
  return(table[code]);
}

struct Command {
  const char *name;
  int has_arg;
  int *flag;
  int val;
  const char **options;
  const char *help;
};

struct Command cmdtable[8] {
  {"count", required_argument, nullptr, cmd_count, nullptr, "how many items"},
  {"stride", required_argument, nullptr, cmd_stride, nullptr, "stride between items"},
  {"threads", required_argument, nullptr, cmd_threads, nullptr, "gpu threads"},
  {"bufloc", required_argument, nullptr, cmd_bufloc, loc_str, "buffer location"},
  {"op", required_argument, nullptr, cmd_op, op_str, "operation"},
  {"from", required_argument, nullptr, cmd_from, loc_str, "who runs operation"},
  {"help", no_argument, nullptr, cmd_help, nullptr, "print this"},
  {nullptr, no_argument, nullptr, 0, nullptr, nullptr}
};

struct option long_opts[8];
    

int g_count = 100;
int g_threads = 1;
int g_stride = 1;
int g_bufloc = loc_c;
int g_op = op_ucs;
int g_from = loc_c;

void Usage()
{
  int i = 0;
  while(cmdtable[i].name) {
    printf("%s ( %s ) ", cmdtable[i].name, cmdtable[i].help);
    if (cmdtable[i].has_arg == no_argument) printf("no argument");
    if (cmdtable[i].has_arg == optional_argument) printf("optional argument");
    if (cmdtable[i].has_arg == required_argument) {
      if (cmdtable[i].options == nullptr) printf("integer argument");
      else {
	printf("options: ");
	int opt = 0;
	while (cmdtable[i].options[opt]) {
	  printf(" %s", cmdtable[i].options[opt]);
	  opt += 1;
	}
      }
    }
    printf("\n");
    i = i + 1;
  }
}

void ProcessArgs(int argc, char **argv)
{
  option long_opts[8];
  for (int i = 0; i < 8; i += 1) {
    long_opts[i].name = cmdtable[i].name;
    long_opts[i].has_arg = cmdtable[i].has_arg;
    long_opts[i].flag = cmdtable[i].flag;
    long_opts[i].val = cmdtable[i].val;
  };
  
  while (true) {
    const auto opt = getopt_long(argc, argv, "", long_opts, nullptr);
    if (-1 == opt)
      break;
    switch (opt) {
    case cmd_count: {
      g_count = std::stoi(optarg);
      printf("count=%u\n", g_count);
      break;
    }
    case cmd_stride: {
      g_stride = std::stoi(optarg);
      printf("stride=%u\n", g_stride);
      break;
    }
    case cmd_threads: {
      g_threads = std::stoi(optarg);
      printf("threads=%u\n", g_threads);
      break;
    }
    case cmd_bufloc: {
      g_bufloc = str_to_code(loc_str, optarg);
      printf("bufloc=%s\n", loc_str[g_bufloc]);
      break;
    }
    case cmd_from: {
      g_from = str_to_code(loc_str, optarg);
      printf("from=%s\n", loc_str[g_from]);
      break;
    }
    case cmd_op: {
      g_op = str_to_code(op_str, optarg);
      printf("op=%s\n", op_str[g_op]);
      break;
    }
    case cmd_help: {
      Usage();
      exit(0);
      break;
    }
    default: {
      Usage();
      exit(1);
    }
    }
  }
  printf("%s op=%s from=%s bufloc=%s count=%u stride=%u threads=%u\n",
	 argv[0], op_str[g_op], loc_str[g_from], loc_str[g_bufloc], g_count, g_stride, g_threads);
}

void writecyclecounter(const char *name, sycl::queue q, uint64_t *buffer, size_t count, unsigned stride, int threads)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
	  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
	      int si = it.get_global_id()[0] * stride;  // send index
	      ptrdiff_t advance = threads * stride;
	      while (si < count) {
		buffer[si] = get_cycle_counter();
		si += advance;
	      }
	    });
	});
    });
  e.wait();
  printduration("writecyclecounter", e);
}

void readcyclecounter(const char *name, sycl::queue q, uint64_t *buffer, size_t count, unsigned stride, int threads)
{
  uint64_t start, end;
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
	  uint64_t start = get_cycle_counter();
	  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
	      int si = it.get_global_id()[0] * stride;  // send index
	      ptrdiff_t advance = threads * stride;
	      while (si < count) {
		volatile uint64_t v = buffer[si];
		si += advance;
	      }
	    });
	  uint64_t end = get_cycle_counter();
	  buffer[0] = end-start;
	});
    });
  e.wait();
  printduration(name, e);
}

void writecyclecounter_uncached(const char *name, sycl::queue q, uint64_t *buffer, size_t count, unsigned stride, int threads)
{
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
	  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
	      int si = it.get_global_id()[0] * stride;  // send index
	      ptrdiff_t advance = threads * stride;
	      while (si < count) {
		ucs_ulong(&buffer[si], get_cycle_counter());
		si += advance;
	      }
	    });
	});
    });
  e.wait();
  printduration("writecyclecounter", e);
}

void readcyclecounter_uncached(const char *name, sycl::queue q, uint64_t *buffer, size_t count, unsigned stride, int threads)
{
  uint64_t start, end;
  auto e = q.submit([&](sycl::handler &h) {
      h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
	  uint64_t start = get_cycle_counter();
	  grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
	      int si = it.get_global_id()[0] * stride;  // send index
	      ptrdiff_t advance = threads * stride;
	      while (si < count) {
		volatile uint64_t v = ucl_ulong(&buffer[si]);
		si += advance;
	      }
	    });
	  uint64_t end = get_cycle_counter();
	  buffer[0] = end-start;
	});
    });
  e.wait();
  printduration(name, e);
}

void printdelta(uint64_t *buffer, unsigned count, unsigned stride) {
  uint64_t old = 0;
  for (size_t i = 0; i < count; i += stride) {
    uint64_t v = buffer[i];
    printf("%8ld %ld, diff %ld\n", i, v, v-old);
    old = v;
  }
}

void hostwrite_uncached(const char *name, uint64_t *buffer, size_t count, unsigned stride, int threads)
{
  uint64_t start_time = rdtsc();
  uint64_t offset = 0;
#pragma omp parallel for
  for (size_t i = 0; i < count; i += 1) {
    uint64_t v[8];
    v[0] = rdtsc();
    v[1] = i;
    _movdir64b(&buffer[offset], v);
    offset += stride;
  }
  uint64_t end_time = rdtsc();
  printdelta(buffer, count, stride);
  printf("csv,hostwritegpu,count,threads,time\n");
  printf("csv,hostwritegpu,%lu,%d,%ld\n", count,  omp_get_num_threads() , end_time-start_time);
}

void hostwrite(const char *name, uint64_t *buffer, size_t count, unsigned stride, int threads)
{
  uint64_t start_time = rdtsc();
  uint64_t offset = 0;
#pragma omp parallel for
  for (size_t i = 0; i < count; i += 1) {
    uint64_t v[8];
    v[0] = rdtsc();
    v[1] = i;
    buffer[offset + 0] = v[0];
    buffer[offset + 1] = v[1];
    offset += stride;
  }
  uint64_t end_time = rdtsc();
  printdelta(buffer, count, stride);
  printf("csv,hostwritegpu,count,threads,time\n");
  printf("csv,hostwritegpu,%lu,%d,%ld\n", count,  omp_get_num_threads() , end_time-start_time);
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
  ProcessArgs(argc, argv);
  
  unsigned count = g_count;
  unsigned threads = g_threads;
  unsigned stride = g_stride;

  assert(count * stride * sizeof(uint64_t) <= BUFSIZE);
  
  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};

  sycl::queue qa1 = sycl::queue(sycl::gpu_selector{}, prop_list);
  
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

  uint64_t *buffer, *check_buffer;
  if (g_bufloc == loc_c) {
    buffer = host_buffer;
    check_buffer = host_buffer;
  } else {
    if (g_from == loc_c) {
      buffer = host_access_device_buffer;
    } else {
      buffer = device_buffer;
    }
    check_buffer = host_access_device_buffer;
  }
  sycl::queue q = qa1;

  if (g_from == loc_c) {
    switch (g_op) {
    case op_store: {
      hostwrite(op_str[g_op], buffer, count, stride, threads);
      break;
    }
    case op_ucs: {
      hostwrite_uncached(op_str[g_op], buffer, count, stride, threads);
      break;
    }
    case op_load: {
      readcyclecounter(op_str[g_op], q, buffer, count, stride, threads);
      printf("%s count %u cycles %ld\n", op_str[g_op], count, check_buffer[0]);
      break;
    }
    }
  } else {
    switch (g_op) {
    case op_store: {
      writecyclecounter(op_str[g_op], q, buffer, count, stride, threads);
      printdelta(check_buffer, count, stride);
      break;
    }
    case op_ucs: {
      writecyclecounter_uncached(op_str[g_op], q, buffer, count, stride, threads);
      printdelta(check_buffer, count, stride);
      break;
    }
    case op_load: {
      readcyclecounter(op_str[g_op], q, buffer, count, stride, threads);
      printf("%s count %u cycles %ld\n", op_str[g_op], count, check_buffer[0]);
      break;
    }
    case op_ucl: {
      readcyclecounter_uncached(op_str[g_op], q, buffer, count, stride, threads);
      printf("%s count %u cycles %ld\n", op_str[g_op], count, check_buffer[0]);
      break;
    }
    case op_pingpong: {
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
      pingpong(q, gr, gw, hr, hw, count);
      break;
    }
    }
  }
  
  
  if (g_op == op_comparetimer) {
    sycl::event ea1;
    {
      ea1 = qa1.submit([&](sycl::handler &h) {
	  h.parallel_for_work_group(sycl::range(1), sycl::range(threads), [=](sycl::group<1> grp) {
	      grp.parallel_for_work_item([&] (sycl::h_item<1> it) {
		  int si = it.get_global_id()[0];  // send index
		  while (device_buffer[0] == 0)
		    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
		  while (si < count) {
		    ucs_ulong(&device_buffer[8], get_cycle_counter());
		    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
		    si += threads;
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
    for (size_t i = 0; i < count; i += 1) {
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
    for (size_t i = 0; i < count*2; i += 1) {
      uint64_t hv = host_buffer[0+(i*2)];
      uint64_t dv = host_buffer[1+(i*2)];
	
      printf("h %8ld d %16ld %ld dh %ld dd %ld\n", i, hv, dv, hv - oldh, dv - oldd);
      oldh = hv;
      oldd = dv;
    }
  }
}
