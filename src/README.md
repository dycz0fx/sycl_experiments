cpu_gpu_flow

can be compiled with or without USE_ATOMIC_READ_WR  
cpu thread writes a 1
gpu spins reading host mem cell until it goes to 1
then writes shared memory with a q
host mainline waits for kernel to exit, then prints shared value


gpu_cpu_flow

can be compiled with or without USE_ATOMIC_READ_WR  
gpu does an atomic store of a variable
cpu spins doing load



divergence_cpu

A work queue with cpu_selector is created

There's a one word buffer, and you run a parallel loop with T workers
(2)
each work item spin waits for dev_mem[0] to be equal to its index,
then adds one

divergence_cpu

a device buffer of size 3 * int64 created
host thread printing whenever buffer[1] is 0 mod 10000
device thread, each work item waits its turn and increments both
buffer[0] and buffer[1]

gpu_cpu_atomics.cpp

comment says "currently not working"

device has a delay loop incrementing a variable with atomic fetch_add
host tracks it

I suppose the ordering of the execution of the different work items is
variable and the atomic is supposed to make the sequenc monotonic

gpu_cpu_memcpy

gpu thread writes variables with fetch_add
cpu thread reads with memcpy
why should this work?  memcpy could be using non-atomic byte reads

gpu_cpu_put

GPU does puts into a host buffer with a shared poiner in device memory
host.  The shared pointer is used with fetch&add with an allocation
variable to get a buffer slot, then uses a different compare exchange
on a notification variable.

The second one is contested in order that they appear to the host in
order.   Better to put "occupied" flags in the buffer itself I
think.

gpu_cpu_put_flags

This has an array of packet objects and an array of flags
Do the flags need to be cache aligned?




My questions about SYCL
what is different between an atomic load and an ordinary load?
  answer, the atomic version has a memory ordering guarantee.  For
  example, a load with acquire semantics assures visibility of other
  data written before a corresponding store with release semantics
  A data handoff will do some data stores, then an atomic_ref store
  with release semantics of a flag.  Then the consumer who is spinning
  on the flag using atomic_ref load with acquire semantics will be
  guaranteed to see the new data once the flag changes



Needed

bandwidth and latency tests:  


RDTSC

flag only

host: set flag1 to N
device: see flag1 changed, copy to flag2
host: see that flag2 matches flag 1, continue

both exit when flag1 and flag2 advance to count-1


gpu_cpu_put_flags

allocate a volatile size_t * malloc_device

of size

2 x size_t
capacity * sizeof(packet_t)  (which is 3x8 bytes)
  and capacity is 64
another
capacity * sizeof (packet_t) 
T * 4 * size_t  where T is 1024,
size_t * capacity * 3 where capacity is

last is the base of everyting
first is &last[1]
buff if &last[2]

flag is buff + cap[acity
tmp_counter is flag+capacity
inex is 

