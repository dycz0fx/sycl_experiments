gpu_cpu_flow


cpu thread writes a 1
gpu spins reading host mem cell until it goes to 1
then writes shared memory with a q
host mainline waits for kernel to exit, then prints shared value



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
