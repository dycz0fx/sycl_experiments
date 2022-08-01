gpu_cpu_flow


cpu thread writes a 1
gpu spins reading host mem cell until it goes to 1
then writes shared memory with a q
host mainline waits for kernel to exit, then prints shared value

