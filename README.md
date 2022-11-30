GPU - CPU Bufffer
=================

Experiments to pass data from GPU to CPU

Build Instructions
------------------
```
$ mkdir build
$ cd build
$ cmake ..

[LCS]
 MKL_DIR=/opt/intel/inteloneapi/mkl/latest/lib/cmake/mkl
 TBB_DIR=/opt/intel/inteloneapi/tbb/latest/lib/cmake/tbb cmake ..
 
$ cmake --build .
$ ctest
```

To run tests manually:
```
$ srun ./src/EXE
```

