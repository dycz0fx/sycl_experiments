// https://www.man7.org/linux/man-pages/man3/shm_open.3.html
// mpirun -prepend-rank -n 4 ./shm_barrier

#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>
#include <mpi.h>
#include <atomic>

#define TYPE size_t

const char *shmpath = "/myshm";

void shm_barrier(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int fd;
    if (rank == 0) {
        fd = shm_open(shmpath, O_CREAT | O_EXCL | O_RDWR, 0600);
        write(fd, 0, sizeof(TYPE));
        MPI_Barrier(comm);
    }
    else {
        MPI_Barrier(comm);
        fd = shm_open(shmpath, O_RDWR, 0);
    }
    assert(fd >= 0);

    int ret = ftruncate(fd, sizeof(TYPE));
    assert(ret >= 0);

    TYPE *sem = (TYPE *)mmap(NULL, sizeof(TYPE), PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
    assert(sem != MAP_FAILED);

    auto atomic_val = reinterpret_cast<std::atomic<size_t>*>(sem);

    atomic_val->operator++();

    while(atomic_val->load() < size);

    if (rank == 0) {
        ret = shm_unlink(shmpath);
        assert(ret >= 0);
    }
}

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(name, &len);

    printf("Hello host %s, rank %d, size %d\n", name, rank, size);

    if (rank == 0) {
        shm_unlink(shmpath);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    shm_barrier(MPI_COMM_WORLD);
    shm_barrier(MPI_COMM_WORLD);
    shm_barrier(MPI_COMM_WORLD);

    MPI_Finalize();
}
