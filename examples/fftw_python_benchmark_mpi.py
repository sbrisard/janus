import sys
import time

import numpy as np

import janus.fft.parallel

from mpi4py import MPI

def benchmark(shape, niter):
    comm = MPI.COMM_WORLD
    root = 0
    transform = janus.fft.parallel.create_real(shape, comm)
    local_sizes = comm.gather((transform.rshape[0], transform.offset0))

    if comm.rank == root:
        r = np.random.uniform(-1., 1., transform.shape)
    else:
        r= None
    rloc = np.empty(transform.rshape, dtype=np.float64)
    comm.Scatterv(r, rloc, root)
    cloc = np.empty(transform.cshape, dtype=np.float64)

    times = []
    for i in range(niter):
        t1 = time.perf_counter()
        transform.r2c(rloc, cloc)
        t2 = time.perf_counter()
        times.append(1E3 * (t2 - t1))

    return np.mean(times), np.std(times)

if __name__ == '__main__':

    janus.fft.parallel.init()
    np.random.seed(20140121)

    params = [((128, 128, 128), 15000),
              ((256, 256, 256), 10000),
              ((512, 512, 512), 1000)]

    for shape, niter in params:
        mean, std = benchmark(shape, niter)
        if MPI.COMM_WORLD.rank == 0:
            args = map(str, shape + (niter, MPI.COMM_WORLD.size, mean, std))
            print(','.join(args), flush=True)

    MPI.Finalize()
