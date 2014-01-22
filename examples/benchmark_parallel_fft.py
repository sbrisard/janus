import sys
import time

import numpy as np
import numpy.random as nprnd

sys.path.append('../src')
import fft.parallel

from mpi4py import MPI

def benchmark(n, niter):
    transform = fft.parallel.create_real((n, n), comm)
    local_sizes = comm.gather((transform.rshape[0], transform.offset0))

    if comm.rank == 0:
        r = nprnd.uniform(-1., 1., transform.shape)
        rlocs = [r[offset0:offset0 + n0] for n0, offset0 in local_sizes]
    else:
        rlocs = None
    rloc = comm.scatter(rlocs)
    cloc = np.empty(transform.cshape, dtype=np.float64)

    times = []
    for i in range(niter):
        t1 = time.perf_counter()
        transform.r2c(rloc, cloc)
        t2 = time.perf_counter()
        times.append(t2 - t1)

    return np.mean(times), np.std(times)

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    fft.parallel.init()
    nprnd.seed(20140121)

    #params = [64, 128, 256, 512, 1024, 2048, 4096]
    #params = [(64, 50000), (128, 10000), (256, 5000), (512, 5000), (1024, 100), (2048, 100)]
    params = [(4096, 100)]
    
    for n, niter in params:
        """
        gathered = comm.gather(benchmark(n, max_err, max_iter))
        if gathered is not None:
            dummy = list(zip(*gathered))
            mean = np.
        """
        mean, std = benchmark(n, niter)
        err = 2.6 * std / np.sqrt(niter)
        print(n, mean, niter, err, err / mean * 100)

    MPI.Finalize()
