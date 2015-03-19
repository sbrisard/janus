import sys
import time

import numpy as np

import janus.fft.serial

def benchmark(shape, niter):
    transform = janus.fft.serial.create_real(shape)
    r = np.random.uniform(-1., 1., transform.shape)
    c = np.empty(transform.oshape, dtype=np.float64)
    times = []
    for i in range(niter):
        t1 = time.perf_counter()
        transform.r2c(r, c)
        t2 = time.perf_counter()
        times.append(1E3 * (t2 - t1))

    return np.mean(times), np.std(times)

if __name__ == '__main__':

    np.random.seed(20140121)

    params = [((128, 128, 128), 15000),
              ((256, 256, 256), 10000),
              ((512, 512, 512), 1000)]

    for shape, niter in params:
        mean, std = benchmark(shape, niter)
        args = map(str, shape + (niter, 1, mean, std))
        print(','.join(args), flush=True)
