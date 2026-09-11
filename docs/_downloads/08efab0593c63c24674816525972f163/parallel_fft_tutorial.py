# Imports
import numpy as np

import janus.fft.parallel

from mpi4py import MPI

# Init some variables
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    root = 0
    shape = (32, 64)
    # Create transform objects
    transform = janus.fft.parallel.create_real(shape, comm)
    if comm.rank == root:
        print('global_ishape  = {}'.format(transform.global_ishape))
        print('global_oshape  = {}'.format(transform.global_oshape))
        print('ishape = {}'.format(transform.ishape))
        print('oshape = {}'.format(transform.oshape))
    # Prepare communications
    counts_and_displs = comm.gather(sendobj=(transform.isize, transform.idispl,
                                             transform.osize, transform.odispl),
                                    root=root)
    if comm.rank == root:
        np.random.seed(20150310)
        x = np.random.rand(*shape)
        icounts, idispls, ocounts, odispls = zip(*counts_and_displs)
    else:
        x, icounts, idispls, ocounts, odispls = None, None, None, None, None
    # Scatter input data
    x_loc = np.empty(transform.ishape, dtype=np.float64)
    comm.Scatterv([x, icounts, idispls, MPI.DOUBLE], x_loc, root)
    # Execute transform
    y_loc = transform.r2c(x_loc)
    # Gather output data
    if comm.rank == root:
        y = np.empty(transform.global_oshape, dtype=np.float64)
    else:
        y = None
    comm.Gatherv(y_loc, [y, ocounts, odispls, MPI.DOUBLE], root)
    # Validate result
    if comm.rank == root:
        serial_transform = janus.fft.serial.create_real(shape)
        y_ref = np.asarray(serial_transform.r2c(x))
        err = np.sum((y-y_ref)**2) / np.sum(y_ref**2)
        assert err <= np.finfo(np.float64).eps
