# begin_imports
import numpy as np

import janus.fft.parallel

from mpi4py import MPI
# end_imports

if __name__ == '__main__':
    # begin_step_1
    comm = MPI.COMM_WORLD
    root = 0
    shape = (32, 64)

    if comm.rank == root:
        np.random.seed(20150310)
        x = np.random.rand(*shape)
    else:
        x = None
    # end_step_1
    # begin_step_2
    transform = janus.fft.parallel.create_real(shape, comm)
    if comm.rank == root:
        print('shape  = {}'.format(transform.shape))
        print('rshape = {}'.format(transform.rshape))
        print('cshape = {}'.format(transform.cshape))
    # end_step_2
    # begin_step_3
    local_shapes = comm.gather(sendobj=(transform.rshape[0],
                                        transform.offset0),
                               root=0)
    # end_step_3
    # begin_step_4
    x_loc = np.empty(transform.rshape,
                     dtype=np.float64)
    comm.Scatterv(x, x_loc, root)
    # end_step_4
    # begin_step_5
    y_loc = transform.r2c(x_loc)
    # end_step_5
    # begin_step_6
    if comm.rank == root:
        y = np.empty((transform.shape[0],) + transform.cshape[1:],
                     dtype=np.float64)
    else:
        y = None
    comm.Gatherv(y_loc, y, root)
    # end_step_6
    # begin_step_7
    if comm.rank == root:
        serial_transform = janus.fft.serial.create_real(shape)
        y_ref = np.asarray(serial_transform.r2c(x))
        err = np.sum((y-y_ref)**2) / np.sum(y_ref**2)
        assert err == 0.0
    # end_step_7
