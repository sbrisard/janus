"""Script to convert the reference data for validation of the discrete Green
operator. In the initial (*.npy) format, the tau and eta arrays are stacked
in one single array prior to saving.
In the new format, the tau and eta arrays are saved as 'x' and 'y' in a
*.npz file.

"""

import os
import os.path
import sys

import numpy as np

def npy2npz(npy_filename):
    npz_filename = os.path.abspath(npy_filename).replace('.npy', '.npz')
    print('Converting {0} to {1}'.format(os.path.basename(npy_filename),
                                         os.path.basename(npz_filename)))
    xy = np.load(npy_filename)
    n = xy.shape[:-1]

    print('Shape of array: {}'.format(xy.shape))

    dim = len(n)
    sym = (dim * (dim + 1)) // 2

    if dim == 2:
        x = xy[:, :, 0:sym]
        y = xy[:, :, sym:]
    else:
        x = xy[:, :, :, 0:sym]
        y = xy[:, :, :, sym:]

    np.savez_compressed(npz_filename, x=x, y=y)

if __name__ == '__main__':
    directory = os.path.abspath(os.path.join('..', 'janus', 'tests', 'data'))
    print(directory)

    for name in os.listdir(directory):
        if name.endswith('.npy'):
            npy2npz(os.path.join(directory, name))
