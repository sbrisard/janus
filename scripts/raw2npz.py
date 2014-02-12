"""Script for the conversion of raw reference data (discrete Green operators)
to *.npz.
"""

import os
import os.path
import re
import sys

import numpy as np

DEPTH = 8
RAW_EXT = '.dat'

def raw2npz(raw_filename):
    npz_filename = os.path.abspath(raw_filename).replace(RAW_EXT, '.npz')
    print('Converting {0} to {1}'.format(os.path.basename(raw_filename),
                                         os.path.basename(npz_filename)))

    pattern = re.compile('_[0-9][0-9]x[0-9][0-9]x[0-9][0-9]_')
    match = re.search(pattern, raw_filename)
    if match is not None:
        dim = 3
        sym = (dim * (dim + 1)) // 2
        n = tuple(int(i)
                  for i in match.group().lstrip('_').rstrip('_').split('x'))
        n0 = n[0]
        n1 = n[1]
        n2 = n[2]
        n3 = 2 * sym
        a = np.fromfile(raw_filename, dtype = '>f{0}'.format(DEPTH))
        a.shape = (n0, n1, n2, n3)
        a.strides = (DEPTH, DEPTH * n0, DEPTH * n0 * n1, DEPTH * n0 * n1 * n2)
        x = np.asarray(a[:, :, :, 0:sym], dtype='<f{0}'.format(DEPTH))
        y = np.asarray(a[:, :, :, sym:],  dtype='<f{0}'.format(DEPTH))

    np.savez_compressed(npz_filename, x=x, y=y)


if __name__ == '__main__':
    directory = os.path.abspath(os.path.join('..', 'janus', 'tests', 'data'))
    print(directory)

    for name in os.listdir(directory):
        if name.endswith(RAW_EXT):
            raw2npz(os.path.join(directory, name))
