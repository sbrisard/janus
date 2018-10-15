import argparse
import inspect
import json

import h5py
import numpy as np
import pytest

from numpy.testing import assert_array_equal


def ellipsoid(radii, size, shape):
    """Discretize a 2D/3D ellipsoid over a cartesian grid.

    `radii`, `size` and `shape` are sequences of length 2 (for
    ellipses) or 3 (for ellipsoids).

    The `radii` of the ellipsoid are specified as a sequence of 2 or 3
    floats, in true units of length.

    The `size` of the returned grid is specified as a sequence of 2 or
    3 floats, in true units of length. The generated ellipsoid is
    centered in the grid.

    The `shape` of the grid (number of cells along each side) is
    specified as a sequence of 2 or 3 ints.

    The returned array is a set of integer values. Cells that are
    centered outside (resp. inside) the ellipsoid are set to 0
    (resp. 1).
    """
    slices = [slice(-0.5*L*(n-1)/n, 0.5*L*(n-1)/n, n*1j)
              for L, n in zip(size, shape)]
    coords = np.mgrid[slices]
    # Reshape `coords` to allow for broadcasting with `start` and `radii`.
    coords = np.moveaxis(coords, 0, -1)
    flag = np.sum((coords/radii)**2, axis=-1) <= 1
    return flag.astype(np.int8)


def ref_ellipse(radii, size, shape):
    a, b = radii
    Lx, Ly = size
    Nx, Ny = shape
    x_max = .5*Lx*(Nx-1)/Nx
    y_max = .5*Ly*(Ny-1)/Ny
    x = np.linspace(-x_max, x_max, num=Nx)
    y = np.linspace(-y_max, y_max, num=Ny)
    x, y = np.meshgrid(x, y, indexing='ij')
    return ((x/a)**2+(y/b)**2 <= 1).astype(np.int8)


def ref_ellipsoid(radii, size, shape):
    a, b, c = radii
    Lx, Ly, Lz = size
    Nx, Ny, Nz = shape
    x_max = .5*Lx*(Nx-1)/Nx
    y_max = .5*Ly*(Ny-1)/Ny
    z_max = .5*Lz*(Nz-1)/Nz
    x = np.linspace(-x_max, x_max, num=Nx)
    y = np.linspace(-y_max, y_max, num=Ny)
    z = np.linspace(-z_max, z_max, num=Nz)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    return ((x/a)**2+(y/b)**2+(z/c)**2 <= 1).astype(np.int8)


@pytest.mark.parametrize('radii, size, shape',
                         [((0.3, 0.4), (1.0, 1.0), (100, 100)),
                          ((0.3, 0.4), (0.8, 1.0), (80, 100)),
                          ((0.3, 0.4), (0.6, 0.8), (60, 80)),
                          ((0.3, 0.4), (0.4, 0.6), (40, 60)),
                          ((0.3, 0.4), (0.4, 0.6), (50, 50)),
                          ((0.3, 0.4, 0.5), (1.0, 1.0, 1.0), (100, 100, 100)),
                          ((0.3, 0.4, 0.5), (0.8, 1.0, 1.2), (80, 100, 120)),
                          ((0.3, 0.4, 0.5), (0.6, 0.8, 1.0), (60, 80, 100)),
                          ((0.3, 0.4, 0.5), (0.4, 0.6, 0.8), (40, 60, 80)),
                          ((0.3, 0.4, 0.5), (0.4, 0.6, 0.8), (50, 50, 50))])
def test_ellipsoid(radii, size, shape):
    actual = ellipsoid(radii, size, shape)
    if len(radii) == 2:
        expected = ref_ellipse(radii, size, shape)
    elif len(radii) == 3:
        expected = ref_ellipsoid(radii, size, shape)
    else:
        raise ValueError('length of sequences must be 2 or 3')
    assert_array_equal(actual, expected)


def create_parser():
    description='DICRETIZATION OF ELLIPSES/ELLIPSOIDS\n\n'
    epilog = ('Docstring of the ellipsoid function\n'+
              '-----------------------------------\n\n'+
              inspect.getdoc(ellipsoid)+'\n\n'+
              'Format of the input file\n'+
              '------------------------\n\n'+
              'Parameters of the simulation are passed as a JSON file.\n'+
              'Here is a short example::\n\n'
              '    {"output": "example.h5",\n'+
              '     "radii": [0.2, 0.3, 0.4],\n'+
              '     "size": [1.0, 1.0, 1.0],\n'+
              '     "shape": [128, 128, 128]}\n\n'+
              'where `radii`, `size` and `shape` have already been defined, and `output`\n'+
              'is the full path to the output file to be created.\n\n'+
              'Format of the output file\n'+
              '-------------------------\n\n'+
              'The discretized ellipse/ellipsoid is output as a HDF5 file, with 3 datasets\n\n'+
              '  - radii: an array of length 2 or 3,\n'+
              '  - size: an array of length 2 or 3,\n'+
              '  - phase: an array of specified shape. The value of cells that are centered\n'+
              'inside (resp. outside) the ellipse/ellipsoid is set to 1 (resp. 0).')
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input',
                        help='full path to the JSON parameter file')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    with open(args.input, 'r') as f:
        data = json.load(f)

    with h5py.File(data['output'], 'w') as f:
        f.create_dataset('radii', data=data['radii'])
        f.create_dataset('size', data=data['size'])
        f['phase'] = ellipsoid(data['radii'], data['size'], data['shape'])
