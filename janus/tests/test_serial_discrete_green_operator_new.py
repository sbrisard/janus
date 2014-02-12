import inspect
import itertools
import operator
import os.path
import sys
import unittest

import numpy as np

# TODO This ugly hack should be removed
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..',
                             '..'))

import janus.fft.serial
import janus.greenop as greenop

from numpy.testing import assert_array_almost_equal_nulp
from numpy.testing import assert_allclose

from janus.discretegreenop import DiscreteGreenOperator2D
from janus.discretegreenop import DiscreteGreenOperator3D
from janus.discretegreenop import TruncatedGreenOperator2D
from janus.discretegreenop import TruncatedGreenOperator3D
from janus.discretegreenop import FilteredGreenOperator2D
from janus.matprop import IsotropicLinearElasticMaterial as Material

# The value of one unit in the last place, for the float64 format.
ULP = np.finfo(np.float64).eps

# All tests are performed with discrete Green operators based on these grids
GRID_SIZES = ([(8, 8), (8, 16), (16, 8), (4, 4, 4)]
              + list(itertools.permutations((4, 8, 16))))

def add_parameterized_test(test, parameters, dct):
    digits = int(np.ceil(np.log10(len(parameters))))
    name_template = test.__name__ + '_{{:0{}d}}'.format(digits)
    args = inspect.getfullargspec(test).args
    num_args = len(args)
    doc_template = ''.join(arg + ' = {}, '
                           for arg in args if arg != 'self') .rstrip(', ')
    i = 0
    for x in parameters:
        i += 1
        def utest(self, x=x):
            return test(self, *x)
        x_default = x + tuple(itertools.repeat(None, num_args - len(x)))
        utest.__doc__ = doc_template.format(*x_default)
        dct[name_template.format(i)] = utest

def get_base(a):
    # TODO This is a hack to avoid writing uggly things like a.base.base.base.
    if a.base is None:
        return a
    else:
        return get_base(a.base)

def multi_indices(n):
    """Return the list of all multi-indices within the specified bounds.

    Return the list of multi-indices ``[b[0], ..., b[dim - 1]]`` such that
    ``0 <= b[i] < n[i]`` for all i.

    """
    iterables = [range(ni) for ni in n]
    return [np.asarray(b, dtype=np.intc)
            for b in itertools.product(*iterables)]

class DiscreteGreenOperatorTestMetaclass(type):
    def __new__(cls, name, bases, dct, test_apply_params):
        return type.__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct, test_apply_params):
        pass

    @classmethod
    def __prepare__(metacls, name, bases, test_apply_params):
        dct = super().__prepare__(name, bases)

        #
        # Test of DiscreteGreenOperator.__init__()
        #
        g2 = greenop.create(Material(0.75, 0.3, 2))
        g3 = greenop.create(Material(0.75, 0.3, 3))

        params = [((9, 9), 1., g3),
                  ((-1, 9), 1., g2),
                  ((9, -1), 1., g2),
                  ((9, 9), -1., g2),
                  ((9, 9), 1., g2, janus.fft.serial.create_real((8, 9))),
                  ((9, 9), 1., g2, janus.fft.serial.create_real((9, 8))),
                  ((9, 9, 9), 1., g2),
                  ((-1, 9, 9), 1., g3),
                  ((9, -1, 9), 1., g3),
                  ((9, 9, -1), 1., g3),
                  ((9, 9, 9), -1., g3)]

        add_parameterized_test(metacls.test_init_invalid_params,
                               params, dct)

        #
        # Test to_memoryview()
        #
        params = [(n, inplace) for n in GRID_SIZES
                               for inplace in [True, False]]
        add_parameterized_test(metacls.test_to_memoryview,
                               params, dct)

        params = [(dim, i, 1 - i) for dim in [2, 3] for i in [0, 1]]
        add_parameterized_test(metacls.test_to_memoryview_invalid_parameters,
                               params, dct)

        #
        # Test of apply_by_freq()
        #
        x = [np.array([0.3, -0.4, 0.5]),
             np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])]
        params = [(n, x[len(n) - 2], flag) for n in GRID_SIZES
                                           for flag in range(3)]
        add_parameterized_test(metacls.test_apply_by_freq, params, dct)

        params = [(dim, i, 1 - i) for dim in [2, 3] for i in [0, 1]]
        add_parameterized_test(metacls.test_apply_by_freq_invalid_params,
                               params, dct)

        #
        # Test of apply()
        #
        add_parameterized_test(metacls.test_apply,
                               test_apply_params,
                               dct)

        delta_shape3 = set(itertools.permutations([1, 0, 0]))
        delta_shape4 = set(itertools.permutations([1, 0, 0, 0]))
        params = ([(2, (0, 0, 0), i) for i in delta_shape3] +
                  [(2, i, (0, 0, 0)) for i in delta_shape3] +
                  [(3, (0, 0, 0, 0), i) for i in delta_shape4] +
                  [(3, i, (0, 0, 0, 0)) for i in delta_shape4])
        add_parameterized_test(metacls.test_apply_invalid_params,
                               params, dct)

        return dct

    def test_init_invalid_params(self, n, h, greenc, transform=None):
        """n = {0}, h = {1}, greenc = {2}

        """
        self.assertRaises(ValueError, self.discrete_green_operator,
                          n, h, greenc, transform)

    def test_to_memoryview(self, n, inplace):
        """n = {0}, inplace = {1}

        """
        greend = self.discrete_green_operator(n, 1.0)
        greenc = greend.green
        k = np.empty((len(n),), dtype=np.float64)

        for b in multi_indices(n):
            expected = self.to_memoryview_expected(greenc, n, b)
            greend.set_frequency(b)

            if inplace:
                base = np.empty_like(expected)
                actual = greend.to_memoryview(base)
                assert get_base(actual) is base
            else:
                actual = greend.to_memoryview()
            assert_allclose(actual, expected, ULP, ULP)

    def test_to_memoryview_invalid_parameters(self, dim,
                                              delta_nrows, delta_ncols):
        n = [2**(i + 3) for i in range(dim)]
        green = self.discrete_green_operator(n)
        out = np.zeros((green.oshape[-1] + delta_nrows,
                        green.ishape[-1] + delta_ncols),
                        dtype=np.float64)
        self.assertRaises(ValueError, green.to_memoryview, out)

    def test_apply_by_freq(self, n, x, flag):
        """flag allows the specification of various calling sequences:
          - flag = 0: apply_single_freq(x)
          - flag = 1: apply_single_freq(x, x)
          - flag = 2: apply_single_freq(x, y)
        """
        dim = len(n)
        green = self.discrete_green_operator(n, 1.)

        for b in multi_indices(n):
            green.set_frequency(b)
            g = np.asarray(green.to_memoryview())
            # expected is by default a matrix, so that it has two dimensions.
            # First convert to ndarray so as to reshape is to a 1D array.
            expected = np.dot(g, x)
            if flag == 0:
                base = None
            elif flag == 1:
                base = x
            elif flag == 2:
                base = np.empty_like(expected)
            else:
                raise(ValueError)
            actual = green.apply_by_freq(x, base)
            if flag != 0:
                assert get_base(actual) is base
            assert_allclose(actual, expected, ULP, ULP)

    def test_apply_by_freq_invalid_params(self, dim,
                                          delta_isize, delta_osize):
        n = tuple(2**(i + 3) for i in range(dim))
        green = self.discrete_green_operator(n, 1.)
        x = np.zeros((green.ishape[-1] + delta_isize,), dtype=np.float64)
        y = np.zeros((green.oshape[-1] + delta_osize,), dtype=np.float64)
        self.assertRaises(ValueError, green.apply_by_freq, x, y)

    #
    # Test of apply()
    #
    def test_apply(self, path_to_ref, rtol):
        npz_file = np.load(path_to_ref)
        x = npz_file['x']
        expected = npz_file['y']
        # This is a workaround to avoid
        # ResourceWarning: unclosed file <_io.BufferedReader name='xxx.npz'>
        #     self.zip.close()
        npz_file.zip.fp.close()
        n = x.shape[:-1]
        transform = janus.fft.serial.create_real(n)
        green = self.discrete_green_operator(n, 1., transform=transform)

        actual = np.zeros(transform.rshape + (green.oshape[-1],), np.float64)
        green.apply(x, actual)

        assert_allclose(actual, expected, rtol, 10 * ULP)

    def test_apply_invalid_params(self, dim,
                                  delta_ishape, delta_oshape):
        n = tuple(2**(i + 3) for i in range(dim))
        transform = janus.fft.serial.create_real(n)
        green = self.discrete_green_operator(n, 1., transform=transform)
        xshape = tuple(map(operator.add, green.ishape, delta_ishape))
        yshape = tuple(map(operator.add, green.oshape, delta_oshape))
        x = np.zeros(xshape, dtype=np.float64)
        y = np.zeros(yshape, dtype=np.float64)
        self.assertRaises(ValueError, green.apply, x, y)


def truncated_test_apply_params():
    directory = os.path.dirname(os.path.realpath(__file__))
    template_2D = ('truncated_green_operator_200x300_'
                   'unit_tau_{0}_10x10+95+145.npz')
    template_3D = ('truncated_green_operator_40x50x60_'
                   'unit_tau_{0}_10x10x10+15+20+25.npz')
    params = [(template_2D.format('xx'), ULP),
              (template_2D.format('yy'), ULP),
              (template_2D.format('xy'), ULP),
              (template_3D.format('xx'), ULP),
              (template_3D.format('yy'), ULP),
              (template_3D.format('zz'), ULP),
              (template_3D.format('yz'), ULP),
              (template_3D.format('zx'), ULP),
              (template_3D.format('xy'), ULP),
               ]
    return [(os.path.join(directory, 'data', filename), rel_err)
            for filename, rel_err in params]


class TruncatedGreenOperatorTest(unittest.TestCase,
                                 metaclass=DiscreteGreenOperatorTestMetaclass,
                                 test_apply_params=truncated_test_apply_params()):

    def discrete_green_operator(self, n, h = 1., greenc=None, transform=None):
        if greenc is None:
            mat = Material(0.75, 0.3, len(n))
            greenc = janus.greenop.create(mat)
        if len(n) == 2:
            return TruncatedGreenOperator2D(greenc, n, h, transform)
        elif len(n) == 3:
            return TruncatedGreenOperator3D(greenc, n, h, transform)
        else:
            raise ValueError('dimension must be 2 or 3 '
                             '(was {})'.format(len(n)))

    def to_memoryview_expected(self, greenc, n, b):
        n = np.asarray(n)
        k = np.empty((len(n),), dtype=np.float64)
        i = 2 * b > n
        k[i] = 2. * np.pi * (b[i] - n[i]) / n[i]
        k[~i] = 2. * np.pi * b[~i] / n[~i]

        greenc.set_frequency(k)
        return greenc.to_memoryview()


def filtered_test_apply_params():
    directory = os.path.dirname(os.path.realpath(__file__))
    template_2D = ('filtered_green_operator_200x300_'
                   'unit_tau_{0}_10x10+95+145.npz')
    template_3D = ('filtered_green_operator_40x50x60_'
                   'unit_tau_{0}_10x10x10+15+20+25.npz')
    params = [(template_2D.format('xx'), 2.38E-10),
              (template_2D.format('yy'), 1.11E-10),
              (template_2D.format('xy'), 7.7E-11),
              (template_3D.format('xx'), 1.97E-10),
              (template_3D.format('yy'), 8.46E-10),
              (template_3D.format('zz'), 1.81E-9),
              (template_3D.format('yz'), 3.1E-10),
              (template_3D.format('zx'), 6.4E-10),
              (template_3D.format('xy'), 1.81E-9),
               ]
    return [(os.path.join(directory, 'data', filename), rel_err)
            for filename, rel_err in params]

"""
class FilteredGreenOperatorTest(unittest.TestCase,
                                metaclass=DiscreteGreenOperatorTestMetaclass,
                                test_apply_params=filtered_test_apply_params()):

    def discrete_green_operator(self, n, h = 1., greenc=None, transform=None):
        if greenc is None:
            mat = Material(0.75, 0.3, len(n))
            greenc = janus.greenop.create(mat)
        if len(n) == 2:
            return FilteredGreenOperator2D(greenc, n, h, transform)
        elif len(n) == 3:
            return FilteredGreenOperator3D(greenc, n, h, transform)
        else:
            raise ValueError('dimension must be 2 or 3 '
                             '(was {})'.format(len(n)))

    def to_memoryview_expected(self, greenc, n, b):
        dim = len(n)
        n = np.asarray(n)
        g = np.zeros((greenc.osize, greenc.isize),
                     dtype=np.float64)
        for i in itertools.product(*itertools.repeat(range(-1, 1), dim)):
            bb = b + i * n
            k = 2 * np.pi * bb / n
            w = np.square(np.product(np.cos(0.25 * k)))
            greenc.set_frequency(k)
            g += w * np.asarray(greenc.to_memoryview())

        return g
"""

if __name__ == '__main__':
    #unittest.main(verbosity=3)
    unittest.main()
