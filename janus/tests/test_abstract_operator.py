import itertools
import inspect

import numpy as np
import pytest

from numpy.testing import assert_allclose

from janus.operators import AbstractOperator
from janus.operators import AbstractLinearOperator
from janus.operators import isotropic_4

ULP = np.finfo(np.float64).eps


class TestAbstractOperator:

    """Test of the `AbstractOperator` class.

    This test case mainly tests that `ValueError`is raised when
    `init_sizes()` and `apply()` are called with invalid arguments.
    This test case can be extended to test concrete implementations of
    this class. In this case, the `setUp()` method must define a
    properly initialized `operator` attribute, as well as the expected
    `isize` and `osize`.

    """

    def valid_size(self):
        return (4, 5)

    def operator(self):
        op = AbstractOperator()
        op.init_sizes(*self.valid_size())
        return op

    def pytest_generate_tests(self, metafunc):
        op = self.operator()
        args = None
        params = None
        if metafunc.function.__name__ == 'test_init_sizes':
            isize, osize = self.valid_size()
            params = [(op, isize, osize)]
        if metafunc.function.__name__ == 'test_init_sizes_invalid_params':
            sizes = [(0, 1), (-1, 1), (1, 0), (1, -1)]
            params = [(op, size[0], size[1]) for size in sizes]
        if metafunc.function.__name__ == 'test_apply_invalid_params':
            params = [(op, op.isize + 1, 0),
                      (op, op.isize + 1, op.osize),
                      (op, op.isize, op.osize + 1)]
        if metafunc.function.__name__ == 'test_apply_specified_output':
            params = [(op,)]
        if params is not None:
            if args is None:
                args = inspect.getargspec(metafunc.function)[0][1:]
            metafunc.parametrize(args, params)

    def test_init_sizes(self, operator, isize, osize):
        assert (operator.isize, operator.osize) == (isize, osize)

    def test_init_sizes_invalid_params(self, operator, isize, osize):
        with pytest.raises(ValueError):
            operator.init_sizes(isize, osize)

    def test_apply_invalid_params(self, operator, isize, osize):
        with pytest.raises(ValueError):
            x = np.zeros((isize,), dtype=np.float64)
            y = np.zeros((osize,), dtype=np.float64) if osize != 0 else None
            operator.apply(x, y)

    def test_apply_specified_output(self, operator):
        x = np.zeros((operator.isize,), dtype=np.float64)
        base = np.zeros((operator.osize,), dtype=np.float64)
        y = operator.apply(x, base)
        assert y.base is base


class TestAbstractLinearOperator(TestAbstractOperator):
    def operator(self):
        op = AbstractLinearOperator()
        op.init_sizes(*self.valid_size())
        return op

    def pytest_generate_tests(self, metafunc):
        super().pytest_generate_tests(metafunc)
        if metafunc.function.__name__ == 'test_to_memoryview_invalid_params':
            op = self.operator()
            params = [(op, op.isize + 1, op.osize),
                      (op, op.isize, op.osize + 1)]
            metafunc.parametrize('operator, isize, osize', params)

    def test_to_memoryview_invalid_params(self, operator, isize, osize):
        with pytest.raises(ValueError):
            operator.to_memoryview(np.empty((osize, isize), dtype=np.float64))


# class FourthRankIsotropicTensorTest(AbstractLinearOperatorTest):
#     def setUp(self):
#         if hasattr(self, 'dim'):
#             self.sym = (self.dim * (self.dim + 1)) // 2
#             self.isize = self.sym
#             self.osize = self.sym
#             self.operator = isotropic_4(1., 1., self.dim)
#         else:
#             raise unittest.SkipTest("abstract test case")

#     def to_array(self, sph, dev):
#         a = np.zeros((self.sym, self.sym), dtype=np.float64)
#         a[0:self.dim, 0:self.dim] = (sph - dev) / self.dim
#         i = range(self.dim)
#         a[i, i] = (sph + (self.dim - 1) * dev) / self.dim
#         i = range(self.dim, self.sym)
#         a[i, i] = dev
#         return a

#     def ptest_apply(self, sph, dev, flag):
#         """flag allows the specification of various calling sequences:
#           - flag = 0: apply(x)
#           - flag = 1: apply(x, x)
#           - flag = 2: apply(x, y)
#         """
#         t = isotropic_4(sph, dev, self.dim)

#         eye = np.eye(self.sym)
#         expected = self.to_array(sph, dev)
#         actual = np.empty_like(eye)

#         for i in range(self.sym):
#             x = eye[:, i]
#             if flag == 0:
#                 base = None
#             elif flag == 1:
#                 base = x
#             elif flag == 2:
#                 base = np.empty_like(x)
#             ret = t.apply(eye[:, i], base)
#             if flag != 0:
#                 assert ret.base is base
#             actual[:, i] = ret

#         assert_allclose(expected, actual, ULP, ULP)

#     def test_apply_1(self):
#         self.ptest_apply(1., 0., 0)

#     def test_apply_2(self):
#         self.ptest_apply(0., 1., 0)

#     def test_apply_3(self):
#         self.ptest_apply(2.5, -3.5, 0)

#     def test_apply_in_place_1(self):
#         self.ptest_apply(1., 0., 1)

#     def test_apply_in_place_2(self):
#         self.ptest_apply(0., 1., 1)

#     def test_apply_in_place_3(self):
#         self.ptest_apply(2.5, -3.5, 1)

#     def test_apply_specified_output_3(self):
#         self.ptest_apply(2.5, -3.5, 2)

#     def test_apply_specified_output_1(self):
#         self.ptest_apply(1., 0., 2)

#     def test_apply_specified_output_2(self):
#         self.ptest_apply(0., 1., 2)


# class FourthRankIsotropicTensor2DTest(FourthRankIsotropicTensorTest):
#     dim = 2


# class FourthRankIsotropicTensor3DTest(FourthRankIsotropicTensorTest):
#     dim = 3

# def suite():
#     suite = unittest.TestSuite()
#     suite.addTest(AbstractOperatorTest())
#     suite.addTest(AbstractLinearOperatorTest())
#     return suite

# if __name__ == '__main__':
#     unittest.main(verbosity=1)
