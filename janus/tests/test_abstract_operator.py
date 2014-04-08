import itertools

import numpy as np
import pytest

from numpy.testing import assert_allclose

from janus.operators import AbstractOperator
from janus.operators import AbstractLinearOperator
from janus.operators import isotropic_4

ULP = np.finfo(np.float64).eps

def pytest_generate_tests(metafunc):
    if metafunc.function == TestAbstractOperator.test_apply_invalid_params:
        isize, osize = TestAbstractOperator.valid_size()
        isize = 4
        osize = 5
        metafunc.parametrize('invalid_size', [(isize + 1, 0),
                                              (isize + 1, osize),
                                              (isize, osize + 1)])

class TestAbstractOperator:

    """Test of the `AbstractOperator` class.

    This test case mainly tests that `ValueError`is raised when
    `init_sizes()` and `apply()` are called with invalid arguments.
    This test case can be extended to test concrete implementations of
    this class. In this case, the `setUp()` method must define a
    properly initialized `operator` attribute, as well as the expected
    `isize` and `osize`.

    """

    @staticmethod
    @pytest.fixture()
    def valid_size():
        return (4, 5)

    @pytest.fixture()
    def operator(self, valid_size):
        op = AbstractOperator()
        op.init_sizes(*valid_size)
        return op

    def test_init_sizes(self, operator, valid_size):
        assert valid_size == (operator.isize, operator.osize)

    @pytest.mark.parametrize('isize, osize',
                             [(0, 1), (-1, 1), (1, 0), (1, -1)])
    def test_init_sizes_invalid_params(self, operator, isize, osize):
        with pytest.raises(ValueError):
            operator.init_sizes(isize, osize)

    def test_apply_invalid_params(self, operator, invalid_size):
        with pytest.raises(ValueError):
            m, n = invalid_size
            x = np.zeros((m,), dtype=np.float64)
            n = invalid_size[1]
            y = np.zeros((n,), dtype=np.float64) if n != 0 else None
            operator.apply(x, y)

    # def test_apply_invalid_input(self, operator):
    #     x = np.zeros((operator.isize + 1,), dtype=np.float64)
    #     with pytest.raises(ValueError):
    #         operator.apply(x)

    # def test_apply_invalid_output(self, operator):
    #     x = np.zeros((operator.isize,), dtype=np.float64)
    #     y = np.zeros((operator.osize + 1,), dtype=np.float64)
    #     with pytest.raises(ValueError):
    #         operator.apply(x, y)

    def test_apply_specified_output(self, operator):
        x = np.zeros((operator.isize,), dtype=np.float64)
        base = np.zeros((operator.osize,), dtype=np.float64)
        y = operator.apply(x, base)
        assert y.base is base


# class AbstractLinearOperatorTest(AbstractOperatorTest):
#     def setUp(self):
#         self.isize = 4
#         self.osize = 5
#         self.operator = AbstractLinearOperator()
#         self.operator.init_sizes(self.isize, self.osize)

#     def test_to_memory_view_invalid_output_1(self):
#         """Check that `to_memory_view` raises `ValueError` when called
#         with invalid argument.

#         In this test, the output array has invalid number of rows.

#         """
#         out = np.empty((self.isize + 1, self.osize), dtype=np.float64)
#         self.assertRaises(ValueError, self.operator.to_memoryview, out)

#     def test_to_memory_view_invalid_output_2(self):
#         """Check that `to_memory_view` raises `ValueError` when called
#         with invalid argument.

#         In this test, the output array has invalid number of columns.

#         """
#         out = np.empty((self.isize, self.osize + 1), dtype=np.float64)
#         self.assertRaises(ValueError, self.operator.to_memoryview, out)


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
