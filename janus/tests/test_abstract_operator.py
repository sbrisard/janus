import array
import itertools
import unittest

import numpy as np

from janus.operators import AbstractOperator
from janus.operators import AbstractLinearOperator
from janus.operators import isotropic_4

class AbstractOperatorTest(unittest.TestCase):

    """Test of the `AbstractOperator` class.

    This test case mainly tests that `ValueError`is raised when
    `init_sizes()` and `apply()` are called with invalid arguments.
    This test case can be extended to test concrete implementations of
    this class. In this case, the `setUp()` method must define a
    properly initialized `operator` attribute, as well as the expected
    `isize` and `osize`.

    """

    def setUp(self):
        self.operator = AbstractOperator()
        self.isize = 4
        self.osize = 5
        self.operator.init_sizes(self.isize, self.osize)

    def test_init_sizes(self):
        self.assertEqual(self.isize, self.operator.isize, 'isize')
        self.assertEqual(self.osize, self.operator.osize, 'osize')

    def test_init_size_null_isize(self):
        self.assertRaises(ValueError, self.operator.init_sizes, 0, 1)

    def test_init_size_negative_isize(self):
        self.assertRaises(ValueError, self.operator.init_sizes, -1, 1)

    def test_init_size_null_osize(self):
        self.assertRaises(ValueError, self.operator.init_sizes, 1, 0)

    def test_init_size_negative_osize(self):
        self.assertRaises(ValueError, self.operator.init_sizes, 1, -1)

    def test_apply_invalid_input(self):
        x = array.array('d', itertools.repeat(0., self.operator.isize + 1))
        self.assertRaises(ValueError, self.operator.apply, x)

    def test_apply_invalid_output(self):
        x = array.array('d', itertools.repeat(0., self.operator.isize))
        y = array.array('d', itertools.repeat(0., self.operator.osize + 1))
        self.assertRaises(ValueError, self.operator.apply, x, y)

    def test_apply_specified_output(self):
        x = array.array('d', itertools.repeat(0., self.operator.isize))
        y = array.array('d', itertools.repeat(0., self.operator.osize))
        yy = self.operator.apply(x, y)
        self.assertIs(y, yy.base)

class AbstractLinearOperatorTest(AbstractOperatorTest):
    def setUp(self):
        self.isize = 4
        self.osize = 5
        self.operator = AbstractLinearOperator()
        self.operator.init_sizes(self.isize, self.osize)

    def test_to_memory_view_invalid_output_1(self):
        """Check that `to_memory_view` raises `ValueError` when called
        with invalid argument.

        In this test, the output array has invalid number of rows.

        """
        out = np.empty((self.isize + 1, self.osize), dtype=np.float64)
        self.assertRaises(ValueError, self.operator.to_memoryview, out)

    def test_to_memory_view_invalid_output_2(self):
        """Check that `to_memory_view` raises `ValueError` when called
        with invalid argument.

        In this test, the output array has invalid number of columns.

        """
        out = np.empty((self.isize, self.osize + 1), dtype=np.float64)
        self.assertRaises(ValueError, self.operator.to_memoryview, out)


class FourthRankIsotropicTensorTest(AbstractLinearOperatorTest):
    def setUp(self):
        if hasattr(self, 'dim'):
            self.isize = (self.dim * (self.dim + 1)) // 2
            self.osize = self.isize
            self.operator = isotropic_4(1., 1., self.dim)
        else:
            raise unittest.SkipTest("abstract test case")


class FourthRankIsotropicTensor2DTest(FourthRankIsotropicTensorTest):
    dim = 2


class FourthRankIsotropicTensor3DTest(FourthRankIsotropicTensorTest):
    dim = 2

def suite():
    suite = unittest.TestSuite()
    suite.addTest(AbstractOperatorTest())
    suite.addTest(AbstractLinearOperatorTest())
    return suite

if __name__ == '__main__':
    unittest.main(verbosity=1)
