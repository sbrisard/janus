import array
import itertools
import os.path # TODO remove
import sys     # TODO remove
import unittest


# TODO This ugly hack should be removed
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..',
                             '..'))

from janus.operators import AbstractOperator

class AbstractOperatorTest(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()
