import itertools
import inspect

import numpy as np
import numpy.random as nprnd
import pytest

from numpy.testing import assert_allclose

from janus.operators import AbstractOperator
from janus.operators import AbstractLinearOperator
from janus.operators import AbstractStructuredOperator2D
from janus.operators import AbstractStructuredOperator3D
from janus.operators import block_diagonal_operator
from janus.operators import block_diagonal_linear_operator
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
        if metafunc.function.__name__ == 'test_to_memoryview_invalid_params':
            op = self.operator()
            params = [(op, op.isize + 1, op.osize),
                      (op, op.isize, op.osize + 1)]
            metafunc.parametrize('operator, isize, osize', params)
        else:
            super().pytest_generate_tests(metafunc)

    def test_to_memoryview_invalid_params(self, operator, isize, osize):
        with pytest.raises(ValueError):
            operator.to_memoryview(np.empty((osize, isize), dtype=np.float64))


class AbstractTestFourthRankIsotropicTensor(TestAbstractLinearOperator):
    def sym(self):
        return (self.dim * (self.dim + 1)) // 2

    def operator(self):
        return isotropic_4(1., 1., self.dim)

    def valid_size(self):
        sym = self.sym()
        return (sym, sym)

    def pytest_generate_tests(self, metafunc):
        if metafunc.function.__name__ == 'test_apply':
            flags = [0, 1, 2]
            params = [(1., 0.), (0., 1.), (2.5, -3.5)]
            params = [(sph, dev, flag) for ((sph, dev), flag)
                      in itertools.product(params, flags)]
            metafunc.parametrize('sph, dev, flag', params)
        elif metafunc.function.__name__ == 'test_to_memoryview':
            flags = [0, 1]
            params = [(1., 0.), (0., 1.), (2.5, -3.5)]
            params = [(sph, dev, flag) for ((sph, dev), flag)
                      in itertools.product(params, flags)]
            metafunc.parametrize('sph, dev, flag', params)
        else:
            super().pytest_generate_tests(metafunc)

    def to_array(self, sph, dev):
        sym = self.sym()
        a = np.zeros((sym, sym), dtype=np.float64)
        a[0:self.dim, 0:self.dim] = (sph - dev) / self.dim
        i = range(self.dim)
        a[i, i] = (sph + (self.dim - 1) * dev) / self.dim
        i = range(self.dim, sym)
        a[i, i] = dev
        return a

    def test_apply(self, sph, dev, flag):
        """flag allows the specification of various calling sequences:
          - flag = 0: apply(x)
          - flag = 1: apply(x, x)
          - flag = 2: apply(x, y)
        """
        t = isotropic_4(sph, dev, self.dim)

        eye = np.eye(self.sym())
        expected = self.to_array(sph, dev)
        actual = np.empty_like(eye)

        for i in range(self.sym()):
            x = eye[:, i]
            if flag == 0:
                base = None
            elif flag == 1:
                base = x
            elif flag == 2:
                base = nprnd.rand(*x.shape)
            ret = t.apply(eye[:, i], base)
            if flag != 0:
                assert ret.base is base
            actual[:, i] = ret

        assert_allclose(expected, actual, 1 * ULP, 0 * ULP)

    def test_to_memoryview(self, sph, dev, flag):
        t = isotropic_4(sph, dev, self.dim)
        expected = self.to_array(sph, dev)
        if flag == 0:
            base = None
        elif flag == 1:
            base = nprnd.rand(t.osize, t.isize)
        actual = t.to_memoryview(base)
        if flag != 0:
            assert actual.base is base
        assert_allclose(expected, actual, 0 * ULP, 0 * ULP)


class TestFourthRankIsotropicTensor2D(AbstractTestFourthRankIsotropicTensor):
    dim = 2


class TestFourthRankIsotropicTensor3D(AbstractTestFourthRankIsotropicTensor):
    dim = 3


class AbstractTestAbstractStructuredOperator:
    def pytest_generate_tests(self, metafunc):
        args = inspect.getargspec(metafunc.function)[0][1:]
        if metafunc.function.__name__ == 'test_init_shapes_invalid_params':
            ones = tuple(itertools.repeat(1, self.dim + 1))
            params = set(itertools.chain(itertools.permutations((0,) + ones),
                                         itertools.permutations((-1,) + ones)))
            params = [(param,) for param in params]
            metafunc.parametrize(args, params)
        if metafunc.function.__name__ == 'test_apply_invalid_params':
            def increment(x, i):
                return tuple(xj + 1 if j == i else xj
                             for j, xj in enumerate(x))
            op = self.operator()
            ishapes = [increment(op.ishape, i) for i in range(len(op.ishape))]
            oshapes = [increment(op.oshape, i) for i in range(len(op.oshape))]
            params = ([(op, ishape, None) for ishape in ishapes] +
                      [(op, ishape, op.oshape) for ishape in ishapes] +
                      [(op, op.ishape, oshape) for oshape in oshapes])
            metafunc.parametrize(args, params)

    def test_shapes(self):
        assert self.dim == 2 or self.dim == 3
        aux = self.valid_shape()
        ishape = aux[:-1]
        oshape = aux[:-2] + (aux[-1],)
        op = self.operator()
        if self.dim == 2:
            assert (op.shape0, op.shape1, op.ishape2) == ishape
            assert (op.shape0, op.shape1, op.oshape2) == oshape
        elif self.dim == 3:
            assert (op.shape0, op.shape1, op.shape2, op.ishape3) == ishape
            assert (op.shape0, op.shape1, op.shape2, op.oshape3) == oshape
        assert op.ishape == ishape
        assert op.oshape == oshape

    def test_init_shapes_invalid_params(self, shapes):
        op = self.operator()
        with pytest.raises(ValueError):
            op.init_shapes(*shapes)

    def test_apply_invalid_params(self, operator, ishape, oshape):
        with pytest.raises(ValueError):
            x = np.zeros(ishape, dtype=np.float64)
            y = (np.zeros(oshape, dtype=np.float64) if oshape is not None
                 else None)
            operator.apply(x, y)

    def test_apply_specified_output(self):
        op = self.operator()
        x = np.zeros(op.ishape, dtype=np.float64)
        base = np.zeros(op.oshape, dtype=np.float64)
        y = op.apply(x, base)
        assert y.base is base

class TestAbstractStructuredOperator2D(AbstractTestAbstractStructuredOperator):
    dim = 2

    def valid_shape(self):
        return (6, 5, 4, 3)

    def operator(self):
        op = AbstractStructuredOperator2D()
        op.init_shapes(*self.valid_shape())
        return op

class TestAbstractStructuredOperator3D(AbstractTestAbstractStructuredOperator):
    dim = 3

    def valid_shape(self):
        return (7, 6, 5, 4, 3)

    def operator(self):
        op = AbstractStructuredOperator3D()
        op.init_shapes(*self.valid_shape())
        return op

class AbstractTestBlockDiagonalOperator(AbstractTestAbstractStructuredOperator):

    def valid_shape(self):
        sym = (self.dim * (self.dim + 1)) // 2
        shape = list(range(sym + 1, sym + 1 + self.dim))
        shape.reverse()
        return (tuple(shape) + (sym, sym))

    def local_operators(self):
        global_shape = self.valid_shape()[0:self.dim]
        loc = np.empty(global_shape, dtype=object)
        for index in itertools.product(*map(range, global_shape)):
            loc[index] = isotropic_4(2. * nprnd.rand() - 1,
                                     2. * nprnd.rand() - 1,
                                     self.dim)
        return loc

    def operator(self):
        return block_diagonal_operator(self.local_operators())

    def pytest_generate_tests(self, metafunc):
        args = inspect.getargspec(metafunc.function)[0][1:]
        if metafunc.function.__name__ == 'test_apply':
            global_shape = self.valid_shape()[0:self.dim]
            loc = self.local_operators()
            operator = block_diagonal_operator(loc)
            params = []
            for i in range(10):
                x = nprnd.rand(*operator.ishape)
                y_expected = np.empty(operator.oshape, dtype=np.float64)
                for index in itertools.product(*map(range,
                                                    global_shape)):
                    loc[index].apply(x[index], y_expected[index])
                params.append((operator, x, y_expected))
            metafunc.parametrize(args, params)
        else:
            super().pytest_generate_tests(metafunc)

    def test_apply(self, operator, x, y_expected):
        y_actual = np.empty(operator.oshape, dtype=np.float64)
        operator.apply(x, y_actual)
        assert_allclose(y_expected, y_actual, 0 * ULP, 0 * ULP)


class TestBlockDiagonalOperator2D(AbstractTestBlockDiagonalOperator):
    dim = 2


class TestBlockDiagonalOperator3D(AbstractTestBlockDiagonalOperator):
    dim = 3


class AbstractTestBlockDiagonalLinearOperator(AbstractTestAbstractStructuredOperator):

    def valid_shape(self):
        sym = (self.dim * (self.dim + 1)) // 2
        return tuple(range(self.dim + 4, 2, -1))

    def local_matrices(self):
        shape = self.valid_shape()
        shape = shape[:-2] + (shape[-1], shape[-2])
        return 2. * nprnd.rand(*shape) - 1.

    def operator(self):
        return block_diagonal_linear_operator(self.local_matrices())

    def pytest_generate_tests(self, metafunc):
        args = inspect.getargspec(metafunc.function)[0][1:]
        if metafunc.function.__name__ == 'test_apply':
            global_shape = self.valid_shape()[0:self.dim]
            a = self.local_matrices()
            operator = block_diagonal_linear_operator(a)
            params = []
            for i in range(10):
                x = nprnd.rand(*operator.ishape)
                y_expected = np.empty(operator.oshape, dtype=np.float64)
                for index in itertools.product(*map(range,
                                                    global_shape)):
                    y_expected[index] = np.dot(a[index], x[index])
                params.append((operator, x, y_expected))
            metafunc.parametrize(args, params)
        elif metafunc.function.__name__ == 'test_apply_transpose':
            operator = self.operator()
            params = []
            for i in range(10):
                x = nprnd.rand(*operator.ishape)
                y = nprnd.rand(*operator.oshape)
                params.append((operator, x, y))
            metafunc.parametrize(args, params)
        else:
            super().pytest_generate_tests(metafunc)

    def test_apply(self, operator, x, y_expected):
        y_actual = np.empty(operator.oshape, dtype=np.float64)
        operator.apply(x, y_actual)
        assert_allclose(y_expected, y_actual, 0 * ULP, 0 * ULP)

    def test_apply_transpose(self, operator, x, y):
        yax = np.sum(y * operator.apply(x))
        atyx = np.sum(operator.apply_transpose(y) * x)
        assert np.abs(yax - atyx) <= 100 * ULP * np.maximum(np.abs(yax), 1.0)


class TestBlockDiagonalLinearOperator2D(AbstractTestBlockDiagonalLinearOperator):
    dim = 2


class TestBlockDiagonalLinearOperator3D(AbstractTestBlockDiagonalLinearOperator):
    dim = 3
