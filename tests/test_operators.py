import itertools
import inspect

import numpy as np
import pytest

from numpy.testing import assert_allclose

from janus.operators import AbstractOperator
from janus.operators import AbstractLinearOperator
from janus.operators import AbstractStructuredOperator2D
from janus.operators import AbstractStructuredOperator3D
from janus.operators import block_diagonal_operator
from janus.operators import block_diagonal_linear_operator
from janus.operators import FourthRankCubicTensor2D
from janus.operators import LinearOperator
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


class TestLinearOperator(TestAbstractLinearOperator):
    def pytest_generate_tests(self, metafunc):
        np.random.seed(20151112)
        if metafunc.function.__name__ == 'test_apply':
            a = np.random.rand(2, 3)
            x = np.random.rand(a.shape[1])
            params = [(a, x, 0), (a, x, 1)]
            metafunc.parametrize('a, x, flag', params)
        elif metafunc.function.__name__ == 'test_apply_transpose':
            a = np.random.rand(2, 3)
            x = np.random.rand(a.shape[0])
            params = [(a, x, 0), (a, x, 1)]
            metafunc.parametrize('a, x, flag', params)
        elif metafunc.function.__name__ == 'test_to_memoryview':
            a = np.random.rand(2, 3)
            metafunc.parametrize('expected, flag', [(a, 0), (a, 1)])
        else:
            super().pytest_generate_tests(metafunc)

    def test_apply(self, a, x, flag):
        expected = np.dot(a, x)
        op = LinearOperator(a)
        if flag == 0:
            y = None
        elif flag == 1:
            y = np.random.rand(op.osize)
        actual = op.apply(x)
        assert_allclose(expected, actual, 1*ULP, 0*ULP)

    def test_apply_transpose(self, a, x, flag):
        expected = np.dot(x, a)
        op = LinearOperator(a)
        if flag == 0:
            y = None
        elif flag == 1:
            y = np.random.rand(op.isize)
        actual = LinearOperator(a).apply_transpose(x)
        assert_allclose(expected, actual, 1*ULP, 0*ULP)

    def test_to_memoryview(self, expected, flag):
        op = LinearOperator(expected)
        if flag == 0:
            out = None
        elif flag == 1:
            out = np.random.rand(op.osize, op.isize)
        actual = op.to_memoryview(out)
        assert_allclose(expected, actual, 1*ULP, 0*ULP)


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
                base = np.random.rand(*x.shape)
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
            base = np.random.rand(t.osize, t.isize)
        actual = t.to_memoryview(base)
        if flag != 0:
            assert actual.base is base
        assert_allclose(expected, actual, 0 * ULP, 0 * ULP)


class TestFourthRankIsotropicTensor2D(AbstractTestFourthRankIsotropicTensor):
    dim = 2


class TestFourthRankIsotropicTensor3D(AbstractTestFourthRankIsotropicTensor):
    dim = 3


class AbstractTestFourthRankCubicTensor(TestAbstractLinearOperator):
    def sym(self):
        return (self.dim * (self.dim + 1)) // 2

    def operator(self):
        return FourthRankCubicTensor2D(0, 0, 0, 0)

    def valid_size(self):
        sym = self.sym()
        return (sym, sym)

    def pytest_generate_tests(self, metafunc):
        if (metafunc.function.__name__ == 'test_apply' or
            metafunc.function.__name__ == 'test_to_memoryview'):
            flags = [0, 1]
            if metafunc.function.__name__ == 'test_apply':
                flags.append(2)
            coeffs = [(1., 0., 0.),
                      (0., 1., 0.),
                      (0., 0., 1.),
                      (2.5, -3.5, 4.5)]
            angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
            params = [(t1111, t1122, t1212, angle, flag) for
                      ((t1111, t1122, t1212), angle, flag) in
                      itertools.product(coeffs, angles, flags)]
            metafunc.parametrize('t1111, t1122, t1212, theta, flag', params)
        else:
            super().pytest_generate_tests(metafunc)

    def to_array(self, t1111, t1122, t1212, theta):
        sqrt2 = np.sqrt(2)
        c = np.cos(theta)
        s = np.sin(theta)
        e1xe1 = np.array([c**2, s**2, sqrt2*c*s],
                         dtype=np.float64).reshape((3, 1))
        e2xe2 = np.array([s**2, c**2, -sqrt2*c*s],
                         dtype=np.float64).reshape(3, 1)
        e1xe2 = np.array([-c*s, c*s, 0.5*sqrt2*(c**2-s**2)],
                         dtype=np.float64).reshape(3, 1)
        t = (t1111*(e1xe1*e1xe1.T+e2xe2*e2xe2.T)+
             t1122*(e1xe1*e2xe2.T+e2xe2*e1xe1.T)+
             4*t1212*(e1xe2*e1xe2.T))
        return t

    def test_apply(self, t1111, t1122, t1212, theta, flag):
        """flag allows the specification of various calling sequences:
          - flag = 0: apply(x)
          - flag = 1: apply(x, x)
          - flag = 2: apply(x, y)
        """
        t = FourthRankCubicTensor2D(t1111, t1122, t1212, theta)

        eye = np.eye(self.sym())
        expected = self.to_array(t1111, t1122, t1212, theta)
        actual = np.empty_like(eye)

        for i in range(self.sym()):
            x = eye[:, i]
            if flag == 0:
                base = None
            elif flag == 1:
                base = x
            elif flag == 2:
                base = np.random.rand(*x.shape)
            ret = t.apply(eye[:, i], base)
            if flag != 0:
                assert ret.base is base
            actual[:, i] = ret

        assert_allclose(expected, actual, 10*ULP, 10*ULP)

    def test_to_memoryview(self, t1111, t1122, t1212, theta, flag):
        t = FourthRankCubicTensor2D(t1111, t1122, t1212, theta)
        expected = self.to_array(t1111, t1122, t1212, theta)
        if flag == 0:
            base = None
        elif flag == 1:
            base = np.random.rand(t.osize, t.isize)
        actual = t.to_memoryview(base)
        if flag != 0:
            assert actual.base is base
        assert_allclose(expected, actual, 10*ULP, 10*ULP)


class TestFourthRankCubicTensor2D(AbstractTestFourthRankCubicTensor):
    dim = 2


@pytest.mark.skipif(True,
                    reason="3D, fourth-rank cubic tensors not yet implemented")
class TestFourthRankCubicTensor3D(AbstractTestFourthRankCubicTensor):
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
            loc[index] = isotropic_4(2. * np.random.rand() - 1,
                                     2. * np.random.rand() - 1,
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
                x = np.random.rand(*operator.ishape)
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
        return 2. * np.random.rand(*shape) - 1.

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
                x = np.random.rand(*operator.ishape)
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
                x = np.random.rand(*operator.ishape)
                y = np.random.rand(*operator.oshape)
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
