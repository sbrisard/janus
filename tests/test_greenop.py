import itertools

import numpy as np
import pytest

import janus.material.elastic.linear.isotropic as material
import test_operators

from numpy import cos
from numpy import sin
from numpy.testing import assert_array_almost_equal_nulp

from janus.mandelvoigt import MandelVoigt

def delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def green_coefficient(i, j, k, l, n, mat):
    return ((delta(i, k) * n[j] * n[l] +
             delta(i, l) * n[j] * n[k] +
             delta(j, k) * n[i] * n[l] +
             delta(j, l) * n[i] * n[k]) -
            2 * n[i] * n[j] * n[k] * n[l] / (1. - mat.nu)) / (4 * mat.g)

def green_matrix(k, mat):
    k2 = sum(k**2)
    if k2 == 0.:
        sym = (mat.dim * (mat.dim + 1)) // 2
        return np.zeros((sym, sym), dtype=np.float64)
    else:
        n = k / np.sqrt(k2)
        return MandelVoigt(mat.dim).as_array(green_coefficient, n, mat)


class AbstractTestGreenOperator(test_operators.TestAbstractLinearOperator):
    mu = 0.7
    nu = 0.3

    def valid_size(self):
        """Inherited from TestAbstractLinearOperator"""
        sym = (self.dim * (self.dim + 1)) // 2
        return (sym, sym)

    def material(self):
        return material.create(self.mu, self.nu, self.dim)

    def pytest_generate_tests(self, metafunc):
        if metafunc.function.__name__ == 'test_apply':
            params = itertools.product(self.wave_vectors(),
                                       [0, 1, 2])
            metafunc.parametrize('k, flag', params)
        elif metafunc.function.__name__ == 'test_to_memoryview':
            params = itertools.product(self.wave_vectors(),
                                       [0, 1])
            metafunc.parametrize('k, flag', params)
        else:
            super().pytest_generate_tests(metafunc)

    def test_init_invalid_dimension(self):
        if self.dim == 2:
            mat = material.create(1.0, 0.3, 3)
            cls = material._GreenOperatorForStrains2D
        elif self.dim == 3:
            mat = material.create(1.0, 0.3, 2)
            cls = material._GreenOperatorForStrains3D
        else:
            raise ValueError()
        with pytest.raises(ValueError):
            cls(mat)

    def test_apply(self, k, flag):
        """flag allows the specification of various calling sequences:
        - flag = 0: apply(b, tau)
        - flag = 1: apply(b, tau, tau)
        - flag = 2: apply(b, tau, eta)
        """
        mat = self.material()
        expected = green_matrix(k,  mat)
        green = mat.green_operator()
        tau = np.zeros((green.isize,), np.float64)
        if flag == 0:
            base = None
        elif flag == 1:
            base = tau
        elif flag == 2:
            base = np.empty((green.osize,), np.float64)
        else:
            raise ValueError()
        for j in range(green.isize):
            tau[:] = 0.
            tau[j] = 1.
            green.set_frequency(k)
            actual = green.apply(tau, base)
            if flag != 0:
                assert actual.base is base
            assert_array_almost_equal_nulp(expected[:, j], actual, 325)

    def test_to_memoryview(self, k, flag):
        """flag allows the specification of various calling sequences:
        - flag = 0: to_memoryview()
        - flag = 1: to_memoryview(a)
        """
        mat = self.material()
        expected = green_matrix(k,  mat)
        green = mat.green_operator()
        green.set_frequency(k)
        if flag == 0:
            actual = green.to_memoryview()
        elif flag == 1:
            base = np.empty((green.osize, green.isize), np.float64)
            actual = green.to_memoryview(base)
            assert actual.base is base
        else:
            raise ValueError()
        assert_array_almost_equal_nulp(expected, actual, 325)

    def test_set_frequency_invalid_params(self):
        k = np.zeros((self.dim + 1,), dtype=np.float64)
        green = self.material().green_operator()
        with pytest.raises(ValueError):
            green.set_frequency(k)


class TestGreenOperator2D(AbstractTestGreenOperator):
    dim = 2

    def wave_vectors(self):
        norms = [2.5, 3.5]
        thetas = np.linspace(0., 2. * np.pi,
                             num=20, endpoint=False)
        def vec(r, theta):
            return np.array([r * cos(theta), r * sin(theta)], dtype=np.float64)
        ret = [vec(k, theta) for theta in thetas for k in norms]
        ret.append(np.zeros((self.dim,), dtype=np.float64))
        return ret


class TestGreenOperator3D(AbstractTestGreenOperator):
    dim = 3

    def wave_vectors(self):
        norms = [2.5, 3.5]
        num_thetas = 10
        num_phis = 20
        thetas = [(np.pi * i) / num_thetas for i in range(num_thetas)]
        phis = [(2. * np.pi * i) / num_phis for i in range(num_phis)]
        def vec(r, theta, phi):
            return np.array([r * cos(theta) * cos(phi),
                             r * cos(theta) * sin(phi),
                             r * sin(theta)],
                            dtype=np.float64)
        ret= [vec(k, theta, phi) for phi in phis for theta in thetas
              for k in norms]
        ret.append(np.zeros((self.dim,), dtype=np.float64))
        return ret
