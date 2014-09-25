import itertools
import os.path

import numpy as np
import pytest

import janus.fft.serial as fft
import janus.material.elastic.linear.isotropic as material

from numpy.testing import assert_allclose

from janus.green import truncated
from janus.green import filtered

# Default material constants
MU = 0.75
NU = 0.3

# All tests are performed with discrete Green operators based on these grids
GRID_SIZES = ([(8, 8), (8, 16), (16, 8), (4, 4, 4)]
              + list(itertools.permutations((4, 8, 16))))

# The value of one unit in the last place, for the float64 format.
ULP = np.finfo(np.float64).eps


def multi_indices(n):
    """Return the list of all multi-indices within the specified bounds.

    Return the list of multi-indices ``[b[0], ..., b[dim - 1]]`` such that
    ``0 <= b[i] < n[i]`` for all i.

    """
    iterables = [range(ni) for ni in n]
    return [np.asarray(b, dtype=np.intc)
            for b in itertools.product(*iterables)]


class AbstractTestDiscreteGreenOperator:

    def pytest_generate_tests(self, metafunc):
        if metafunc.function.__name__ == 'test_init_invalid_params':
            g2 = material.create(MU, NU, 2).green_operator()
            g3 = material.create(MU, NU, 3).green_operator()
            params = [(g3, (9, 9), 1., None),
                      (g2, (-1, 9), 1., None),
                      (g2, (9, -1), 1., None),
                      (g2, (9, 9), -1., None),
                      (g2, (9, 9), 1., fft.create_real((8, 9))),
                      (g2, (9, 9), 1., fft.create_real((9, 8))),
                      (g2, (9, 9, 9), 1., None),
                      (g3, (-1, 9, 9), 1., None),
                      (g3, (9, -1, 9), 1., None),
                      (g3, (9, 9, -1), 1., None),
                      (g3, (9, 9, 9), -1., None)]
            metafunc.parametrize('greenc, n, h, transform', params)
        elif metafunc.function.__name__ == 'test_to_memoryview':
            params = [(n, flag) for n in GRID_SIZES for flag in [0, 1]]
            metafunc.parametrize('n, flag', params)
        elif metafunc.function.__name__ == 'test_to_memoryview_invalid_params':
            g2 = self.greend(material.create(MU, NU, 2).green_operator(),
                             (8, 16), 1.0)
            g3 = self.greend(material.create(MU, NU, 3).green_operator(),
                             (8, 16, 32), 1.0)
            params = [(g2, (g2.oshape[-1], g2.ishape[-1] + 1)),
                      (g2, (g2.oshape[-1] + 1, g2.ishape[-1])),
                      (g3, (g3.oshape[-1], g3.ishape[-1] + 1)),
                      (g3, (g3.oshape[-1] + 1, g3.ishape[-1])),]
            metafunc.parametrize('greend, out_shape', params)
        elif metafunc.function.__name__ == 'test_apply_by_freq':
            x = [np.array([0.3, -0.4, 0.5]),
                 np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])]
            params = [(n, x[len(n) - 2], flag) for n in GRID_SIZES
                      for flag in range(3)]
            metafunc.parametrize('n, x, flag', params)
        elif metafunc.function.__name__ == 'test_apply_by_freq_invalid_params':
            g2 = self.greend(material.create(MU, NU, 2).green_operator(),
                             (8, 16), 1.0)
            g3 = self.greend(material.create(MU, NU, 3).green_operator(),
                             (8, 16, 32), 1.0)
            params = [(g2, g2.ishape[-1], g2.oshape[-1] + 1),
                      (g2, g2.ishape[-1] + 1, g2.oshape[-1]),
                      (g3, g3.ishape[-1], g3.oshape[-1] + 1),
                      (g3, g3.ishape[-1] + 1, g3.oshape[-1]),]
            metafunc.parametrize('greend, x_size, y_size', params)
        elif metafunc.function.__name__ == 'test_apply':
            if self.operates_in_place:
                flags = [0, 1, 2]
            else:
                flags = [0, 2]
            params = [i + (j,) for i in self.params_test_apply() for j in flags]
            metafunc.parametrize('path_to_ref, rtol, flag', params)
        elif metafunc.function.__name__ == 'test_apply_invalid_params':
            g = self.greend(material.create(MU, NU, 2).green_operator(),
                            (8, 16), 1.0)
            i0, i1, i2 = g.ishape
            o0, o1, o2 = g.oshape
            params2 = [(g, (i0 + 1, i1, i2), (o0, o1, o2)),
                      (g, (i0, i1 + 1, i2), (o0, o1, o2)),
                      (g, (i0, i1, i2 + 1), (o0, o1, o2)),
                      (g, (i0, i1, i2), (o0 + 1, o1, o2)),
                      (g, (i0, i1, i2), (o0, o1 + 1, o2)),
                      (g, (i0, i1, i2), (o0, o1, o2 + 1))]
            g = self.greend(material.create(MU, NU, 3).green_operator(),
                            (8, 16, 32), 1.0)
            i0, i1, i2, i3 = g.ishape
            o0, o1, o2, o3 = g.oshape
            params3 = [(g, (i0 + 1, i1, i2, i3), (o0, o1, o2, o3)),
                       (g, (i0, i1 + 1, i2, i3), (o0, o1, o2, o3)),
                       (g, (i0, i1, i2 + 1, i3), (o0, o1, o2, o3)),
                       (g, (i0, i1, i2, i3 + 1), (o0, o1, o2, o3)),
                       (g, (i0, i1, i2, i3), (o0 + 1, o1, o2, o3)),
                       (g, (i0, i1, i2, i3), (o0, o1 + 1, o2, o3)),
                       (g, (i0, i1, i2, i3), (o0, o1, o2 + 1, o3)),
                       (g, (i0, i1, i2, i3), (o0, o1, o2, o3 + 1))]
            metafunc.parametrize('greend, x_shape, y_shape', params2 + params3)

    def test_init_invalid_params(self, greenc, n, h, transform):
        with pytest.raises(ValueError):
            self.greend(greenc, n, h, transform)

    def test_to_memoryview(self, n, flag):
        dim = len(n)
        greenc = material.create(MU, NU, dim).green_operator()
        greend = self.greend(greenc, n, 1.0)
        k = np.empty((dim,), dtype=np.float64)
        for b in multi_indices(n):
            expected = self.to_memoryview_expected(greenc, n, b)
            greend.set_frequency(b)
            if flag == 0:
                actual = greend.to_memoryview()
            elif flag == 1:
                base = np.empty_like(expected)
                actual = greend.to_memoryview(base)
                assert actual.base is base
            else:
                raise ValueError
            assert_allclose(actual, expected, ULP, ULP)

    def test_to_memoryview_invalid_params(self, greend, out_shape):
        out = np.zeros(out_shape, dtype=np.float64)
        with pytest.raises(ValueError):
            greend.to_memoryview(out)

    def test_apply_by_freq(self, n, x, flag):
        """flag allows the specification of various calling sequences:
          - flag = 0: apply_by_freq(x)
          - flag = 1: apply_by_freq(x, x)
          - flag = 2: apply_by_freq(x, y)
        """
        dim = len(n)
        green = self.greend(material.create(MU, NU, dim).green_operator(),
                            n, 1.)
        for b in multi_indices(n):
            green.set_frequency(b)
            g = np.asarray(green.to_memoryview())
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
                #assert get_base(actual) is base
                assert actual.base is base
            assert_allclose(actual, expected, ULP, ULP)

    def test_apply_by_freq_invalid_params(self, greend, x_size, y_size):
        x = np.zeros((x_size,), dtype=np.float64)
        y = np.zeros((y_size,), dtype=np.float64)
        with pytest.raises(ValueError):
            greend.apply_by_freq(x, y)

    def test_apply(self, path_to_ref, rtol, flag):
        npz_file = np.load(path_to_ref)
        x = npz_file['x']
        expected = npz_file['y']
        # This is a workaround to avoid
        # ResourceWarning: unclosed file <_io.BufferedReader name='xxx.npz'>
        #     self.zip.close()
        npz_file.zip.fp.close()
        n = x.shape[:-1]
        transform = fft.create_real(n)
        green = self.greend(material.create(MU, NU, len(n)).green_operator(),
                            n, 1., transform=transform)
        if flag == 0:
            actual = green.apply(x)
        elif flag == 1:
            actual = green.apply(x, x)
        elif flag == 2:
            base = np.zeros(transform.rshape + (green.oshape[-1],), np.float64)
            actual = green.apply(x, base)
        else:
            raise ValueError()
        assert_allclose(actual, expected, rtol, 10 * ULP)

    def test_apply_invalid_params(self, greend, x_shape, y_shape):
        x = np.zeros(x_shape, dtype=np.float64)
        y = np.zeros(y_shape, dtype=np.float64)
        with pytest.raises(ValueError):
            greend.apply(x, y)


class TestTruncatedGreenOperator(AbstractTestDiscreteGreenOperator):
    operates_in_place = True

    def greend(self, greenc, n, h, transform=None):
        return truncated(greenc, n, h, transform)

    def to_memoryview_expected(self, greenc, n, b):
        n = np.asarray(n)
        k = np.empty((len(n),), dtype=np.float64)
        i = 2 * b > n
        k[i] = 2. * np.pi * (b[i] - n[i]) / n[i]
        k[~i] = 2. * np.pi * b[~i] / n[~i]

        greenc.set_frequency(k)
        return greenc.to_memoryview()

    def params_test_apply(self):
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
                  (template_3D.format('xy'), ULP)]
        return [(os.path.join(directory, 'data', filename), rtol)
                for filename, rtol in params]


class TestFilteredGreenOperator(AbstractTestDiscreteGreenOperator):
    operates_in_place = True

    def greend(self, greenc, n, h, transform=None):
        return filtered(greenc, n, h, transform)

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

    def params_test_apply(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        template_2D = ('filtered_green_operator_200x300_'
                       'unit_tau_{0}_10x10+95+145.npz')
        template_3D = ('filtered_green_operator_40x50x60_'
                       'unit_tau_{0}_10x10x10+15+20+25.npz')
        params = [(template_2D.format('xx'), ULP),
                  (template_2D.format('yy'), ULP),
                  (template_2D.format('xy'), ULP),
                  (template_3D.format('xx'), ULP),
                  (template_3D.format('yy'), ULP),
                  (template_3D.format('zz'), ULP),
                  (template_3D.format('yz'), ULP),
                  (template_3D.format('zx'), ULP),
                  (template_3D.format('xy'), ULP)]
        return [(os.path.join(directory, 'data', filename), rtol)
                for filename, rtol in params]
