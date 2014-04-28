import itertools

import numpy as np
import pytest

from numpy.testing import assert_array_equal

from janus.mandelvoigt import MandelVoigt


class TestMandelVoigt:

    def pytest_generate_tests(self, metafunc):
        if metafunc.function.__name__ == 'test_create_array':
            params = []
            for dim in [2, 3]:
                r = range(dim)
                multi_indices = itertools.product(r, r, r, r)
                params += [(i, j, k, l, dim) for i, j, k, l in multi_indices]
            metafunc.parametrize('m, n, p, q, dim', params)

    @pytest.mark.parametrize('dim', [2, 3])
    def test_singleton(self, dim):
        assert MandelVoigt(dim) is MandelVoigt(dim)

    @pytest.mark.parametrize('dim', [(0,), (4,)])
    def test_init_invalid_params(self, dim):
        with pytest.raises(ValueError):
            MandelVoigt(dim)

    @pytest.mark.parametrize('dim, index, expected',
                             [(2, 0, (0, 0)),
                              (2, 1, (1, 1)),
                              (2, 2, (0, 1)),
                              (3, 0, (0, 0)),
                              (3, 1, (1, 1)),
                              (3, 2, (2, 2)),
                              (3, 3, (1, 2)),
                              (3, 4, (2, 0)),
                              (3, 5, (0, 1))])
    def test_unravel_index(self, dim, index, expected):
        assert MandelVoigt(dim).unravel_index(index) == expected

    @pytest.mark.parametrize('dim, index', [(2, -1), (2, 3), (3, -1), (3, 6)])
    def test_unravel_index_invalid_params(self, dim, index):
        with pytest.raises(ValueError):
            MandelVoigt(dim).unravel_index(index)


    @pytest.mark.parametrize('dim, multi_index, expected',
                             [(2, (0, 0), 0),
                              (2, (1, 1), 1),
                              (2, (0, 1), 2),
                              (2, (1, 0), 2),
                              (3, (0, 0), 0),
                              (3, (1, 1), 1),
                              (3, (2, 2), 2),
                              (3, (1, 2), 3),
                              (3, (2, 1), 3),
                              (3, (2, 0), 4),
                              (3, (0, 2), 4),
                              (3, (0, 1), 5),
                              (3, (1, 0), 5)])
    def test_ravel_multi_index(self, dim, multi_index, expected):
        assert MandelVoigt(dim).ravel_multi_index(*multi_index) == expected

    def test_create_array(self, m, n, p, q, dim):
        mv = MandelVoigt(dim)
        mn = mv.ravel_multi_index(m, n)
        pq = mv.ravel_multi_index(p, q)
        sym = (dim * (dim + 1)) // 2
        expected = np.zeros((sym, sym), dtype=np.float64)
        if mn < dim:
            if pq < dim:
                expected[mn, pq] = 1.0
            else:
                expected[mn, pq] = np.sqrt(2.0)
        else:
            if pq < dim:
                expected[mn, pq] = np.sqrt(2.0)
            else:
                expected[mn, pq] = 2.0

        def coeff(i, j, k, l):
            ijkl = (i, j, k, l)
            if (ijkl == (m, n, p, q) or ijkl == (m, n, q, p) or
                ijkl == (n, m, p, q) or ijkl == (n, m, q, p)):
                return 1.0
            else:
                return 0.0
        actual = MandelVoigt(dim).create_array(coeff)
        assert_array_equal(expected, actual)
