import pytest

from janus.mandelvoigt import MandelVoigt


# def test_create_array_3d():
#     # TODO finish test
#     mv = mandelvoigt.get_instance(3)
#     coeff = lambda i, j, k, l: 1.
#     a = mv.create_array(coeff)
#     print(a)

class TestMandelVoigt:

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
