import pytest

from janus.matprop import IsotropicLinearElasticMaterial as Material

class TestIsotropicLinearElasticMaterial:

    @pytest.mark.parametrize('g, k, nu, dim', [(1.5, 3.75, 0.3, 2),
                                               (1.5, 3.25, 0.3, 3)])
    def test_init(self, g, k, nu, dim):
        mat = Material(g, nu, dim)
        assert mat.g == g
        assert mat.k == k
        assert mat.nu == nu

    @pytest.mark.parametrize('g, nu, dim', [(0.0, 0.3, 2),
                                            (0.0, 0.3, 3),
                                            (-1.0, 0.3, 2),
                                            (-1.0, 0.3, 3),
                                            (1.0, -1.0, 2),
                                            (1.0, -1.5, 2),
                                            (1.0, 0.5, 2),
                                            (1.0, 1.0, 2),
                                            (1.0, -1.0, 3),
                                            (1.0, -1.5, 3),
                                            (1.0, 0.5, 3),
                                            (1.0, 1.0, 3),
                                            (1.0, 0.3, 0),
                                            (1.0, 0.3, 4)])
    def test_init_invalid_params(self, g, nu, dim):
        with pytest.raises(ValueError):
            mat = Material(g, nu, dim)

    @pytest.mark.parametrize('name, value', [('g', 2.0),
                                             ('k', 10.0),
                                             ('nu', 0.5),
                                             ('dim', 2)])
    def test_read_only_attribute(self, name, value):
        mat = Material(1.0, 0.3, 3)
        with pytest.raises(AttributeError):
            setattr(mat, name, value)
