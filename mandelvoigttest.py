from nose.tools import nottest
from nose.tools import raises

import mandelvoigt


@nottest
@raises(ValueError)
def do_test_constructor_invalid_dimension(dim):
    mandelvoigt.MandelVoigt(dim)


def test_constructor_invalid_dimension():
    for dim in [1, 4]:
        yield do_test_constructor_invalid_dimension, dim


@nottest
@raises(ValueError)
def do_test_get_instance_invalid_dimension(dim):
    mandelvoigt.get_instance(dim)


def test_get_instance_invalid_dimension():
    for dim in [1, 4]:
        yield do_test_get_instance_invalid_dimension, dim


@nottest
def do_test_unravel_index(index, expected, dim):
    mv = mandelvoigt.get_instance(dim)
    actual = mv.unravel_index(index)
    assert expected == actual


def test_unravel_index_2d():
    dim = 2
    indices = [0, 1, 2]
    expecteds = [(0, 0), (1, 1), (0, 1)]
    for index, expected in zip(indices, expecteds):
        yield do_test_unravel_index, index, expected, dim


def test_unravel_index_3d():
    dim = 3
    indices = [0, 1, 2, 3, 4, 5, 6]
    expecteds = [(0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1)]
    for index, expected in zip(indices, expecteds):
        yield do_test_unravel_index, index, expected, dim


@nottest
@raises(ValueError)
def do_test_unravel_invalid_index(index, dim):
    mandelvoigt.get_instance(dim).unravel_index(index)


def test_unravel_invalid_index():
    indices = [-1, 3, -1, 6]
    dims = [2, 2, 3, 3]
    for index, dim in zip(indices, dims):
        yield do_test_unravel_invalid_index, index, dim


@nottest
def do_test_ravel_multi_index(i, j, expected, dim):
    mv = mandelvoigt.get_instance(dim)
    actual = mv.ravel_multi_index(i, j)
    assert expected == actual


def test_ravel_multi_index_2d():
    dim = 2
    multi_indices = [(0, 0), (1, 1), (0, 1), (1, 0)]
    expecteds = [0, 1, 2, 2]
    for (i, j), expected in zip(multi_indices, expecteds):
        yield do_test_ravel_multi_index, i, j, expected, dim


def test_ravel_multi_index_3d():
    dim = 3
    multi_indices = [(0, 0), (1, 1), (2, 2), (1, 2), (2, 1),
                     (2, 0), (0, 2), (0, 1), (1, 0)]
    expecteds = [0, 1, 2, 3, 3, 4, 4, 5, 5]
    for (i, j), expected in zip(multi_indices, expecteds):
        yield do_test_ravel_multi_index, i, j, expected, dim


def test_create_array_3d():
    # TODO finish test
    mv = mandelvoigt.get_instance(3)
    coeff = lambda i, j, k, l: 1.
    a = mv.create_array(coeff)
    print(a)
