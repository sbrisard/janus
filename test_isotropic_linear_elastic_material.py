from nose.tools import assert_almost_equals
from nose.tools import assert_equals
from nose.tools import raises

from janus.matprop import IsotropicLinearElasticMaterial as Material

def dotest_cinit(g, k, nu, dim):
    mat = Material(g, nu, dim)
    assert_equals(g, mat.g)
    assert_equals(k, mat.k)
    assert_equals(nu, mat.nu)
    
def test_cinit_2d():
    """Constructor (2D material)"""
    dotest_cinit(1.5, 3.75, 0.3, 2)

def test_cinit_3d():
    """Constructor (3D material)"""
    dotest_cinit(1.5, 3.25, 0.3, 3)

@raises(ValueError)
def test_cinit_invalid_dimension():
    """Constructor (invalid dimension)"""
    mat = Material(1., 0.3, 4)

@raises(AttributeError)
def test_cinit_readonly_dimension():
    """dim attribute is read-only"""
    mat = Material(1., 0.3, 3)
    mat.dim = 2

@raises(AttributeError)
def test_cinit_readonly_shear_modulus():
    """g attribute is read-only"""
    mat = Material(1., 0.3, 3)
    mat.g = 0.0

@raises(AttributeError)
def test_cinit_readonly_bulk_modulus():
    """k attribute is read-only"""
    mat = Material(1., 0.3, 3)
    mat.k = 0.0

@raises(AttributeError)
def test_cinit_readonly_poisson_ratio():
    """nu attribute is read-only"""
    mat = Material(1., 0.3, 3)
    mat.nu = 0.0

