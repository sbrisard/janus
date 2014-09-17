# cython: embedsignature=True

"""This module defines isotropic, linear elastic materials.

"""

from cython cimport cdivision

def create(g, nu, dim):
    """Create a new isotropic, linear and elastic material.

    `g` is the shear modulus, `nu` is the Poisson ratio. `dim` is the
    dimension of the physical space. It must be  2 (plane strain
    elasticity) or 3 (3D elasticity, default value).

    Returns a new instance of :class:`IsotropicLinearElasticMaterial`.

    """
    return IsotropicLinearElasticMaterial(g, nu, dim)

@cdivision(True)
cpdef double poisson_from_bulk_and_shear_moduli(double k, double g, int dim=3):
    """Compute the Poisson ratio from the bulk and shear moduli.

    `k` (resp. `g`) is the bulk (resp. shear) modulus. `dim` is the
    dimension of the physical space. It must be 2 (plane strain
    elasticity) or 3 (3D elasticity). No checks are performed on the
    positivity of the bulk and shear moduli, or on the validity of the
    returned Poisson ratio (which should lie between -1 and 1/2).

    """
    if dim == 2:
        return (k - g) / (2. * k)
    elif dim == 3:
        return (3 * k - 2 * g) / (2 * (3 * k + g))
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(dim))

cdef class IsotropicLinearElasticMaterial:

    @cdivision(True)
    def __cinit__(self, double g, double nu, int dim=3):
        if g < 0.0:
            raise ValueError('g must be >= 0 (was {0})'.format(g))
        if nu <= -1.0 or nu >= 0.5:
            raise ValueError('nu must be > -1 and < 1/2 (was {0})'.format(nu))
        if dim != 2 and dim != 3:
            raise ValueError('dim must be 2 or 3 (was {0})'.format(dim))
        self.g = g
        self.nu = nu
        self.dim = dim
        if dim == 2:
            self.k = g / (1. - 2. * nu)
        elif dim == 3:
            self.k = 2. / 3. * (1. + nu) / (1. - 2. * nu) * g
        else:
            raise RuntimeError('Inconsistent state')

    def __repr__(self):
        return ('IsotropicLinearElasticMaterial'
                '(g={0}, nu={1}, dim={2})').format(self.g,
                                                   self.nu,
                                                   self.dim)
