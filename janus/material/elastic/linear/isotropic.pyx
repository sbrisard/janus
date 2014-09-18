"""This module defines isotropic, linear elastic materials.

Materials defined in this module can operate on physical space of
dimension dim = 2 or 3. dim = 2 refers to plane strain elasticity, while
dim = 3 refers to classical 3D elasticity. Plane stress elasticity is
not available.

"""

from cython cimport cdivision


def create(g, nu, dim):
    """Create a new isotropic, linear and elastic material.

    Args:
        g: the shear modulus
        nu: the Poisson ratio
        dim: the dimension of the physical space (default 3)

    Returns:
        A new instance of IsotropicLinearElasticMaterial.

    """
    return IsotropicLinearElasticMaterial(g, nu, dim)


@cdivision(True)
cpdef double poisson_from_bulk_and_shear_moduli(double k, double g, int dim=3):
    """Compute the Poisson ratio from the bulk and shear moduli.

    No checks are performed on the positivity of the bulk and shear
    moduli, or on the validity of the returned Poisson ratio (which
    should lie between -1 and 1/2).

    Args:
        k: the bulk modulus
        g: the shear modulus
        dim: the dimension of the physical space (default 3). Must be 2
            (plane strain elasticity) or 3 (3D elasticity).


    Returns:
        The Poisson ratio.

    """
    if dim == 2:
        return (k - g) / (2. * k)
    elif dim == 3:
        return (3 * k - 2 * g) / (2 * (3 * k + g))
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(dim))


cdef class IsotropicLinearElasticMaterial:

    """This class defines isotropic, linear and elastic materials.

    The dimension of the physical space must be 2 (plane strain
    elasticity) or 3 (3D elasticity). Plane stress elasticity is not
    available.

    Args:
        g: the shear modulus
        nu: the Poisson ratio
        dim: the dimension of the physical space (default 3)

    Attributes:
        k: the bulk modulus (read-only)
        g: the shear modulus (read-only)
        nu: the Poisson ratio (read-only)
        dim: the dimension of the physical space (read-only)

    """

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
