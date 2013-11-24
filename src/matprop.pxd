cdef class IsotropicLinearElasticMaterial:
    """This class defines an isotropic, linearly elastic material.

    Parameters
    ----------
    g : double
        Shear modulus.
    nu : double
        Poisson ratio.
    dim : int
        Dimension of the physical space [2: plane strain elasticity,
        3: 3d elasticity (default)].

    Attributes
    ----------
    dim : int, read-only
        Dimension of the physical space (2: plane strain elasticity,
        3: 3d elasticity).
    k : double, read-only
        Bulk modulus.
    g : double, read-only
        Shear modulus.
    nu : double, read-only
        Poisson ratio.

    """
    cdef readonly int dim
    cdef readonly double k
    cdef readonly double g
    cdef readonly double nu
