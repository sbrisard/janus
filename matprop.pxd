cdef class IsotropicLinearElasticMaterial:
    """This class defines an isotropic, linearly elastic material.

    Attributes
    ----------
        dim: int, read-only
            The dimension of the physical space (2: plane strain elasticity,
            3: 3d elasticity).
        k: float, read-only
            The bulk modulus.
        g: float, read-only
            The shear modulus.
        nu: float, read-only
            The Poisson ratio
            
    """
    cdef readonly int dim
    cdef readonly double k
    cdef readonly double g
    cdef readonly double nu

