cdef class IsotropicLinearElasticMaterial:
    cdef readonly int dim
    """Dimension of the physical space (``int``, read-only)."""

    cdef readonly double k
    """Bulk modulus (``float``, read-only)."""

    cdef readonly double g
    """Shear modulus (``float``, read-only)."""

    cdef readonly double nu
    """Poisson ratio (``float``, read-only)."""
