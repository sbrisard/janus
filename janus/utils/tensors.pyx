"""Creation and manipulation of tensors (:mod:`janus.utils.tensors`)
=================================================================

Functions and classes for the creation and manipulation of tensors.

Functions:

- :func:`isotropic_4` -- create a fourth-rank, isotropic tensor with minor
  symmetries.

Classes:

- :class:`FourthRankIsotropicTensor` -- fourth-rank, isotropic tensor.
- :class:`FourthRankIsotropicTensor2D`
- :class:`FourthRankIsotropicTensor3D`

"""
from cython cimport boundscheck
from cython cimport cdivision
from cython cimport wraparound


def isotropic_4(sph, dev, dim):
    """isotropic_4(sph, dev, dim)

    Create a fourth rank, isotrpic tensor.

    Parameters
    ----------
    sph: float
        The spherical projection of the returned tensor.
    dev: float
        The deviatoric projection of the returned tensor.
    dim: int
        The dimension of the physical space on which the returned tensor
        operates.

    """
    if dim == 2:
        return FourthRankIsotropicTensor2D(sph, dev, dim)
    elif dim == 3:
        return FourthRankIsotropicTensor3D(sph, dev, dim)
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(dim))


cdef class FourthRankIsotropicTensor(Operator):

    """
    Fourth rank, isotropic tensor with minor symmetries.

    Such a tensor is defined by its spherical and deviatoric
    projections. **Warning:** this class should *not* be instantiated
    directly, as the object returned by ``__cinit__`` would not be in a
    legal state. Use :func:`isotropic_4` instead.

    Parameters
    ----------
    sph : float64
        The spherical projection of the tensor.
    dev : float64
        The deviatoric projection of the tensor.
    dim : int
        The dimension of the space on which the tensor operates.

    See also
    --------
    isotropic_4

    Notes
    -----
    Any fourth rank, isotropic tensor `T` is a linear combination of the
    spherical projection tensor `J` and the deviatoric projection tensor
    `K`::

        T = sph * J + dev * K.

    `sph` and `dev` are the two coefficients which are passed to the
    constructor of this class. The components of `J` are::

        J_ijkl = δ_ij * δ_kl / d,

    where ``δ_ij`` denotes the Kronecker symbol, and `d` is the
    dimension of the space on which the tensor operates. The components
    of `K` are found from the identity ``K = I - J``, where `I` is the
    fourth-rank identity tensor::

        I_ijkl = (δ_ik * δ_jl + δ_il * δ_jk) / 2.

    """

    @cdivision(True)
    def __cinit__(self, double sph, double dev, int dim):
        self.dim = dim
        self.nrows = (self.dim * (self.dim + 1)) / 2
        self.ncols = self.nrows
        self.sph = sph
        self.dev = dev
        self.tr = (sph - dev) / <double> self.dim

    def __repr__(self):
        return ('FourthRankIsotropicTensor(sph={0}, dev={1}, dim={2})'
                .format(self.sph, self.dev, self.dim))


cdef class FourthRankIsotropicTensor2D(FourthRankIsotropicTensor):

    """Specialization of :class:`FourthRankIsotropicTensor` to 2D.

    """

    @boundscheck(False)
    @wraparound(False)
    cdef inline void c_apply(self, double[:] x, double[:] y):
        cdef double self_dev = self.dev
        cdef double x0 = x[0]
        cdef double x1 = x[1]
        cdef double aux = self.tr * (x0 + x1)
        y[0] = aux + self_dev * x0
        y[1] = aux + self_dev * x1
        y[2] = self_dev * x[2]


cdef class FourthRankIsotropicTensor3D(FourthRankIsotropicTensor):

    @boundscheck(False)
    @wraparound(False)
    cdef inline void c_apply(self, double[:] x, double[:] y):
        cdef double self_dev = self.dev
        cdef double x0 = x[0]
        cdef double x1 = x[1]
        cdef double x2 = x[2]
        cdef double aux = self.tr * (x0 + x1 + x2)
        y[0] = aux + self_dev * x0
        y[1] = aux + self_dev * x1
        y[2] = aux + self_dev * x2
        y[3] = self_dev * x[3]
        y[4] = self_dev * x[4]
        y[5] = self_dev * x[5]
