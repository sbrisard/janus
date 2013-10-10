from libc.stddef cimport ptrdiff_t
from mpi4py.MPI cimport Comm

cpdef init()
cdef create_real_2D(ptrdiff_t n0, ptrdiff_t n1, Comm comm)
cdef create_real_3D(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2, Comm comm)

