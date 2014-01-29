from cython.view cimport array

cdef inline void check_shape_1d(double[:] a, int n0) except *:
    if a.shape[0] != n0:
        raise ValueError('invalid shape: expected ({0},), actual ({1},)'
                         .format(n0, a.shape[0]))

cdef inline double[:] create_or_check_shape_1d(double[:] a, int n0) except *:
    if a is None:
        return array((n0,), sizeof(double), 'd')
    check_shape_1d(a, n0)
    return a

cdef inline void check_shape_2d(double[:, :] a, int n0, int n1) except *:
    if (a.shape[0] != n0) or (a.shape[1] != n1):
        raise ValueError('invalid shape: expected ({0}, {1}), actual ({2}, {3})'
                         .format(n0, n1, a.shape[0], a.shape[1]))

cdef inline double[:, :] create_or_check_shape_2d(double[:, :] a,
                                                  int n0, int n1) except *:
    if a is None:
        return array((n0, n1), sizeof(double), 'd')
    check_shape_2d(a, n0, n1)
    return a

cdef inline void check_shape_3d(double[:, :, :] a,
                                int n0, int n1, int n2) except *:
    if (a.shape[0] != n0) or (a.shape[1] != n1) or (a.shape[2] != n2):
        raise ValueError('invalid shape: expected ({0}, {1}, {2}), actual ({3}, {4}, {5})'.format(n0, n1, n2, a.shape[0], a.shape[1], a.shape[2]))

cdef inline double[:, :, :] create_or_check_shape_3d(double[:, :, :] a,
                                                     int n0, int n1, int n2) except *:
    if a is None:
        return array((n0, n1, n2), sizeof(double), 'd')
    check_shape_3d(a, n0, n1, n2)
    return a

cdef inline void check_shape_4d(double[:, :, :, :] a,
                                int n0, int n1, int n2, int n3) except *:
    if (a.shape[0] != n0) or (a.shape[1] != n1) or (a.shape[2] != n2) or (a.shape[3] != n3):
        raise ValueError('invalid shape: expected ({0}, {1}, {2}, {3}), actual ({4}, {5}, {6}, {7})'.format(n0, n1, n2, n3, a.shape[0], a.shape[1], a.shape[2], a.shape[3]))

cdef inline double[:, :, :, :] create_or_check_shape_4d(double[:, :, :, :] a,
                                                        int n0, int n1, int n2, int n3) except *:
    if a is None:
        return array((n0, n1, n2, n3), sizeof(double), 'd')
    check_shape_4d(a, n0, n1, n2, n3)
    return a

cdef inline void check_shape_1i(int[:] a, int n0) except *:
    if a.shape[0] != n0:
        raise ValueError('invalid shape: expected ({0},), actual ({1},)'
                         .format(n0, a.shape[0]))
