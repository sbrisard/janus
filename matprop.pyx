cdef class IsotropicLinearElasticMaterial:

    def __cinit__(self, double g, double nu, int dim=3):
        self.dim = dim
        self.g = g
        self.nu = nu
        if dim == 2:
            self.k = g / (1. - 2. * nu)
        elif dim == 3:
            self.k = 2. / 3. * (1. + nu) / (1. - 2. * nu) * g
        else:
            raise ValueError('Parameter dim must be 2 or 3.')

