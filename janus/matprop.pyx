cdef class IsotropicLinearElasticMaterial:

    def __cinit__(self, double g, double nu, int dim=3):
        if g <= 0.0:
            raise ValueError('g must be > 0 (was {0})'.format(g))
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
