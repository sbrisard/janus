
def _check_index(i, dim):
    if i < 0 or i >= dim:
        raise ValueError('index must be >= 0 and < {0} (was {1})'
                         .format(dim, i))

def _check_multi_index(multi_index, dim):
    if len(multi_index) != 2:
        raise ValueError('length of multi_index must be 2 (was {0})'
                         .format(len(multi_index)))
    _check_index(multi_index[0], dim)
    _check_index(multi_index[1], dim)

class MandelVoigt:
    def __init__(self, dim):
        x = 0
        y = 1
        if dim == 2:
            self._to_index = [[0, 2], [2, 1]]
            self._to_multi_index = [(x, x), (y, y), (x, y)]
        elif dim == 3:
            z = 2
            self._to_index = [[0, 5, 4], [5, 1, 3], [4, 3, 2]]
            self._to_multi_index = [(x, x), (y, y), (z, z),
                                    (y, z), (z, x), (x, y)]
        else:
            raise ValueError('dim must be 2 or 3 (was {0})'.format(dim))
        self.dim = dim
        self.sym = (dim * (dim + 1)) // 2

    def unravel_index(self, index):
        _check_index(index, self.sym)
        return self._to_multi_index[index]

    def ravel_multi_index(self, multi_index):
        _check_multi_index(multi_index, self.dim)
        return self._to_index[multi_index[0]][multi_index[1]]
        
mandel_voigt_2d = MandelVoigt(2)
mandel_voigt_3d = MandelVoigt(3)

def get_instance(dim):
    if dim == 2:
        return mandel_voigt_2d
    elif dim == 3:
        return mandel_voigt_3d
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(dim))
