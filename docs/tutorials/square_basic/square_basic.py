# Step 0
import itertools

import numpy as np

import janus.green as green
import janus.fft.serial as fft
import janus.material.elastic.linear.isotropic as material
import janus.operators as operators

from janus.operators import isotropic_4

# Step 1
class Example:
    def __init__(self, mat_i, mat_m, mat_0, n, a=0.5, dim=3):
        shape = tuple(itertools.repeat(n, dim))
        transform = fft.create_real(shape)
        self.mat_i = mat_i
        self.mat_m = mat_m
        self.n = n
        # ...
        # Step 2: creation of (C_i - C_0) and (C_m - C_0)
        # ...
        delta_C_i = isotropic_4(dim*(mat_i.k-mat_0.k),
                                2*(mat_i.g-mat_0.g), dim)
        delta_C_m = isotropic_4(dim*(mat_m.k-mat_0.k),
                                2*(mat_m.g-mat_0.g), dim)
        # ...
        # Step 3: creation of local operator ε ↦ (C-C_0):ε
        # ...
        ops = np.empty(shape, dtype=object)
        ops[:, :] = delta_C_m
        imax = int(np.ceil(n*a-0.5))
        ops[:imax, :imax] = delta_C_i
        self.eps_to_tau = operators.BlockDiagonalOperator2D(ops)
        # ...
        # Step 4: creation of non-local operator ε ↦ Γ_0[ε]
        # ...
        self.green = green.truncated(mat_0.green_operator(), shape, 1.,
                                     transform)

        # End of __init__


    def apply(self, x, out=None):
        if out is None:
            out = np.zeros_like(x)
        self.eps_to_tau.apply(x, out)
        self.green.apply(out, out)


if __name__ == '__main__':
    dim = 2
    sym = (dim*(dim+1))/2
    n = 256

    example = Square(mat_i=material.create(100.0, 0.2, dim),
                     mat_m=material.create(1.0, 0.3, dim),
                     mat_0=material.create(50.0, 0.3, dim),
                     n=n,
                     dim=dim)

    eps_macro = np.zeros((sym,), dtype=np.float64)
    eps_macro[-1] = 1.0

    eps = np.empty(example.green.ishape, dtype=np.float64)
    eps[...] = eps_macro
    eps_new = np.empty_like(eps)

    num_iter = 400
    res = np.empty((num_iter,), dtype=np.float64)

    for i in range(num_iter):
        example.apply(eps, out=eps_new)
        np.subtract(eps_macro, eps_new, out=eps_new)
        res[i] = np.sum((eps_new-eps)**2)
        eps, eps_new = eps_new, eps

    res = np.sqrt(res)
    tau = example.eps_to_tau.apply(eps)
    avg_tau = np.mean(tau, axis=(0, 1))
    C_1212 = example.green.green.mat.g+0.5*avg_tau[-1]
    print(C_1212)
