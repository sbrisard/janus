# Begin: imports
import itertools

import numpy as np
import matplotlib.pyplot as plt

import janus.green as green
import janus.fft.serial as fft
import janus.material.elastic.linear.isotropic as material
import janus.operators as operators

from janus.operators import isotropic_4
# End: imports

# Begin: init
class Example:
    def __init__(self, mat_i, mat_m, mat_0, n, a=0.5, dim=3):
        self.mat_i = mat_i
        self.mat_m = mat_m
        self.n = n
        shape = tuple(itertools.repeat(n, dim))
        # ...
        # End: init
        # Begin: create (C_i - C_0) and (C_m - C_0)
        # ...
        delta_C_i = isotropic_4(dim*(mat_i.k-mat_0.k),
                                2*(mat_i.g-mat_0.g), dim)
        delta_C_m = isotropic_4(dim*(mat_m.k-mat_0.k),
                                2*(mat_m.g-mat_0.g), dim)
        # ...
        # End: create (C_i - C_0) and (C_m - C_0)
        # Begin: create local operator ε ↦ (C-C_0):ε
        # ...
        ops = np.empty(shape, dtype=object)
        ops[:, :] = delta_C_m
        imax = int(np.ceil(n*a-0.5))
        ops[:imax, :imax] = delta_C_i
        self.eps_to_tau = operators.BlockDiagonalOperator2D(ops)
        # ...
        # End: create local operator ε ↦ (C-C_0):ε
        # Begin: create non-local operator ε ↦ Γ_0[ε]
        # ...
        self.green = green.truncated(mat_0.green_operator(),
                                     shape, 1.,
                                     fft.create_real(shape))
        # End: create non-local operator ε ↦ Γ_0[ε]


    # Begin: apply
    def apply(self, x, out=None):
        if out is None:
            out = np.zeros_like(x)
        self.eps_to_tau.apply(x, out)
        self.green.apply(out, out)
    # End: apply


# Begin: params
if __name__ == '__main__':
    dim = 2                # Spatial dimension
    sym = (dim*(dim+1))//2 # Dim. of space of second rank, symmetric tensors
    n = 256                # Number of cells along each side of the grid
    mu_i, nu_i = 100, 0.2  # Shear modulus and Poisson ratio of inclusion
    mu_m, nu_m = 1, 0.3    # Shear modulus and Poisson ratio of matrix
    mu_0, nu_0 = 50, 0.3   # Shear modulus and Poisson ratio of ref. mat.
    num_cells = n**dim     # Total number of cells
    # End: params
    # Begin: instantiate example
    example = Example(mat_i=material.create(mu_i, nu_i, dim),
                      mat_m=material.create(mu_m, nu_m, dim),
                      mat_0=material.create(mu_0, nu_0, dim),
                      n=n,
                      dim=dim)
    # End: instantiate example
    # Begin: define strains
    avg_eps = np.zeros((sym,), dtype=np.float64)
    avg_eps[-1] = 1.0

    eps = np.empty(example.green.ishape, dtype=np.float64)
    new_eps = np.empty_like(eps)
    # End: define strains
    # Begin: iterate
    num_iter = 400
    res = np.empty((num_iter,), dtype=np.float64)
    eps[...] = avg_eps
    normalization = 1/np.sqrt(num_cells)/np.linalg.norm(avg_eps)

    for i in range(num_iter):
        example.apply(eps, out=new_eps)
        np.subtract(avg_eps, new_eps, out=new_eps)
        res[i] = normalization*np.linalg.norm(new_eps-eps)
        eps, new_eps = new_eps, eps
    # End: iterate
    # Begin: post-process
    tau = example.eps_to_tau.apply(eps)
    avg_tau = np.mean(tau, axis=tuple(range(dim)))
    C_1212 = mu_0+0.5*avg_tau[-1]/avg_eps[-1]
    print(C_1212)

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Normalized residual')
    ax.loglog(res)
    fig.tight_layout(pad=0.2)
    fig.savefig('residual.png', transparent=True)

    fig, ax_array = plt.subplots(nrows=1, ncols=3)
    width, height = fig.get_size_inches()
    fig.set_size_inches(width, width/3)
    for i, ax in enumerate(ax_array):
        ax.set_axis_off()
        ax.imshow(eps[..., i], interpolation='nearest')
    fig.tight_layout(pad=0)
    fig.savefig('eps.png', transparent=True)
    # End: post-process
