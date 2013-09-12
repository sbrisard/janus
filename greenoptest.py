import mandelvoigt
from matprop import IsotropicLinearElasticMaterial as Material

def delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def green_coefficient(i, j, k, l, n, mat):
    return ((delta(i, k) * n[j] * n[l]
             + delta(i, l) * n[i] * n[k]
             + delta(j, k) * n[i] * n[l]
             + delta(j, l) * n[i] * n[k]) / mat.g
            - n[i] * n[j] * n[k] * n[l] / (2. * mat.g * (1. - mat.nu)))

def green_matrix(n, mat):
    return mandelvoigt.get_instance(mat.dim).create_array(green_coefficient, (n, mat))

if __name__ == '__main__':
    import numpy as np

    mat = Material(g=1., nu=0.3)
    n = np.array([0., 0., 1.])
    a = green_matrix(n, mat)
