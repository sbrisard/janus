.. coding: utf-8-unix

.. _materials:

*********
Materials
*********

Classes and functions for the definition of materials are provided in
the :mod:`janus.material` package. This package is structured in three
levels

  1. the physical model (e.g. elasticity, thermal conductivity, ...),
  2. linear/nonlinear constitutive law,
  3. material symmetries (isotropic, transverse isotropic, ...).

Regardless of the actual constitutive law, an attempt is made to expose
a unified interface. For example, a new instance of a specific material
can be created through the function ``create()`` of the corresponding
module.

Elastic materials
=================

Isotropic, linear, elastic materials
------------------------------------

Such materials are defined in the module :mod:`janus.material.elastic.linear.isotropic`. They are created from their shear modulus (μ) and Poisson ratio (ν) as follows

>>> import janus.material.elastic.linear.isotropic as material
>>> mat = material.create(1.0, 0.3, 3)
>>> mat
IsotropicLinearElasticMaterial(g=1.0, nu=0.3, dim=3)
>>> mat.g # Shear modulus
1.0
>>> mat.nu # Poisson ratio
0.3
>>> mat.k # Bulk modulus
2.1666666666666665

The function :func:`janus.material.elastic.linear.isotropic.create` takes two positional arguments: the shear modulus and the Poisson ratio, and one optional argument: the dimension of the physical space, which can be 2 (plane strain elasticity) or 3 (3D elasticity); the default value is 3. To create a *plane stress* (μ, ν) material, a *plane strain* (μ, ν') material should be created, with ν' = ν / (1 + ν).

A helper function, :func:`janus.material.elastic.linear.isotropic.poisson_from_bulk_and_shear_moduli` is also provided. It returns the Poisson ratio, computed from the bulk and shear moduli.

Green operators for strains associated with a given material are instanciated with the ``green_operator()`` method, like so

>>> green = mat.green_operator()
>>> green
Green Operator(IsotropicLinearElasticMaterial(g=1.0, nu=0.3, dim=3))

The returned operator can then be manipulated frequency-wise

>>> import numpy as np
>>> k = np.array([1.0, 2.0, 3.0])
>>> green.set_frequency(k)
>>> np.asarray(green.to_memoryview())
array([[ 0.06778426, -0.01457726, -0.03279883, -0.03092304,  0.13606136,
         0.09070758],
       [-0.01457726,  0.22740525, -0.13119534,  0.17935362, -0.06184607,
         0.05978454],
       [-0.03279883, -0.13119534,  0.34766764,  0.02473843,  0.01236921,
        -0.09276911],
       [-0.03092304,  0.17935362,  0.02473843,  0.20189504, -0.05976676,
         0.0196793 ],
       [ 0.13606136, -0.06184607,  0.01236921, -0.05976676,  0.29154519,
         0.17055394],
       [ 0.09070758,  0.05978454, -0.09276911,  0.0196793 ,  0.17055394,
         0.14941691]])

API of module :mod:`janus.material.elastic.linear.isotropic`
------------------------------------------------------------

.. automodule:: janus.material.elastic.linear.isotropic
   :members:
   :private-members:
   :undoc-members:
