.. coding: utf-8-unix

***********************************
Defining material constitutive laws
***********************************

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

Linear, elastic materials
-------------------------

Isotropic, linear, elastic materials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Such materials are created from their shear modulus (μ) and Poisson ratio (ν) as follows

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
>>>

The function :func:`janus.material.elastic.linear.isotropic.create` takes two positional arguments: the shear modulus and the Poisson ratio, and one optional argument: the dimension of the physical space, which can be 2 (plane strain elasticity) or 3 (3D elasticity); the default value is 3. To create a *plane stress* (μ, ν) material, a *plane strain* (μ, ν') material should be created, with ν' = ν / (1 + ν).

A helper function, :func:`janus.material.elastic.linear.isotropic.poisson_from_bulk_and_shear_moduli` is also provided. It returns the Poisson ratio, computed from the bulk and shear moduli.
