***********************************************************
Defining material constitutive laws (:mod:`janus.material`)
***********************************************************

.. automodule:: janus.material
   :members:

Elastic materials
=================

Isotropic, linear, elastic materials
------------------------------------

Classes and functions described in this chapter can be found in module
:mod:`janus.material.elastic.linear.isotropic`.

Such materials are created from their shear modulus and Poisson ratio as follows

>>> import janus.material.elastic.linear.isotropic as material
>>> mat = material.create(1.0, 0.3, 3)
>>> mat
IsotropicLinearElasticMaterial(g=1.0, nu=0.3, dim=3)
>>> mat.g # Shows the shear modulus
1.0
>>> mat.nu # Shows the Poisson ratio
0.3
>>> mat.k # Shows the bulk modulus
2.1666666666666665
>>>

where the :func:`janus.material.elastic.linear.isotropic.create` returns an instance of :class:`janus.material.elastic.linear.isotropic.IsotropicLinearElasticMaterial`

.. automodule:: janus.material.elastic.linear.isotropic
   :members:
