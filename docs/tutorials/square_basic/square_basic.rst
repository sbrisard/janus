.. -*- coding: utf-8-unix -*-

******************************
Square inclusion, basic scheme
******************************

In this tutorial, we will compute the effective elastic properties of a simple 2D microstructure in plane strain. More precisely, we consider a periodic microstructure made of a square inclusion of size :math:`a`, embedded in a unit-cell of size :math:`L` (see Fig. ":ref:`fig_microstructure`").

.. _fig_microstructure:
.. figure:: microstructure.*
   :align: center

   The periodic microstructure under consideration.

   :math:`\shearModulus_\mathrm i` (resp. :math:`\shearModulus_\mathrm m`) denotes the shear modulus of the inclusion (resp. the matrix); :math:`\PoissonRatio_\mathrm i` (resp. :math:`\PoissonRatio_\mathrm m`) denotes the Poisson ratio of the inclusion (resp. the matrix).

The effective properties of this periodic microstructure are derived from the solution to the so-called *corrector* problem

.. math::
   :nowrap:
   :label: corrector_problem

   \begin{subequations}
   \begin{gather}
      \nabla\cdot\tens\stress=\vec 0,\\
      \tens\stress=\tens[4]\Stiffness:\tens\strain,\\
      \tens\strain=\tens\Strain+\nabla^\s\vec\displacement,
   \end{gather}
   \end{subequations}

where :math:`\vec\displacement` denotes the unknown, periodic displacement, :math:`\tens\strain` (resp. :math:`\tens\stress`) is the local strain (resp. stress) and :math:`\tens[4]\Stiffness` is the local stiffness (inclusion or matrix). From the solution to the above problem, the effective stiffness :math:`\tens[4]\Stiffness^\eff` is defined as the tensor mapping the macroscopic (imposed) strain :math:`\tens\Strain=\volavg{\tens\strain}` to the macroscopic stress :math:`\tens\Stress=\overline{\tens\stress}` (where overlined quantities denote volume averages)

.. math::

   \tens[4]\Stiffness^\eff:\tens\Strain=\frac1{L^2}\int_{\left(0,L\right)^2}\tens\stress(x_1,x_2)\,\diff x_1\,\diff x_2.

In the present tutorial, we shall concentrate on the 1212 component of the effective stiffness, that is to say that the following macroscopic strain will be imposed

.. math::

   \tens\Strain=\vec e_1\otimes\vec e_2+\vec e_2\otimes\vec e_1,

and the volume average :math:`\volavg{\stress_{12}}` will be evaluated. To do so, the boundary value problem :eq:`corrector_problem` is transformed into an integral equation, known as the Lippmann--Schwinger equation (:ref:`Korringa, 1973 <KORR1973>`; :ref:`Zeller & Dederichs, 1973 <ZELL1973>` ; :ref:`Kröner, 1974 <KRON1974>`) . This equation reads

.. math::
   :label: Lippmann-Schwinger

   \tens\strain+\tens[4]\GreenOperator_0[\left(\tens[4]\Stiffness-\tens[4]\Stiffness_0\right):\tens\strain]=\tens\Strain,

where :math:`\tens[4]\Stiffness_0` denotes the stiffness of the reference material, :math:`\tens[4]\GreenOperator_0` the related Green operator for strains, and :math:`\tens\strain` the local strain tensor. We will assume that the reference material is isotropic, with shear modulus :math:`\shearModulus_0` and Poisson ratio :math:`\PoissonRatio_0`.

Following :ref:`Moulinec and Suquet (1998) <MOUL1998>`, the above Lippmann--Schwinger equation :eq:`Lippmann-Schwinger` is solved by means of fixed point iterations

.. math::
   :label: basic_scheme

   \tens\strain^{n+1}=\tens\Strain-\tens[4]\GreenOperator_0[\left(\tens[4]\Stiffness-\tens[4]\Stiffness_0\right):\tens\strain^n].

Finally, the above iterative scheme is discretized over a regular grid, leading to the basic uniform grid, periodic Lippmann--Schwinger solver. Let's see now how this is done with Janus: we must first create a class which implements the operator to be iterated

.. math::
   \tens\strain\mapsto\tens\Strain-\tens[4]\GreenOperator_0\left[\left(\tens[4]\Stiffness-\tens[4]\Stiffness_0\right):\tens\strain\right]

for the microstructure under consideration. This will be done by composing to successive operators, namely (i) the local operator

.. math::
   :label: local_operator

   \tens\strain\mapsto\left(\tens[4]\Stiffness-\tens[4]\Stiffness_0\right):\tens\strain,

and (ii) the Green operator for strains

.. math::
   :label: non-local_operator

   \tens\stressPolarization\mapsto\tens[4]\GreenOperator_0[\tens\stressPolarization].

For the implementation of the local operator defined by Eq. :eq:`local_operator`, it is first observed that :math:`\tens[4]\Stiffness_0`, :math:`\tens[4]\Stiffness_\mathrm{i}` and :math:`\tens[4]\Stiffness_\mathrm{m}` being isotropic materials, :math:`\tens[4]\Stiffness-\tens[4]\Stiffness_0` is an isotropic tensor at any point of the unit-cell. In other words, both :math:`\tens[4]\Stiffness_\text i-\tens[4]\Stiffness_0` and :math:`\tens[4]\Stiffness_\text m-\tens[4]\Stiffness_0` will be defined as :class:`FourthRankIsotropicTensor <janus.operators.FourthRankIsotropicTensor>`.

Furthermore, this operator is *local*. In other words, the output value in cell ``(i0, i1)`` depends on the input value in the same cell only (the neighboring cells are ignored). More precisely, we assume that a uniform grid of shape ``(n, n)`` is used to discretized Eq. :eq:`basic_scheme`. Then the material properties are constant in each cell, and we define ``delta_C[i0, i1, :, :]`` the matrix representation of :math:`\tens[4]\Stiffness-\tens[4]\Stiffness_0` (see :ref:`Mandel_notation`). Likewise, ``eps[i0, i1, :]`` is the vector representation of the strain tensor in cell ``(i0, i1)``. Then, the stress-polarization :math:`\left(\tens[4]\Stiffness-\tens[4]\Stiffness_0\right):\tens\strain` in cell ``(i0, i1)`` is given by the expression::

    tau[i0, i1] = delta_C[i0, i1] @ eps[i0, i1],

where ``@`` denotes the matrix multiplication operator. It results from the above relation that the lcoal operator defined by :eq:`local_operator` should be implemented as a :class:`BlockDiagonalOperator2D <janus.operators.BlockDiagonalOperator2D>`. As for the non-local operator, it is instanciated by a simple call to the ``green_operator`` method of the relevant material (see :ref:`materials`).

We start with imports from the standard library, the SciPy stack and Janus itself:

.. literalinclude:: square_basic.py
   :start-after: Step 0
   :end-before: Step 1

The first few lines of the initializer are pretty standard

.. literalinclude:: square_basic.py
   :start-after: Step 1
   :end-before: Step 2

We then define the local operators :math:`\tens[4]\Stiffness_\text i-\tens[4]\Stiffness_0` and :math:`\tens[4]\Stiffness_\text m-\tens[4]\Stiffness_0`, which will be used to create the operator :math:`\tens\strain\mapsto\left(\tens[4]\Stiffness-\tens[4]\Stiffness_0\right):\tens\strain` as a . It is first observed that the inclusion and matrix are made of isotropic materials,

.. literalinclude:: square_basic.py
   :start-after: Step 2
   :end-before: Step 3

Block-diagonal operators are initialized from an array of local operators, called `ops` below

.. literalinclude:: square_basic.py
   :start-after: Step 3
   :end-before: Step 4

The upper-left quarter of the unit-cell is filled with `aux_i` (:math:`\tens[4]\Stiffness_\text i-\tens[4]\Stiffness_0`), while the remainder of the unit-cell receives `aux_m` (:math:`\tens[4]\Stiffness_\text m-\tens[4]\Stiffness_0`).

The complete program
====================

.. literalinclude:: square_basic.py
