janus.fft package
=================

Planner flags
-------------

From the FFTW manual (Sec. `Planner Flags <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_)

    All of the planner routines in FFTW accept an integer ``flags`` argument, which is
    a bitwise ``OR`` (‘|’) of zero or more of the flag constants defined below. These
    flags control the rigor (and time) of the planning process, and can also impose
    (or lift) restrictions on the type of transform algorithm that is employed.

    *Important:* the planner overwrites the input array during planning unless a saved
    plan (see Wisdom) is available for that problem, so you should initialize your
    input data after creating the plan. The only exceptions to this are the
    :data:`FFTW_ESTIMATE` and :data:`FFTW_WISDOM_ONLY` flags, as mentioned below.

    In all cases, if wisdom is available for the given problem that was created with
    equal-or-greater planning rigor, then the more rigorous wisdom is used. For
    example, in :data:`FFTW_ESTIMATE` mode any available wisdom is used, whereas in
    :data:`FFTW_PATIENT` mode only wisdom created in patient or exhaustive mode can be
    used. See Words of Wisdom-Saving Plans.

Only the names of the flags are reproduced below. The reader should refer to the FFTW manual for a complete description.

Planning-rigor flags
^^^^^^^^^^^^^^^^^^^^

.. data:: FFTW_ESTIMATE

    Use simple heuristic to pick a plan.

.. data:: FFTW_MEASURE

    More accurate selection of a plan (default planning option).

.. data:: FFTW_PATIENT

    Even more accurate selection of a plan.

.. data:: FFTW_EXHAUSTIVE

    Even more accurate selection of a plan.

.. data:: FFTW_WISDOM_ONLY

    Should be used only to check whether wisdom *is* available.

Algorithm-restriction flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^

These flags are exposed for future extensions of the module. They are not used for the time being.

.. data:: FFTW_DESTROY_INPUT

    Unused.

.. data:: FFTW_PRESERVE_INPUT

    Unused.

.. data:: FFTW_UNALIGNED

    Unused.


Subpackages
-----------

.. toctree::

    janus.fft.parallel
    janus.fft.serial

Module contents
---------------

.. automodule:: janus.fft
    :members:
    :undoc-members:
    :show-inheritance:
