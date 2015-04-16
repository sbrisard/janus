"""
This module is a Python wrapper around the FFTW library. The core of the
wrapper is located in the two submodules :mod:`janus.fft.serial` and
:mod:`janus.fft.parallel`, which should be imported explicitely.

.. _planner-flags:

Planner flags
-------------

From the FFTW manual (Sec. `Planner Flags <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_)

    All of the planner routines in FFTW accept an integer ``flags`` argument,
    which is a bitwise ``OR`` (‘|’) of zero or more of the flag constants
    defined below. These flags control the rigor (and time) of the planning
    process, and can also impose (or lift) restrictions on the type of
    transform algorithm that is employed.

Only the names of the flags are reproduced below. The reader should refer to
the FFTW manual for a complete description.

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

These flags are exposed for future extensions of the module. They are not used
for the time being.

.. data:: FFTW_DESTROY_INPUT

    Unused.

.. data:: FFTW_PRESERVE_INPUT

    Unused.

.. data:: FFTW_UNALIGNED

    Unused.
"""

from .serial._serial_fft import FFTW_ESTIMATE
from .serial._serial_fft import FFTW_MEASURE
from .serial._serial_fft import FFTW_PATIENT
from .serial._serial_fft import FFTW_EXHAUSTIVE
from .serial._serial_fft import FFTW_WISDOM_ONLY
from .serial._serial_fft import FFTW_DESTROY_INPUT
from .serial._serial_fft import FFTW_PRESERVE_INPUT
from .serial._serial_fft import FFTW_UNALIGNED
from .serial._serial_fft import FFTW_CONSERVE_MEMORY
