# Only the functions which are necessary to the code are exposed

cdef extern from "fftw3.h":
    #Memory allocation
    ctypedef double fftw_complex[2]

    void *fftw_malloc(size_t)
    void fftw_free(void *)
    double *fftw_alloc_real(size_t)
    fftw_complex *fftw_alloc_complex(size_t)

    #
    # Functions related to plans
    #
    cdef int FFTW_MEASURE
    cdef int FFTW_DESTROY_INPUT
    cdef int FFTW_UNALIGNED
    cdef int FFTW_CONSERVE_MEMORY
    cdef int FFTW_EXHAUSTIVE
    cdef int FFTW_PRESERVE_INPUT
    cdef int FFTW_PATIENT
    cdef int FFTW_ESTIMATE
    cdef int FFTW_WISDOM_ONLY

    ctypedef struct _fftw_plan:
       pass

    ctypedef _fftw_plan *fftw_plan

    void fftw_execute(fftw_plan)
    void fftw_destroy_plan(fftw_plan)
    fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1,
                                   double *input, fftw_complex *output,
                                   unsigned flags)
    fftw_plan fftw_plan_dft_c2r_2d(int n0, int n1,
                                   fftw_complex *input, double *output,
                                   unsigned flags)
    fftw_plan fftw_plan_dft_r2c_3d(int n0, int n1, int n2,
                                   double *input, fftw_complex *output,
                                   unsigned flags)
    fftw_plan fftw_plan_dft_c2r_3d(int n0, int n1, int n2,
                                   fftw_complex *input, double *output,
                                   unsigned flags)
