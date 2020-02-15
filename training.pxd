import numpy as np
cimport numpy as np

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil

FLOAT = np.float32
ctypedef np.float32_t FLOAT_t

DOUBLE = np.float64
ctypedef np.float64_t DOUBLE_t

INT = np.int32
ctypedef np.int32_t INT_t
