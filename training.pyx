import cython
cimport cython
import random
import numpy as np
cimport numpy as np
from cpython cimport array

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport exp
from libc.math cimport log
from libc.stdio cimport printf

try:
    from scipy.linalg.blas import fblas
except ImportError:
    import scipy.linalg.blas as fblas


cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)


#cdef unsigned long long next_random = 1
cdef unsigned long long lkg_f = 25214903917ULL
cdef int MAX_EXP = 6

cdef int one_int = 1
cdef FLOAT_t one_float = 1.
cdef FLOAT_t zero_float = 0.


@cython.cdivision(True)
cdef inline FLOAT_t sigmoid(float x) nogil:
    cdef FLOAT_t ret_val = exp(<double> -x)
    ret_val = 1/(1+ret_val)
    return ret_val


@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
cdef void update_gradients_word(int center_word_ind, FLOAT_t *center_word_vectors, int outside_word_ind, FLOAT_t *outside_word_vectors, int num_features, int[:] rand2sample, int sample_table_size, int num_samples, FLOAT_t step, double *loss, int* neg_samples, int processed_words, bint get_loss) nogil:

    cdef int i
    cdef int target_ind

    cdef int cwv_ind = center_word_ind * num_features
    cdef int owv_ind = 0

    cdef float dot_product
    cdef float l


    cdef FLOAT_t label 
    cdef FLOAT_t grad

    for i in range(num_samples + 1):
        if i == 0:
            target_ind = outside_word_ind
            label = one_float
        else:
            target_ind = rand2sample[neg_samples[processed_words*num_samples + i-1]]
            if target_ind == outside_word_ind:
                continue

            label = zero_float


        owv_ind = target_ind * num_features

        dot_product = sdot(&num_features, &center_word_vectors[cwv_ind], &one_int, &outside_word_vectors[owv_ind], &one_int)

        if dot_product <= -MAX_EXP or dot_product >= MAX_EXP:
            continue

        grad = (label - sigmoid(dot_product)) * step


        saxpy(&num_features, &grad, &outside_word_vectors[owv_ind], &one_int, &center_word_vectors[cwv_ind], &one_int)
        saxpy(&num_features, &grad, &center_word_vectors[cwv_ind], &one_int, &outside_word_vectors[owv_ind], &one_int)

        if get_loss:
            dot_product = dot_product if i == 0 else -dot_product
            l = log(sigmoid(dot_product))
            loss[0] = loss[0] - l
            

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
@cython.cdivision(True)
def update_gradients_batch(np.ndarray[FLOAT_t, ndim=2] _center_word_vectors, np.ndarray[FLOAT_t, ndim=2] _outside_word_vectors, list sentences, int batch_size, int num_features, int[:] rand2sample, int sample_table_size, int num_samples, int window_size, FLOAT_t step, float[:] undersampling_table, unsigned long long[:] next_random, bint get_loss):

    cdef int[:] sentence

    cdef int processed_words = 0
    cdef np.ndarray[INT_t, ndim = 1] _reduced_windows = np.random.randint(0, window_size, batch_size + 50000)
    reduced_windows = <int *>(np.PyArray_DATA(_reduced_windows))


    #An extremely hacky and memory-intensive solution to generating negative samples
    #Ideally, the negative sample indexes should be created on-the-fly 
    #I opted for this to avoid thinking about how to implement a RNG with a uniform distribution over an uncertain integer interval
    cdef np.ndarray[INT_t, ndim = 1] _neg_samples = np.random.randint(0, sample_table_size, (batch_size+1000)*num_samples) 
    neg_samples = <int *>(np.PyArray_DATA(_neg_samples))

    cdef int i, j
    cdef int len_sentence
    cdef int l, h


    cdef double loss = 0.

    center_word_vectors = <FLOAT_t *>(np.PyArray_DATA(_center_word_vectors))
    outside_word_vectors = <FLOAT_t *>(np.PyArray_DATA(_outside_word_vectors))

    for sentence in sentences:
        len_sentence = len(sentence)
        for i in range(len_sentence):
            if undersampling_table[sentence[i]] < <float>(rand()/RAND_MAX):
                continue
            l = i - window_size + reduced_windows[processed_words]
            if l < 0:
                l = 0
            h = i + window_size + 1 - reduced_windows[processed_words]
            if h > len_sentence:
                h = len_sentence
            for j in range(l,h):
                if j == i:
                    continue
                update_gradients_word(sentence[i], center_word_vectors,  sentence[j], outside_word_vectors, num_features, rand2sample, sample_table_size, num_samples, step, &loss, neg_samples, processed_words, get_loss)
            processed_words += 1
            if(processed_words % 100000 == 0):
                printf("%d\n", processed_words)
    return loss


