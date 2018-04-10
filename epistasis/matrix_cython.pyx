import numpy as np

DTYPE = np.int

cimport numpy as np

ctypedef np.int_t DTYPE_t

def build_model_matrix(np.ndarray encoding_vectors, np.ndarray sites):

    cdef int n = encoding_vectors.shape[0]
    cdef int m = sites.shape[0]
    cdef int i, j, k, l, element
    cdef np.ndarray matrix = np.ones([n, m], dtype=DTYPE)

    # Interate over rows
    for i in range(n):

        # Iterate over cols
        for j in range(m):
            element = 1
            # Product of sites in encoding vector
            l = sites[j].shape[0]
            for k in range(l):
                element = element * encoding_vectors[i, sites[j][k]]

            # Multiply these elements
            matrix[i,j] = element

    return matrix
