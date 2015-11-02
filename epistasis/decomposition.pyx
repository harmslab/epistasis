# ---------------------------------------------------
# Decomposition matrix for epistasis models
# ---------------------------------------------------

import numpy as np

def generate_dv_matrix(sequences, interactions, encoding={"1": 1, "0": 0}):
    """ Build the X matrix of dummy variable for linear regression
        in epistasis model.
    """
    # Type all iterators to generate matrix in C
    cdef int i, j, k, m, n, l

    # Get bounds on loops
    dim1 = len(sequences)
    dim2 = len(interactions)

    # Initialize the matrix
    x = np.ones((dim1, dim2), dtype=int)

    # Iterate through matrix elements
    for n in range(dim1):

        # Skip first index -- this is wildtype
        for i in range(1,dim2):

            # initialize element encoding
            element = 1

            # Iterate through interactions terms
            for j in range(len(interactions[i])):

                # interaction term and sites are shifted by 1
                m = interactions[i][j]-1

                element = element * encoding[sequences[n][m]]

            # add final element to matrix
            x[n][i] = element

    # Returns the dummy matrix
    return np.asarray(x)
