# ---------------------------------------------------
# Decomposition matrix for epistasis models
# ---------------------------------------------------

import numpy as np

def get_model_matrix(sequences, interactions, model_type="local"):
    """ Build the X matrix of dummy variable for linear regression
        in epistasis model.
    """
    # Determine matrix type
    model_options = {"local":{"0": 0, "1": 1}, "global":{"0": -1, "1": 1}}
    encoding = model_options[model_type]

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
    X = np.asarray(x)

    #if model_type == "global":
    #    order = len(sequences[0])
    #    vii = np.array([-1**len(label) / 2**(order - len(label)) for label in interactions])
    #    W = np.eye(len(vii))
    #    W[range(len(vii)), range(len(vii))] = vii
    #    X = np.dot(W, X)

    return X
