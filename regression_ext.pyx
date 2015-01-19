import numpy as np

def generate_dv_matrix(sequences, indices):
    """ Build the X matrix of dummy variable for linear regression
        in epistasis model.
    """
    cdef int i, j, k, m, n, l
    dim1 = len(sequences)
    dim2 = len(indices)
    
    x = np.ones((dim1,dim2), dtype=int)
    
    for n in range(dim1):
        for i in range(1,dim2):
            for j in range(len(indices[i])):
                m = indices[i][j]-1
                if k == 0:
                    pass
                elif sequences[n][m] == "1":
                    x[n][i] = 1
                else:
                    x[n][i] = 0
                    k = 0
            k = 1
    return np.asarray(x)