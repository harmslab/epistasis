import numpy as np

# Try importing model matrix builder from cython extension for speed up.
try:
    from .matrix_cython import build_model_matrix

except ImportError:

    import warnings as _warnings

    # Raise warning
    _warnings.warn('Could not load cython extension, "build_model_matrix".', Warning)

    def build_model_matrix(encoding_vectors, sites):
        """Build model matrix.

        This can be made faster by cython.
        """
        # get dimensions of matrix
        n, m = len(encoding_vectors), len(sites)

        matrix = np.ones((n,m), dtype=int)

        # Interate over rows
        for i in range(n):
            vec = encoding_vectors[i]

            # Iterate over cols
            for j in range(m):
                # Get sites in this coefficient column
                players = sites[j]

                # Multiply these elements
                matrix[i,j] = np.prod(vec[players])

        return matrix


def encode_vectors(binary_genotypes, model_type='global'):
    """Encode a set of binary genotypes is input vectors for the given model

    """
    # Initialize vector container
    vectors = []

    # Handle a global model
    if model_type == 'global':

        for i, genotype in enumerate(binary_genotypes):
            vector = np.array([0] + list(genotype), dtype=float)
            vector[vector==1] = -1
            vector[vector==0] = 1
            vectors.append(vector)

    # Handle a local model.
    elif model_type == 'local':

        for i, genotype in enumerate(binary_genotypes):
            vector = np.array([1] + list(genotype), dtype=float)
            vectors.append(vector)

    # Don't understand the model
    else:
        Exception("Unrecognized model type.")

    return np.array(vectors)


def get_model_matrix(binary_genotypes, sites, model_type='global'):
    """Get a model matrix for a given set of genotypes and coefficients.
    """
    # Convert sites to array of arrays
    sites = np.array([np.array(s) for s in sites])

    # Encode genotypes
    encoded_vector = encode_vectors(binary_genotypes, model_type=model_type)

    # Build matrix.
    X = build_model_matrix(encoded_vector, sites)
    return X
