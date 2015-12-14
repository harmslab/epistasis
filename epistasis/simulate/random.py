__doc__ = """Submodule with various classes for generating/simulating genotype-phenotype maps."""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np

# local imports
from epistasis.decomposition import generate_dv_matrix
from epistasis.simulate.base import BaseArtificialMap

def hadamard_weight_vector(genotypes):
    """ Build the hadamard weigth vector. """
    l = len(genotypes)
    n = len(genotypes[0])
    weights = np.zeros((l, l), dtype=float)
    for g in range(l):
        epistasis = float(genotypes[g].count("1"))
        weights[g][g] = ((-1)**epistasis)/(2**(n-epistasis))
    return weights

# ------------------------------------------------------------
# ArtificialMap object can be used to quickly generating a toy
# space for testing the EpistasisModels
# ------------------------------------------------------------


class RandomEpistasisMap(BaseArtificialMap):

    """ Generate genotype-phenotype map from random epistatic interactions. """

    def __init__(self, length, order, magnitude, log_transform=False, model='local', allow_neg=True):
        """ Choose random values for epistatic terms below and construct a genotype-phenotype map.

            ASSUMES ADDITIVE MODEL (UNLESS LOG TRANSFORMED).

            __Arguments__:

            `length` [int] : length of strings

            `order` [int] : order of epistasis in space

            `magnitude` [float] : maximum value of abs(epistatic) term.

            `log_transform` [bool] : return the log_transformed phenotypes.

        """
        super(RandomEpistasisMap,self).__init__(length, order, log_transform)
        
        high = magnitude
        low = 0
        if allow_neg:
            low = -magnitude 
        
        self.Interactions.values = self.random_epistasis(low, high)
        self.model = model
        self.build_phenotypes()

    def build_phenotypes(self, values=None):
        """ Build the phenotype map from epistatic interactions. """
        # Allocate phenotype numpy array
        _phenotypes = np.zeros(self.n, dtype=float)

        # Check for custom values
        if values is None:
            values = self.Interactions.values

        # Get model type:
        if self.model == "local":
            encoding = {"1": 1, "0": 0}
            
            # Build phenotypes from binary representation of space
            self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding=encoding)
            self.Binary.phenotypes = np.dot(self.X,values)
            
        elif self.model == "global":
            encoding = {"1": -1, "0": 1}
            #self.weight_matrix = np.diag(np.diag(np.linalg.inv(hadamard_weight_vector(self.Binary.genotypes))))

            # Build phenotypes from binary representation of space
            self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding=encoding)
            self.Binary.phenotypes = np.dot( self.X, values)
            #self.Binary.phenotypes = np.dot( np.dot( self.weight_matrix, self.X) , values)
        
        else:
            raise Exception("Invalid model type given.")

        # Reorder phenotypes to map back to genotypes
        for i in range(len(self.Binary.indices)):
            _phenotypes[self.Binary.indices[i]] = self.Binary.phenotypes[i]

        self.phenotypes = _phenotypes

class ZeroTermsEpistasisMap(RandomEpistasisMap):

    """ Generate genotype-phenotype map with parameters set to zero. """

    def __init__(self, length, magnitude, zero_params, log_transform=False):
        """ Generate a genotype phenotype map with parameters set to zero.

            __Arguments__:

            `zero_params`  [list of ints] : indices of parameters to set to zero.
        """
        super(ZeroTermsEpistasisMap, self).__init__(length, length, magnitude, log_transform=False)

        if np.amax(zero_params) >= len(self.Interactions.values):
            raise Exception("indices in zero_params are more than number of interactions. ")

        interactions = self.Interactions.values

        for i in zero_params:
            interactions[i] = 0.0

        self.Interactions.values = interactions
        self.phenotypes = self.build_phenotypes()
