__doc__ = """Submodule with various classes for generating/simulating genotype-phenotype maps."""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np

# local imports
from epistasis.decomposition import generate_dv_matrix
from epistasis.simulate.base import BaseArtificialMap

# ------------------------------------------------------------
# ArtificialMap object can be used to quickly generating a toy
# space for testing the EpistasisModels
# ------------------------------------------------------------


class RandomEpistasisMap(BaseArtificialMap):

    """ Generate genotype-phenotype map from random epistatic interactions. """

    def __init__(self, length, order, magnitude, model='local', log_transform=False):
        """ Choose random values for epistatic terms below and construct a genotype-phenotype map.

            ASSUMES ADDITIVE MODEL (UNLESS LOG TRANSFORMED).

            __Arguments__:

            `length` [int] : length of strings

            `order` [int] : order of epistasis in space

            `magnitude` [float] : maximum value of abs(epistatic) term.

            `log_transform` [bool] : return the log_transformed phenotypes.

        """
        super(RandomEpistasisMap,self).__init__(length, order, log_transform)
        self.Interactions.values = self.random_epistasis(-1,1)
        self.phenotypes = self.build_phenotypes(model=model)

    def build_phenotypes(self, values=None, module_type='local'):
        """ Build the phenotype map from epistatic interactions. """
        # Allocate phenotype numpy array
        phenotypes = np.zeros(self.n, dtype=float)

        # Check for custom values
        if values is None:
            values = self.Interactions.values

        # Get model type:
        if model == "local":
            encoding = {"1": 1, "0": 0}
        elif model == "global":
            encoding = {"1": 1, "0": -1}
        else:
            raise Exception("Invalid model type given.")

        # Build phenotypes from binary representation of space
        self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding=encoding)
        self.Binary.phenotypes = bit_phenotypes = np.dot(self.X,values)

        # Handle log_transformed phenotypes
        if self.log_transform:
            self.Binary.phenotypes = 10**bit_phenotypes

        # Reorder phenotypes to map back to genotypes
        for i in range(len(self.Binary.indices)):
            phenotypes[self.Binary.indices[i]] = self.Binary.phenotypes[i]

        return phenotypes

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
