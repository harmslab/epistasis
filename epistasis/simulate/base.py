__doc__ = """
Base class for epistasis models.
"""
# -------------------------------------------------
# Global imports
# -------------------------------------------------

import numpy as np

from seqspace.utils import enumerate_space, binary_mutations_map

# -------------------------------------------------
# Local imports
# -------------------------------------------------

from epistasis.mapping.epistasis import EpistasisMap

class BaseArtificialMap(EpistasisMap):

    """ Base class for generating genotype-phenotype maps from epistasis interactions."""

    def __init__(self, length, order, log_transform=False):
        """ Generate a binary genotype-phenotype mape with the given length from epistatic interactions.

            __Arguments__:

            `length` [int] : length of sequences.

            `order` [int] : order of epistasis in model.

            `log_transform` [bool] : log transform the phenotypes if true.
        """
        wildtype = '0'*length
        mutant = '1'*length
        mutations = binary_mutations_map(wildtype, mutant)
        genotypes, binaries = enumerate_space(wildtype, mutant)
        phenotypes = np.zeros(len(genotypes), dtype=float)

        # Initialize base map.
        super(BaseArtificialMap, self).__init__(wildtype, genotypes, phenotypes, log_transform=log_transform, mutations=mutations)
        self.order = order
        self.log_transform = log_transform
        self.stdevs = None
        self._construct_binary()
        self._construct_interactions()

    def build(self, values=None, **kwargs):
        """ Method for construction phenotypes from model. """
        raise Exception( """ Must be implemented in subclass. """)

    def random_epistasis(self, low, high, allow_neg=True):
        """ Generate random values for epistatic interactions (with equal magnitude sampling).

            __Arguments__:

            `low` [int] : minimum values to select from uniform random distribution

            `high` [int] : maximum values to select from uniform random distribution

            `allow_neg` [bool] : If False, all random values will be positive.

            __Returns__:

            `vals` [array] : array of random values with length == number of Interactions.

        """

        # if negatives are not allowed, compute set lower limit to zero.
        if allow_neg is False:
            low = 0

        vals = (high-low) * np.random.random(size=len(self.Interactions.labels)) + low
        return vals

    def rm_epistasis(self, n_terms):
        """ Remove a specified number of epistatic terms. Choose these terms randomly. """
        indices = np.random.randint(len(self.Interactions.labels), size=n_terms)

        for i in indices:
            self.Interactions._values[i] = 0.0

        self.build()

    def add_noise(self, percent):
        """ Add noise to the phenotypes. """
        noise = np.empty(self.n, dtype=float)
        for i in range(self.n):
            noise[i] = percent*self.phenotypes[i]
        self.stdevs = noise


    def model_input(self):
        """ Get input for a generic Epistasis Model.

            __Returns__:

            `wildtype` [str] :  wildtype sequence for reference when calculating epistasis

            `genotypes` [list of str] : List of all genotypes in space.

            `phenotypes` [array of floats] : Array of phenotype map.

            `order` [int]: Order of epistasis in epistasis map.
        """
        return self.wildtype, self.genotypes, self.phenotypes, self.order
