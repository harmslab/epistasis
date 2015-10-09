__doc__ = """
Base class for epistasis models.
"""
# -------------------------------------------------
# Global imports
# -------------------------------------------------

import numpy

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
        self._construct_binary()
        self._construct_interactions()

    def build_phenotypes(self, values=None):
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

    def random_knockout(self, n_knockouts):
        """ Set parameters"""
        indices = np.random.randint(len(self.Interactions.labels), size=n_knockouts)
        for i in indices:
            self.Interactions._values[i] = 0.0

    def add_noise(self, percent):
        """ Add noise to the phenotypes. """
        noise = np.empty(self.n, dtype=float)
        for i in range(self.n):
            noise[i] = percent*self.phenotypes[i]
        self.errors = noise

    def create_samples(self, n_samples):
        """ Generate artificial data sampled from phenotype and percent error. """
        try:
            errors = self.errors
        except:
            self.add_noise(0.05)
            errors = self.errors

        gen_phenotypes = np.empty((self.n, n_samples), dtype=float)
        gen_genotypes = np.empty((self.n, n_samples), dtype='<U'+str(self.length))

        for s in range(len(self.genotypes)):
            seq = self.genotypes[s]
            gen_genotypes[s] = np.array([seq for i in range(n_samples)])
            gen_phenotypes[s] = errors[s] * np.random.randn(n_samples) + self.phenotypes[s]

        return gen_genotypes, gen_phenotypes

    def model_input(self):
        """ Get input for a generic Epistasis Model.

            __Returns__:



            `wildtype` [str] :  wildtype sequence for reference when calculating epistasis

            `genotypes` [list of str] : List of all genotypes in space.

            `phenotypes` [array of floats] : Array of phenotype map.

            `order` [int]: Order of epistasis in epistasis map.
        """
        return self.wildtype, self.genotypes, self.phenotypes, self.order
