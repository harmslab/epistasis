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

class Sample:

    def __init__(self, replicate_genotypes, replicate_phenotypes, indices=None):
        """ Sample from simulated experiment """
        self.replicate_genotypes = replicate_genotypes
        self.replicate_phenotypes = replicate_phenotypes
        self.genotypes = self.replicate_genotypes[:,0]
        self.phenotypes = np.mean(self.replicate_phenotypes, axis=1)
        self.stdevs = np.std(self.replicate_phenotypes, ddof=1, axis=1)
        self.indices = indices


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

    def build_phenotypes(self, values=None, **kwargs):
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
        
        self.build_phenotypes()

    def add_noise(self, percent):
        """ Add noise to the phenotypes. """
        noise = np.empty(self.n, dtype=float)
        for i in range(self.n):
            noise[i] = percent*self.phenotypes[i]
        self.stdevs = noise

    def sample(self, n_samples=1, fraction=1.0):
        """ Generate artificial data sampled from phenotype and percent error.

            __Arguments__:

            `n_samples` [int] : Number of samples to take from space

            `fraction` [float] : fraction of space to sample.

            __Return__:

            `samples` [Sample object]: returns this object with all stats on experiment
        """
        # make sure fraction is float between 0 and 1
        if fraction < 0 or fraction > 1:
            raise Exception("fraction is invalid.")

        # fractional length of space.
        frac_length = int(fraction * self.n)

        # random genotypes and phenotypes to sample
        random_indices = np.sort(np.random.choice(range(self.n), size=frac_length, replace=False))

        # initialize arrays
        phenotypes = np.empty((frac_length, n_samples), dtype=float)
        genotypes = np.empty((frac_length, n_samples), dtype='<U'+str(self.length))

        # If errors are present, sample from error distribution
        try:
            stdevs = self.stdevs
            for i in random_indices:
                seq = self.genotypes[i]
                genotypes[i] = np.array([seq for j in range(n_samples)])
                phenotypes[i] = stdevs[i] * np.random.randn(n_samples) + self.phenotypes[i]
        except:
            # Can't sample if no error distribution is given.
            if n_samples != 1:
                raise Exception("Won't create samples if sample error is not given.")

            genotypes = np.array([self.genotypes[i] for i in random_indices])
            phenotypes = np.array([self.phenotypes[i] for i in random_indices])

        samples = Sample(genotypes, phenotypes, random_indices)
        return samples

    def model_input(self):
        """ Get input for a generic Epistasis Model.

            __Returns__:

            `wildtype` [str] :  wildtype sequence for reference when calculating epistasis

            `genotypes` [list of str] : List of all genotypes in space.

            `phenotypes` [array of floats] : Array of phenotype map.

            `order` [int]: Order of epistasis in epistasis map.
        """
        return self.wildtype, self.genotypes, self.phenotypes, self.order
