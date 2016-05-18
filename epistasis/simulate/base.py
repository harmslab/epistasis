__doc__ = """
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


def construct(self, length, order, log_transform=False, mutations=None):

    if mutations is None:

        wildtype = '0'*length
        mutant = '1'*length
        mutations = binary_mutations_map(wildtype, mutant)
        genotypes, binaries = enumerate_space(wildtype, mutant)
        phenotypes = np.zeros(len(genotypes), dtype=float)

    else:
        pass

    # Initialize base map.
    super(BaseSimulation, self).__init__(wildtype, genotypes, phenotypes, log_transform=log_transform, mutations=mutations)
    self.order = order
    self.log_transform = log_transform
    self.stdevs = None

class BaseSimulation(object):
    """ Base class for simulating genotype-phenotype maps built from epistatic
    interactions.

    Parameters
    ----------
    length : int
        Length of the genotypes in the map.
    order : int
        Order of epistasis in the genotype phenotype map
    """

    def build(self, values=None, **kwargs):
        """ Method for construction phenotypes from model. """
        raise Exception( """ Must be implemented in subclass. """)

    def noise(self, func=np.random.normal, loc=0.0, scale=1.0):
        """ Simulate noise in the experimental measurement.

        Parameters
        ----------
        func : callable
            Callable function that returns samples from an error distribution
            shape. Default distribution is normal.
        """
        self._error_distribution = func
        self._error_mean = loc
        self._error_scale = scale

    def sample(self, ):
        """ """
        pass
