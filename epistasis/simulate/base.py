# -------------------------------------------------
# Global imports
# -------------------------------------------------

import numpy as np

from seqspace.gpm import GenotypePhenotypeMap
from seqspace.utils import (enumerate_space,
    binary_mutations_map,
    encode_mutations,
    mutations_to_genotypes)


# -------------------------------------------------
# Local imports
# -------------------------------------------------

class BaseSimulation(GenotypePhenotypeMap):
    """ Base class for simulating genotype-phenotype maps built from epistatic
    interactions.

    Parameters
    ----------
    length : int
        Length of the genotypes in the map.
    order : int
        Order of epistasis in the genotype phenotype map
    """
    def __init__(self, wildtype, mutations,
        log_transform=False,
        logbase=np.log10,
        ):
        genotypes = np.array(mutations_to_genotypes(wildtype, mutations))
        phenotypes = np.ones(len(genotypes))
        # Initialize a genotype-phenotype map
        super(BaseSimulation, self).__init__(
            wildtype,
            genotypes,
            phenotypes,
            log_transform=log_transform,
            logbase=logbase,
            mutations=mutations
        )

    @classmethod
    def quick_start(cls, length, order, **kwargs):
        """Constructs genotype from binary sequences with given length and
        phenotypes from epistasis with a given order.

        Parameters
        ----------
        length : int
            length of the genotypes
        order : int
            order of epistasis in phenotypes.

        Returns
        -------
        Simulation object
        """
        wildtype = "0"*length
        mutations = binary_mutations_map(wildtype, "1"*length)
        return cls(wildtype, mutations, order, **kwargs)

    @classmethod
    def from_epistasis(cls, wildtype, mutations, order, betas, model_type="local"):
        """Add genotypic epistasis to genotype-phenotype map.
        """
        space = cls(wildtype, mutations, order, model_type=model_type)
        if len(betas) != space.epistasis.n:
            raise Exception("""Number of betas does not match order/mutations given.""")
        space.epistasis.values = betas
        space.build()
        return space

    def build(self, values=None, **kwargs):
        """ Method for construction phenotypes from model. """
        raise Exception( """ Must be implemented in subclass. """)

    def set_stdeviations(self, sigma):
        """Add standard deviations to the simulated phenotypes, which can then be
        used for sampling error in the genotype-phenotype map.

        Parameters
        ----------
        sigma : float or array-like
            Adds standard deviations to the phenotypes. If float, all phenotypes
            are given the same stdeviations. Else, array must be same length as
            phenotypes and will be assigned to each phenotype.
        """
        stdeviations = np.ones(len(self.phenotypes)) * sigma
        self.stdeviations = stdeviations
