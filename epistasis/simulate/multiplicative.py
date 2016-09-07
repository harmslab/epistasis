__doc__ = """Submodule with various classes for generating/simulating genotype-phenotype maps."""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np

# local imports
from epistasis.decomposition import generate_dv_matrix
from epistasis.simulate.base import BaseSimulation
from epistasis.mapping import EpistasisMap


from seqspace import utils
# ------------------------------------------------------------
# ArtificialMap object can be used to quickly generating a toy
# space for testing the EpistasisModels
# ------------------------------------------------------------

class MultiplicativeSimulation(BaseSimulation):
    """ Construct an genotype-phenotype from multiplicative building blocks and
    epistatic coefficients.

    Example
    -------
    Phenotype = b0 * b1 * b2 * b3 * b12 * b13 * b13 * b123
    or
    log(phenotype) = log(b0) + log(b1) + log(b2) + log(b3) + log(b12)+ log(b13) + log(b13) + log(b123)

    Arguments
    ---------
    wildtype : str
        Wildtype genotype
    mutations : dict
        Mapping for each site to its alphabet
    order : int
        Order of epistasis in simulated genotype-phenotype map
    model_type : str
        Use a local or global (i.e. Walsh space) epistasis model to construct
        phenotypes
    """

    def __init__(self, wildtype, mutations, order,
        coeff_range=(-1, 1),
        model_type='local'
        ):
        # Construct epistasis mapping objects (empty)
        super(MultiplicativeSimulation,self).__init__(
            wildtype,
            mutations,
            log_transform=True,
        )
        self.model_type = model_type
        self.epistasis = EpistasisMap(self)
        # Add values to epistatic interactions
        self.epistasis.order = order
        self.epistasis.values = self.base**np.random.uniform(coeff_range[0],
            coeff_range[1],
            size=len(self.epistasis.keys)
        )
        # build the phenotypes from the epistatic interactions
        self.build()

    def build(self):
        """ Build the phenotype map from epistatic interactions.

        For a multiplicative model, this means log-transforming the phenotypes
        first, using linear system of equations to construct the log-phenotypes,
        then back-transforming the phenotypes to non-log space.
        """
        # Get model type:
        if self.model_type == "local":
            encoding = {"1": 1, "0": 0}
            # Build phenotypes from binary representation of space
            self.X = generate_dv_matrix(self.binary.genotypes, self.epistasis.labels, encoding=encoding)
            log_phenotypes = np.dot(self.X, self.epistasis.log.values)
        elif self.model_type == "global":
            encoding = {"1": 1, "0": -1}
            # Build phenotypes from binary representation of space
            self.X = generate_dv_matrix(self.binary.genotypes, self.epistasis.labels, encoding=encoding)
            log_phenotypes = np.dot(self.X, self.epistasis.log.values)
        else:
            raise Exception("Invalid model type given.")
        # Unlog the phenotypes
        self.phenotypes = self.base**log_phenotypes
