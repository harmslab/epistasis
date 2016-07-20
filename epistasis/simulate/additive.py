__doc__ = """Submodule with various classes for generating/simulating genotype-phenotype maps."""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np

from seqspace.gpm import GenotypePhenotypeMap

# local imports
from epistasis.decomposition import generate_dv_matrix
from epistasis.simulate.base import BaseSimulation

# ------------------------------------------------------------
# ArtificialMap object can be used to quickly generating a toy
# space for testing the EpistasisModels
# ------------------------------------------------------------

class AdditiveSimulation(BaseSimulation):
    """Construct an genotype-phenotype from additive building blocks and
    epistatic coefficients.

    Example
    -------
    Phenotype = b0 + b1 + b2 + b3 + b12 + b13 + b13 + b123

    Parameters
    ---------
    wildtype : str
        Wildtype genotype
    mutations : dict
        Mapping for each site to its alphabet
    order : int
        Order of epistasis in simulated genotype-phenotype map
    betas : array-like
        values of epistatic coefficients (must be positive for this function
        to work. Log is taken)
    model_type : str
        Use a local or global (i.e. Walsh space) epistasis model to construct
        phenotypes
    """
    def __init__(self, wildtype, mutations, order,
        coeff_range=(-1, 1),
        model_type='local',
        ):
        # Construct epistasis mapping objects (empty)
        super(AdditiveSimulation,self).__init__(
            wildtype,
            mutations,
        )
        self.model_type = model_type
        # Add values to epistatic interactions
        self.epistasis.order = order
        self.epistasis.values = np.random.uniform(coeff_range[0], coeff_range[1], size=len(self.epistasis.keys))
        # build the phenotypes from the epistatic interactions
        self.build()

    def build(self):
        """ Build the phenotype map from epistatic interactions. """
        # Allocate phenotype numpy array
        _phenotypes = np.zeros(self.n, dtype=float)
        # Get model type:
        if self.model_type == "local":
            encoding = {"1": 1, "0": 0}
            # Build phenotypes from binary representation of space
            self.X = generate_dv_matrix(self.binary.genotypes, self.epistasis.labels, encoding=encoding)
            self.phenotypes = np.dot(self.X, self.epistasis.values)
        elif self.model_type == "global":
            encoding = {"1": -1, "0": 1}
            # Build phenotypes from binary representation of space
            self.X = generate_dv_matrix(self.binary.genotypes, self.epistasis.labels, encoding=encoding)
            self.phenotypes = np.dot( self.X, self.epistasis.values)
        else:
            raise Exception("Invalid model type given.")
