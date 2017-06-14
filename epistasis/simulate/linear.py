sites__doc__ = """Submodule with various classes for generating/simulating genotype-phenotype maps."""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
from gpmap.gpm import GenotypePhenotypeMap

# local imports
from epistasis.model_matrix_ext import get_model_matrix
from epistasis.simulate.base import BaseSimulation

# ------------------------------------------------------------
# ArtificialMap object can be used to quickly generating a toy
# space for testing the EpistasisModels
# ------------------------------------------------------------

class LinearSimulation(BaseSimulation):
    """Construct an genotype-phenotype from linear building blocks and
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
    def __init__(self, wildtype, mutations,
        model_type='global',
        **kwargs
        ):
        # Construct epistasis mapping objects (empty)
        super(LinearSimulation,self).__init__(
            wildtype,
            mutations,
            **kwargs
        )
        self.model_type = model_type

    @property
    def p_additive(self):
        """Get the additive phenotypes"""
        orders = self.epistasis.getorder
        sites = list(orders[0].sites) + list(orders[1].sites)
        vals = list(orders[0].values) + list(orders[1].values)
        x = get_model_matrix(self.binary.genotypes, sites, model_type=self.model_type)
        return np.dot(x, vals)

    def build(self):
        """ Build the phenotype map from epistatic interactions. """
        # Allocate phenotype numpy array
        _phenotypes = np.zeros(self.n, dtype=float)
        # Get model type:
        self.X = get_model_matrix(self.binary.genotypes, self.epistasis.sites, model_type=self.model_type)
        self.phenotypes = np.dot( self.X, self.epistasis.values)
