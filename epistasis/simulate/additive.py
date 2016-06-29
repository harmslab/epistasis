__doc__ = """Submodule with various classes for generating/simulating genotype-phenotype maps."""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np

# local imports
from epistasis.decomposition import generate_dv_matrix
from epistasis.simulate.base import BaseSimulation
from epistasis.mapping.epistasis import EpistasisMap

# ------------------------------------------------------------
# ArtificialMap object can be used to quickly generating a toy
# space for testing the EpistasisModels
# ------------------------------------------------------------

class AdditiveSimulation(EpistasisMap, BaseSimulation):
    """ Generate genotype-phenotype map from random epistatic interactions.

    """
    def __init__(self, wildtype, mutations, order,
            magnitude=1.0,
            distribution=None,
            model_type='local',
            neg_coeffs=True
        ):
        # Construct epistasis mapping objects (empty)
        super(AdditiveSimulation,self).__init__(wildtype, genotypes, phenotypes)

        # Set order, which builds interaction mapping.
        self.order = order

        # Add values to epistatic interactions
        self.Interactions.values = np.random.uniform(low, high, size=len(self.Interactions.keys))
        self.model_type = model_type

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
            self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding=encoding)
            self.Binary.phenotypes = np.dot(self.X,values)

        elif self.model_type == "global":
            encoding = {"1": -1, "0": 1}

            # Build phenotypes from binary representation of space
            self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding=encoding)
            self.Binary.phenotypes = np.dot( self.X, values)

        else:
            raise Exception("Invalid model type given.")

        # Reorder phenotypes to map back to genotypes
        for i in range(len(self.Binary.indices)):
            _phenotypes[self.Binary.indices[i]] = self.Binary.phenotypes[i]

        self.phenotypes = _phenotypes
