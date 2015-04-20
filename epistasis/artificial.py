# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
from regression_ext import generate_dv_matrix
from .core.utils import generate_binary_space
from .core.mapping import EpistasisMap

# ------------------------------------------------------------
# ArtificialMap object can be used to quickly generating a toy
# space for testing the EpistasisModels
# ------------------------------------------------------------

class ArtificialMap(EpistasisMap):
    
    def __init__(self, length, order, log_transform=False):
        """ Generate a binary genotype-phenotype mape with the given length from epistatic interactions. """
        wildtype = 'A'*length
        mutant = 'T'*length
        self.genotypes = generate_binary_space(wildtype, mutant)
        self.wildtype = wildtype
        self.order = order
        self.log_transform = log_transform
        self._random_epistasis()
        self._build_phenotypes()

    def _random_epistasis(self):
        """Assign random values to epistatic terms. """ 
        vals = (-1)**(np.random.randint(1,10, size=len(self.interaction_labels))) * np.random.rand(len(self.interaction_labels))
        self.interaction_values = vals
        
    def _build_phenotypes(self):
        """ Build the phenotype map from epistatic interactions. """
        # Allocate phenotype numpy array
        phenotypes = np.zeros(self.n, dtype=float)
        
        # Build phenotypes for binary representation of space
        self.X = generate_dv_matrix(self.bits, self.interaction_labels)
        bit_phenotypes = np.dot(self.X,self.interaction_values)
    
        # Reorder phenotypes to map back to genotypes
        for i in range(len(self.bit_indices)):
            phenotypes[self.bit_indices[i]] = bit_phenotypes[i]
        self.phenotypes = phenotypes
        
    def model_input(self):
        """ Get input for a generic Epistasis Model.
        
            Returns:
            -------
            wildtype: str
                wildtype sequence for reference when calculating epistasis
            genotypes: list of str
                List of all genotypes in space.
            phenotypes: array of floats
                Array of phenotype map.
            order: int
                Order of epistasis in epistasis map.
        """
        return self.wildtype, self.genotypes, self.phenotypes, self.order