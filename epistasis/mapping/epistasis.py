# Main mapping object to be used the epistasis models in this package.
#
# Author: Zach Sailer
#
# ----------------------------------------------------------
# Outside imports
# ----------------------------------------------------------

import numpy as np
from seqspace.gpm import GenoPhenoMap
from seqspace.utils import hamming_distance, encode_mutations, construct_genotypes

# ----------------------------------------------------------
# Local imports
# ----------------------------------------------------------

from epistasis.mapping.interaction import InteractionMap
from epistasis.utils import params_index_map, build_model_params

class EpistasisMap(GenoPhenoMap):
    
    def __init__(self, wildtype, genotypes, phenotypes, errors=None, log_transform=False, mutations=None):
        """
            Object that maps epistasis in a genotype-phenotype map. 
        
            Attributes:
            ----------
            length: int, 
                length of genotypes
            n: int
                size of genotype-phenotype map
            order: int
                order of epistasis in system
            wildtype: str
                wildtype genotype
            mutations: array of chars
                individual mutations from wildtype that are in the system
            genotypes: array
                genotype in system
            phenotypes: array
                quantitative phenotypes in system
            errors: array
                errors for each phenotype value
            indices: array
                genotype indices
        """
        super(EpistasisMap, self).__init__(wildtype, genotypes, phenotypes, errors=errors, log_transform=log_transform, mutations=mutations)
        
    # ------------------------------------------------------
    # Getter methods for attributes that can be set by user.
    # ------------------------------------------------------

    @property
    def order(self):
        """ Get order of epistasis in system. """
        return self._order
        
    # ------------------------------------------------------
    # Getter methods for attributes that can be set by user.
    # ------------------------------------------------------
    
    @order.setter
    def order(self, order):
        """ Set the order of epistasis in the system. As a consequence, 
            this mapping object creates the """
        self._order = order

    def _construct_interactions(self):
        """ Construct the interactions mapping for an epistasis model. 
            
            Must populate the Mutations subclass before setting interactions. 
        """
        self.Interactions = InteractionMap(self.Mutations)
        self.Interactions._length = self.length
        self.Interactions.log_transform = self.log_transform
        self.Interactions.mutations = params_index_map(self.mutations) # construct the mutations mapping
        
        # If an order is specified, construct epistatic interaction terms.
        #try:
        self.Interactions.order = self.order
        self.Interactions.labels = build_model_params(self.Interactions.length, 
                                                      self.Interactions.order, 
                                                      self.Interactions.mutations)
        #except:
         #   pass
