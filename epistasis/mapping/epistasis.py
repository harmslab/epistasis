# Main mapping object to be used the epistasis models in this package.
#
# Author: Zach Sailer
#
# ----------------------------------------------------------
# Outside imports
# ----------------------------------------------------------

import numpy as np

# ----------------------------------------------------------
# Core internal mapping for this package is inheritted from
# seqspace module.
# ----------------------------------------------------------

from seqspace.gpm import GenotypePhenotypeMap
from seqspace.utils import hamming_distance, encode_mutations, construct_genotypes

# ----------------------------------------------------------
# Local imports
# ----------------------------------------------------------

from epistasis.mapping.interaction import InteractionMap
from epistasis.utils import params_index_map, build_model_params

# ----------------------------------------------------------
# Internal mapping object for all models in this package
# Notice: mapping inherits seqspace base mapping object
# ----------------------------------------------------------

class EpistasisMap(GenotypePhenotypeMap):

    def __init__(self, wildtype, genotypes, phenotypes, errors=None, log_transform=False, mutations=None, n_replicates=1):
        """
            Object that maps epistasis in a genotype-phenotype map.

            __Attributes__:

            `length` [int] : length of genotypes

            `n` [int] : size of genotype-phenotype map

            `order` [int] : order of epistasis in system

            `wildtype` [str] : wildtype genotype

            `mutations` [array of chars] : individual mutations from wildtype that are in the system

            `genotypes` [array] : genotype in system

            `phenotypes` [array] : quantitative phenotypes in system

            `errors` [array] : errors for each phenotype value

            `indices` [array] : genotype indices
        """
        super(EpistasisMap, self).__init__(wildtype, genotypes, phenotypes, errors=errors, log_transform=log_transform, mutations=mutations, n_replicates=n_replicates)

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
        if hasattr(self, "_order"):
            self.Interactions.order = self.order
            self.Interactions.labels = build_model_params(self.Interactions.length,
                                                      self.Interactions.order,
                                                      self.Interactions.mutations)
