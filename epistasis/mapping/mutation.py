# Mapping Object for mutations-to-genotypes for epistasis maps
#
# Author: Zach Sailer
#
# ----------------------------------------------------------
# Local imports
# ----------------------------------------------------------

from epistasis.mapping.base import BaseMap

class MutationMap(BaseMap):
    """
        This object tracks the index and order of mutations in an epistatic map.
    """
    # ------------------------------------------------------------------
    # Getter methods for attributes that are not set explicitly by user.
    # ------------------------------------------------------------------
    
    @property
    def wildtype(self):
        """ Get possible that occur from reference system. """
        return self._wildtype
    
    @property
    def mutations(self):
        """ Get possible that occur from reference system. """
        return self._mutations
        
    @property
    def indices(self):
        """ Get the indices of mutations in the sequence. """
        return self._indices
        
    @property
    def n(self):
        """ Get the number of mutations in the space. """
        return self._n
    
    # ------------------------------------------------------------------
    # Setter methods for attributes that are not set explicitly by user.
    # ------------------------------------------------------------------
    
    @wildtype.setter
    def wildtype(self, wildtype):
        """ Set the wildtype genotype. """
        self._wildtype = wildtype
        
    @mutations.setter
    def mutations(self, mutations):
        """ Set the mutations from the genotypes. """
        self._mutations = mutations
        
    @indices.setter
    def indices(self, indices):
        """ Set the indices of where mutations occur in the wildtype genotype."""
        self._indices = indices
        
    @n.setter
    def n(self, n):
        """ Set the number of mutations. """
        self._n = n