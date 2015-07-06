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
        """ Set the mutation alphabet for all sites in wildtype genotype. 
         
            `mutations = { site_number : alphabet }`. If the site 
            alphabet is note included, the model will assume binary 
            between wildtype and derived.

            ``` 
            mutations = {
                0: [alphabet],
                1: [alphabet],

            }
            ```
        
        """
        if type(mutations) != dict:
            raise TypeError("mutations must be a dict")
        self._mutations = mutations
        self.n = len(mutations)
        
    @n.setter
    def n(self, n):
        """ Set the number of mutations. """
        self._n = n