from epistasis.mapping.base import BaseMap

class MutationMap(BaseMap):
    
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