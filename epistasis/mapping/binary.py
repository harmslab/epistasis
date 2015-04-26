from epistasis.mapping.base import BaseMap

class BinaryMap(BaseMap):
    
    """
        Map for holding a binary representation of the epistasis map.
    """
    
    @property
    def genotypes(self):
        """ Get Binary representation of genotypes. """
        return self._genotypes
        
    @property
    def indices(self):
        """ Get indices of genotypes in self.genotypes that mapped to their binary representation. """
        return self._indices
        
    @property
    def phenotypes(self):
        """ Get the phenotype values in an array orderd same as binary reprentation."""
        return self.phenotypes
        
    @property
    def phenotype_errors(self):
        """ Get the phenotype values in an array orderd same as binary reprentation. 
        
            BROKEN --- NEEDS SOME WORK
        
        """
        if self.log_transform is True:
            return np.array((self.phenotype_errors[0,self.bit_indices],self.phenotype_errors[1,self.bit_indices]))
        else:
            return self.phenotype_errors[self.bit_indices]
            
    @property
    def geno2pheno(self):
        """ Return dict of genotypes mapped to phenotypes. """
        return self._map(self.genotypes, self.phenotypes)