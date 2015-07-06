# Mapping object for tracking a binary representation of the epistasis map.
#
# Author: Zach Sailer
#
# ----------------------------------------------------------
# Local imports
# ----------------------------------------------------------

from epistasis.mapping.base import BaseMap

class BinaryMap(BaseMap):
    """
        Map for holding a binary representation of an epistasis map.
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
        return self._phenotypes
        
    @property
    def mutations(self):
        """ Return a binary representation of each site-mutation in the genotype-phenotype map"""
        return self._mutations
         
    @property
    def errors(self):
        """ Get the phenotype values in an array orderd same as binary reprentation."""
        return self._errors

    # ----------------------------------------------------------
    # Getter methods for mapping objects
    # ----------------------------------------------------------
    
    @property
    def geno2pheno(self):
        """ Return dict of genotypes mapped to phenotypes. """
        return self._map(self.genotypes, self.phenotypes)
        
    # ----------------------------------------------------------
    # Setter methods
    # ----------------------------------------------------------
    
    @genotypes.setter
    def genotypes(self, genotypes):
        """ Set Binary representation of genotypes. """
        self._genotypes = genotypes
        
    @indices.setter
    def indices(self, indices):
        """ Set indices of genotypes in self.genotypes that mapped to their binary representation. """
        self._indices = indices

    @mutations.setter
    def mutations(self, mutations):
        """ Set the mapping for site-to-mutation-to-binary-representation."""
        self._mutations = mutations

    @phenotypes.setter
    def phenotypes(self, phenotypes):
        """ Set the phenotype values in an array orderd same as binary reprentation."""
        self._phenotypes = phenotypes
        
    @errors.setter
    def errors(self, errors):
        """ Set the phenotype values in an array orderd same as binary reprentation."""
        self._errors = errors