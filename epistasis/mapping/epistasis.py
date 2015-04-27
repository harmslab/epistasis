# Main mapping object to be used the epistasis models in this package.
#
# Author: Zach Sailer
#
# ----------------------------------------------------------
# Outside imports
# ----------------------------------------------------------

import numpy as np

# ----------------------------------------------------------
# Local imports
# ----------------------------------------------------------

from epistasis.mapping.base import BaseMap
from epistasis.mapping.binary import BinaryMap
from epistasis.mapping.interaction import InteractionMap
from epistasis.utils import hamming_distance, find_differences, enumerate_space

class EpistasisMap(BaseMap):
    
    def __init__(self):
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
            phenotype_errors: array
                errors for each phenotype value
            indices: array
                genotype indices
        """
        self.Interactions = InteractionMap()
        self.Binary = BinaryMap()
        
    # ------------------------------------------------------
    # Getter methods for attributes that can be set by user.
    # ------------------------------------------------------
    
    @property
    def length(self):
        """ Get length of the genotypes. """
        return self._length    
    
    @property
    def n(self):
        """ Get number of genotypes, i.e. size of the system. """
        return self._n

    @property
    def order(self):
        """ Get order of epistasis in system. """
        return self._order
    
    @property
    def log_transform(self):
        """ Boolean argument telling whether space is log transformed. """
        return self._log_transform
        
    @property
    def wildtype(self):
        """ Get reference genotypes for interactions. """
        return self._wildtype
        
    @property
    def mutant(self):
        """ Get the furthest genotype from the wildtype genotype. """
        return self._mutant

    @property
    def genotypes(self):
        """ Get the genotypes of the system. """
        return self._genotypes
        
    @property
    def phenotypes(self):
        """ Get the phenotypes of the system. """
        return self._phenotypes
    
    @property
    def errors(self):
        """ Get the phenotypes' errors in the system. """
        return self._errors

    @property    
    def indices(self):
        """ Return numpy array of genotypes position. """
        return self._indices
        
    # ----------------------------------------------------------
    # Getter methods for mapping objects
    # ----------------------------------------------------------   
    
    @property
    def geno2pheno(self):
        """ Return dict of genotypes mapped to phenotypes. """
        return self._map(self.genotypes, self.phenotypes)

    @property
    def geno2index(self):
        """ Return dict of genotypes mapped to their indices in transition matrix. """
        return self._map(self.genotypes, self.indices)
        
    @property
    def geno2binary(self):
        """ Return dictionary of genotypes mapped to their binary representation. """
        mapping = dict()
        for i in range(self.n):
            mapping[self.genotypes[self.Binary.indices[i]]] = self.Binary.genotypes[i] 
        return mapping
        
    # ----------------------------------------------------------
    # Setter methods
    # ----------------------------------------------------------
    
    @log_transform.setter
    def log_transform(self, boolean):
        """ True/False to log transform the space. """
        self._log_transform = boolean
        self.Interactions.log_transform = boolean
    
    @genotypes.setter
    def genotypes(self, genotypes):
        """ Set genotypes from ordered list of sequences. """
        self._n = len(genotypes)
        self._length = len(genotypes[0])
        self._genotypes = np.array(genotypes)
        self._indices = np.arange(self._n)
        
    @wildtype.setter
    def wildtype(self, wildtype):
        """ Set the reference genotype among the mutants in the system. """
        self._wildtype = wildtype
        self._mutant = self._farthest_genotype(wildtype)
        self.Interactions.Mutations._indices = find_differences(self.wildtype, self.mutant)
        self.Interactions.Mutations._wildtype = [self.wildtype[i] for i in self.Interactions.Mutations.indices]
        self.Interactions.Mutations._mutations = [self.mutant[i] for i in self.Interactions.Mutations.indices]
        self.Interactions.Mutations._n = len(self.Interactions.Mutations.mutations)
        self._to_bits()
    
    @order.setter
    def order(self, order):
        """ Set the order of epistasis in the system. As a consequence, 
            this mapping object creates the """
        self._order = order
        self.Interactions.order = order
        
    @phenotypes.setter
    def phenotypes(self, phenotypes):
        """ NORMALIZE and set phenotypes from ordered list of phenotypes 
            
            Args:
            -----
            phenotypes: array-like or dict
                if array-like, it musted be ordered by genotype; if dict,
                this method automatically orders the phenotypes into numpy
                array.
        """
        if type(phenotypes) is dict:
            self._phenotypes = self._if_dict(phenotypes)/phenotypes[self.wildtype]
        else:
            if len(phenotypes) != len(self._genotypes):
                raise("Number of phenotypes does not equal number of genotypes.")
            else:
                wildtype_index = self.geno2index[self.wildtype]
                self._phenotypes = phenotypes/phenotypes[wildtype_index] 

        # log transform if log_transform = True
        if self.log_transform is True:
            self._phenotypes = np.log10(self._phenotypes)
            
        self.Binary._phenotypes = np.array([self.phenotypes[i] for i in self.Binary.indices])

        
    @errors.setter
    def errors(self, errors):
        """ Set error from ordered list of phenotype error. 
            
            Args:
            -----
            error: array-like or dict
                if array-like, it musted be ordered by genotype; if dict,
                this method automatically orders the errors into numpy
                array.
        """
        # Order phenotype errors from geno2pheno_err dictionary
        if type(errors) is dict:
            errors = self._if_dict(errors)
        
        # For log-transformations of error, need to translate errors to center around 1,
        # then take the log.
        if self.log_transform is True:
            errors = np.array((np.log10(1-errors), np.log10(1 + errors)))
        
        self._errors = errors
    
    # ---------------------------------
    # Useful methods for mapping object
    # ---------------------------------

    def _farthest_genotype(self, reference):
        """ Find the genotype in the system that differs at the most sites. """ 
        mutations = 0
        for genotype in self.genotypes:
            differs = hamming_distance(genotype, reference)
            if differs > mutations:
                mutations = int(differs)
                mutant = str(genotype)
        return mutant

    def _to_bits(self):
        """ Encode the genotypes an ordered binary set of genotypes with 
            wildtype as reference state (ref is all zeros).
            
            Essentially, this method maps each genotype to their binary representation
            relative to the 'wildtype' sequence.
        """
        w = list(self.wildtype)
        m = list(self.mutant)

        # get genotype indices
        geno2index = self.geno2index
        
        # Build binary space
        # this can be a really slow/memory intensive step ... need to revisit this
        full_genotypes, binaries = enumerate_space(self.wildtype, self.mutant, binary = True)
        bin2geno = dict(zip(binaries, full_genotypes))
        bits = list()
        bit_indices = list()
        # initialize bit_indicies
        for b in binaries:
            try:
                bit_indices.append(geno2index[bin2geno[b]])
                bits.append(b)
            except:
                pass
        self.Binary._genotypes = np.array(bits)
        self.Binary._indices = np.array(bit_indices)