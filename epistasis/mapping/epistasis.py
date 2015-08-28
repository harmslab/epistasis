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
from epistasis.mapping.mutation import MutationMap
from epistasis.mapping.interaction import InteractionMap
from epistasis.utils import hamming_distance, encode_mutations, construct_genotypes, params_index_map, build_model_params

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
        self.Mutations = MutationMap()
        
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
    def mutations(self):
        """ Get the furthest genotype from the wildtype genotype. """
        return self._mutations

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
    
    @genotypes.setter
    def genotypes(self, genotypes):
        """ Set genotypes from ordered list of sequences. """
        self._n = len(genotypes)
        self._length = len(genotypes[0])
        self._genotypes = np.array(genotypes)
        self._indices = np.arange(self.n)
        
    @wildtype.setter
    def wildtype(self, wildtype):
        """ Set the reference genotype among the mutants in the system. """
        self._wildtype = wildtype
        self.Mutations.wildtype = wildtype
    
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
        self.Mutations.mutations = mutations
        self.Mutations.n = len(mutations)
    
    @order.setter
    def order(self, order):
        """ Set the order of epistasis in the system. As a consequence, 
            this mapping object creates the """
        self._order = order
        
    @phenotypes.setter
    def phenotypes(self, phenotypes):
        """ Set phenotypes from ordered list of phenotypes 
            
            Args:
            -----
            phenotypes: array-like or dict
                if array-like, it musted be ordered by genotype; if dict,
                this method automatically orders the phenotypes into numpy
                array.
        """
        if type(phenotypes) is dict:
            self._phenotypes = self._if_dict(phenotypes)
        else:
            if len(phenotypes) != len(self._genotypes):
                raise ValueError("Number of phenotypes does not equal number of genotypes.")
            else:
                self._phenotypes = phenotypes

        # log transform if log_transform = True
        if self.log_transform is True:
            self._untransformed_phenotypes = self._phenotypes
            self._phenotypes = np.log10(self._phenotypes)

        
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
            # Reference = http://onlinelibrary.wiley.com/doi/10.1002/sim.1525/epdf
            # \sigma_{log(f)}^{2} = log(1 + \sigma_{f}6{2}/mean(f)^{2}) 
            
            self._errors = np.array((   -np.sqrt(np.log10(1 + (errors**2)/self._untransformed_phenotypes**2)), 
                                        np.sqrt(np.log10(1 + (errors**2)/self._untransformed_phenotypes**2))))
                                                
            self.Binary._errors = np.array([self._errors[:,i] for i in self.Binary.indices]).T
        else:
            self._errors = errors
            self.Binary._errors = np.array([errors[i] for i in self.Binary.indices])
        
    
    # ------------------------------------------------------------
    # Useful methods for mapping object
    # ------------------------------------------------------------

    def _construct_binary(self):
        """ Encode the genotypes an ordered binary set of genotypes with 
            wildtype as reference state (ref is all zeros).
            
            This method maps each genotype to their binary representation
            relative to the 'wildtype' sequence.
        """
        self.Binary = BinaryMap()
        self.Binary.mutations = encode_mutations(self.wildtype, self.mutations)
        genotypes, self.Binary.genotypes = construct_genotypes(self.Binary.mutations)
        self.Binary.indices = np.array([self.geno2index[genotypes[i]] for i in range(len(self.Binary.genotypes))])
        
        # Grab phenotypes if they exist. Otherwise, pass.
        try:
            self.Binary.phenotypes = np.array([self.geno2pheno[genotypes[i]] for i in range(len(self.Binary.genotypes))])
        except:
            pass
        
    def _construct_interactions(self):
        """ Construct the interactions mapping for an epistasis model. 
            
            Must populate the Mutations subclass before setting interactions. 
        """
        self.Interactions = InteractionMap(self.Mutations)
        self.Interactions._length = self.length
        self.Interactions.log_transform = self.log_transform
        self.Interactions.mutations = params_index_map(self.Binary.mutations) # construct the mutations mapping
        
        # If an order is specified, construct epistatic interaction terms.
        try:
            self.Interactions.order = self.order
            self.Interactions.labels = build_model_params(self.Interactions.length, 
                                                      self.Interactions.order, 
                                                      self.Interactions.mutations)
        except:
            pass
