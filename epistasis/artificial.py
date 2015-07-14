# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
from epistasis.regression_ext import generate_dv_matrix
from epistasis.mapping.epistasis import EpistasisMap
from epistasis.utils import enumerate_space, binary_mutations_map, genotype_params, label_to_key

# ------------------------------------------------------------
# ArtificialMap object can be used to quickly generating a toy
# space for testing the EpistasisModels
# ------------------------------------------------------------

class BaseArtificialMap(EpistasisMap):
    
    def __init__(self, length, order, log_transform=False):
        """ Generate a binary genotype-phenotype mape with the given length from epistatic interactions. """
        super(BaseArtificialMap, self).__init__()
        wildtype = '0'*length
        mutant = '1'*length
        self.mutations = binary_mutations_map(wildtype, mutant)
        self.genotypes, binaries = enumerate_space(wildtype, mutant)
        self.wildtype = wildtype
        self.order = order
        self.log_transform = log_transform
        self._construct_binary()
        self._construct_interactions()
        
    def build_phenotypes(self, values=None):
        """ Method for construction phenotypes from model. """
        raise Exception( """ Must be implemented in subclass. """)
        
    def random_epistasis(self, low, high, allow_neg=True):
        """Assign random values to epistatic terms. """ 
        vals = (high-low) * np.random.random(size=len(self.Interactions.labels)) + low
        return vals
        
    def random_knockout(self, n_knockouts):
        """ Set parameters"""
        indices = np.random.randint(len(self.Interactions.labels), size=n_knockouts)
        for i in indices:
            self.Interactions._values[i] = 0.0
            
    def add_noise(self, percent):
        """ Add noise to the phenotypes. """
        noise = np.empty(self.n, dtype=float)
        for i in range(self.n):
            noise[i] = percent*self.phenotypes[i]
        self.errors = noise
        
    def create_samples(self, n_samples):
        """ Generate artificial data sampled from phenotype and percent error. """
        try:
            errors = self.errors
        except:
            self.add_noise(0.05)
            errors = self.errors
            
        gen_phenotypes = np.empty((self.n, n_samples), dtype=float)
        gen_genotypes = np.empty((self.n, n_samples), dtype='<U'+str(self.length))
        
        for s in range(len(self.genotypes)):
            seq = self.genotypes[s]
            gen_genotypes[s] = np.array([seq for i in range(n_samples)])
            gen_phenotypes[s] = errors[s] * np.random.randn(n_samples) + self.phenotypes[s]
        
        return gen_genotypes, gen_phenotypes
        
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
        
        
class RandomEpistasisMap(BaseArtificialMap):
    
    def __init__(self, length, order, magnitude, log_transform=False):
        """ Choose random values for epistatic terms below and construct a genotype-phenotype map. 
            
            ASSUMES ADDITIVE MODEL (UNLESS LOG TRANSFORMED).
        
            Args:
            ----
            length: int
                length of strings
            order: int
                order of epistasis in space
            magnitude: float
                maximum value of abs(epistatic) term. 
            log_transform: bool
                return the log_transformed phenotypes.
        
        """
        super(RandomEpistasisMap,self).__init__(length, order, log_transform)
        self.random_epistasis(-1,1)
        self.phenotypes = self.build_phenotypes()
        
    def build_phenotypes(self, values=None):
        """ Build the phenotype map from epistatic interactions. """
        # Allocate phenotype numpy array
        phenotypes = np.zeros(self.n, dtype=float)
    
        # Check for custom values
        if values is None:
            values = self.Interactions.values
    
        # Build phenotypes for binary representation of space
        self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels)
        bit_phenotypes = np.dot(self.X,values)
        if self.log_transform:
            bit_phenotypes = 10**bit_phenotypes
        # Reorder phenotypes to map back to genotypes
        for i in range(len(self.Binary.indices)):
            phenotypes[self.Binary.indices[i]] = bit_phenotypes[i]
        
        return phenotypes

class ThresholdEpistasisMap(BaseArtificialMap):
    
    def __init__(self, length, order, threshold, sharpness, magnitude):
        """ Build an epistatic genotype phenotype map with thresholding behavior. Built from
            the function:
            
                 f(epistasis_model) = \theta - exp(-\nu * epistasis_model)
        
            where epistasis model is a MULTIPLICATIVE MODEL.      
            
            Args:
            ----
            length: int
                length of strings
            order: int
                order of epistasis in space
            threshold: float
                fitness/phenotype thresholding value.
            sharpness: float
                rate of exponential growth towards thresholding value.
            log_transform: bool
                return the log_transformed phenotypes.
        """
        super(ThresholdEpistasisMap,self).__init__(length, order, log_transform=False)
        #if magnitude > threshold:
         #   raise Warning(""" Magnitude of epistasis could be greater than thesholding value. """)
        vals = self.random_epistasis(1, 1+magnitude, allow_neg=False)
        vals[0] = 1.0
        self.Interactions.values = vals
        self.raw_phenotypes = self.build_phenotypes()
        self.phenotypes = self.threshold_func(self.raw_phenotypes, threshold, sharpness)
        
    def build_phenotypes(self, values=None):
        """ Uses the multiplicative model to construct raw phenotypes. """
        phenotypes = np.empty(len(self.genotypes), dtype=float)
        param_map = self.Interactions.key2value
        for i in range(len(phenotypes)):
            params = genotype_params(self.genotypes[i], order=self.order)
            values = np.array([param_map[label_to_key(p)] for p in params])
            phenotypes[i] = np.prod(values)
        return phenotypes
    
    def threshold_func(self, raw_phenotypes, threshold, sharpness):
        """ Apply the thresholding effect. """
        phenotypes = np.empty(self.n, dtype=float)
        for i in range(self.n):
            phenotypes[i] = threshold * (1 - np.exp(-sharpness*raw_phenotypes[i]))
        return phenotypes
        
        