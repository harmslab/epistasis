# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
from epistasis.regression_ext import generate_dv_matrix
from epistasis.utils import enumerate_space
from epistasis.mapping.epistasis import EpistasisMap

# ------------------------------------------------------------
# ArtificialMap object can be used to quickly generating a toy
# space for testing the EpistasisModels
# ------------------------------------------------------------

class BaseArtificialMap(EpistasisMap):
    
    def __init__(self, length, order, log_transform=False):
        """ Generate a binary genotype-phenotype mape with the given length from epistatic interactions. """
        super(ArtificialMap, self).__init__()
        wildtype = 'A'*length
        mutant = 'T'*length
        self.genotypes, binaries = enumerate_space(wildtype, mutant)
        self.wildtype = wildtype
        self.order = order
        self.log_transform = log_transform
        
    def build_phenotypes(self):
        """ Build the phenotype map from epistatic interactions. """
        # Allocate phenotype numpy array
        phenotypes = np.zeros(self.n, dtype=float)
        
        # Build phenotypes for binary representation of space
        self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels)
        bit_phenotypes = np.dot(self.X,self.Interactions.values)
        if self.log_transform:
            bit_phenotypes = 10**bit_phenotypes
        # Reorder phenotypes to map back to genotypes
        for i in range(len(self.Binary.indices)):
            phenotypes[self.Binary.indices[i]] = bit_phenotypes[i]
            
        return phenotypes
        
    def random_epistasis(self, magnitude):
        """Assign random values to epistatic terms. """ 
        vals = (-1)**(np.random.randint(1,10, size=len(self.Interactions.labels))) * np.random.rand(len(self.Interactions.labels))*magnitude
        self.Interactions.values = vals
        
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
        super(RandomEpistasisMap,self).__init__(self, length, order, log_transform):
        self.random_epistasis(magnitude)
        self.phenotypes = self.build_phenotypes()

class ThresholdEpistasisMap(BaseArtificialMap):
    
    def __init__(self, length, order, threshold, sharpness, log_transform=False):
        """ Build an epistatic genotype phenotype map with thresholding behavior. Built from
            the function:
            
                 f(epistasis_model) = \theta - exp(-\nu * epistasis_model)
        
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
        super(ThresholdEpistasisMap,self).__init__(self, length, order, log_transform):
        self.random_epistasis(thresold)
        self.raw_phenotypes = self.build_phenotypes()
        self.phenotypes = self.threshold_func(self.raw_phenotypes)
        
        
    def threshold_func(self, x, threshold, sharpness):
        """ Apply the thresholding effect. """
        return threshold - np.exp(-sharpness*x)
        
        