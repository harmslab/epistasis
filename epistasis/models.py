# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import itertools as it
import numpy as np
from scipy.linalg import hadamard
from regression_ext import generate_dv_matrix
from sklearn.linear_model import LinearRegression
from .core.em import EpistasisMap

# ------------------------------------------------------------
# Unique Epistasis Functions
# ------------------------------------------------------------   

def hadamard_weight_vector(genotypes):
    l = len(genotypes)
    n = len(genotypes[0])
    weights = np.zeros((l, l), dtype=float)
    for g in range(l):
        epistasis = float(genotypes[g].count("1"))
        weights[g][g] = ((-1)**epistasis)/(2**(n-epistasis))    
    return weights    

def cut_interaction_labels(labels, order):
    """ Cut off interaction labels at certain order of interactions. """
    return [l for l in labels if len(l) <= order]
    
# ------------------------------------------------------------
# Epistasis Mapping Classes
# ------------------------------------------------------------
class GenericModel(EpistasisMap):
    
    def __init__(self, genotypes, phenotypes, phenotype_errors=None, log_phenotypes=True):
        """ Populate an Epistasis mapping object. """
        self.genotypes = genotypes
        if log_phenotypes is True:
            self.phenotypes = np.log(phenotypes)
        else:
            self.phenotypes = phenotypes
        if phenotype_errors is not None:
            self.phenotype_errors = phenotype_errors


class LocalEpistasisMap(GenericModel):
        
    def __init__(self, genotypes, phenotypes, phenotype_errors=None, log_phenotypes=True):
        """ Create a map of the local epistatic effects using expanded mutant 
            cycle approach.
            
            i.e.
            Phenotype = K_0 + sum(K_i) + sum(K_ij) + sum(K_ijk) + ...
            
            Parameters:
            ----------
            geno_pheno_dict: OrderedDict
                Dictionary with keys=ordered genotypes by their binary value, 
                mapped to their phenotypes.
            log_phenotypes: bool (default=True)
                Log transform the phenotypes for additivity.
        """
        # Populate Epistasis Map
        super(LocalEpistasisMap, self).__init__(genotypes, phenotypes, phenotype_errors, log_phenotypes)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.X = generate_dv_matrix(self.bits, self.interaction_labels)
        self.X_inv = np.linalg.inv(self.X)
        
    def estimate_interactions(self):
        """ Estimate the values of all epistatic interactions using the expanded
            mutant cycle method to order=number_of_mutations.
        """
        self.interactions = np.dot(self.X_inv, self.Y)
        
    def estimate_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        self.interaction_errors = np.sqrt(np.dot(self.X, self.phenotype_errors**2))
    
class GlobalEpistasisMap(EpistasisMap):
    
    def __init__(self, genotypes, phenotypes, phenotype_errors=None, log_phenotypes=True):
        """ Create a map of the global epistatic effects using Hadamard approach.
            This is the related to LocalEpistasisMap by the discrete Fourier 
            transform of mutant cycle approach. 
        """
        # Populate Epistasis Map
        super(LocalEpistasisMap, self).__init__(genotypes, phenotypes, phenotype_errors, log_phenotypes)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.weight_vector = hadamard_weight_vector(self.bits)
        self.X = hadamard(2**self.length)
        
    def estimate_interactions(self):
        """ Estimate the values of all epistatic interactions using the hadamard
        matrix transformation.
        """
        self.interactions = np.dot(self.weight_vector,np.dot(self.X, self.Y))
        
    def estimate_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        self.interaction_errors = np.dot(self.weight_vector, np.sqrt(np.dot(abs(self.X), self.phenotype_errors**2)))
    
class ProjectedEpistasisMap(EpistasisMap):
    
    def __init__(self, regression_order, genotypes, phenotypes, phenotype_errors=None, log_phenotypes=True):
        """ Create a map from local epistasis model projected into lower order
            order epistasis interactions. Requires regression to estimate values. 
        """
        # Populate Epistasis Map
        super(LocalEpistasisMap, self).__init__(genotypes, phenotypes, phenotype_errors, log_phenotypes)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.order = regression_order
       
        self.X = generate_dv_matrix(self.bits, self.interaction_labels)
        
        # Regression properties
        self.regression_model = LinearRegression(fit_intercept=False)
        self.error_model = LinearRegression(fit_intercept=False)
        self.r_squared = None
        
        
    def estimate_interactions(self):
        """ Estimate the values of all epistatic interactions using the expanded
            mutant cycle method to any order<=number of mutations.
        """
        self.regression_model.fit(self.X, self.Y)
        self.r_squared = self.regression_model.score(self.X, self.Y)
        self.interactions = self.regression_model.coef_
        
        
    def estimate_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        interaction_errors = np.empty(len(self.interaction_labels), dtype=float)
        for i in range(len(self.interaction_labels)):
            n = len(self.interaction_labels[i])
            interaction_errors[i] = np.sqrt(n*self.phenotype_errors[i]**2)
        self.interaction_errors = interaction_errors
        