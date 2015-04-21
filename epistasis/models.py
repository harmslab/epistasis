# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import itertools as it
import numpy as np
from scipy.linalg import hadamard
from sklearn.linear_model import LinearRegression
from epistasis.core.mapping import EpistasisMap
from epistasis.regression_ext import generate_dv_matrix


# ------------------------------------------------------------
# Unique Epistasis Functions
# ------------------------------------------------------------   

def hadamard_weight_vector(genotypes):
    """ Build the hadamard weigth vector"""
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
    
    def __init__(self, wildtype, genotypes, phenotypes, phenotype_errors=None, log_phenotypes=False):
        """ Populate an Epistasis mapping object. """
        self.genotypes = genotypes
        self.wildtype = wildtype
        self.log_transform = log_phenotypes
        self.phenotypes = phenotypes
        if phenotype_errors is not None:
            self.phenotype_errors = phenotype_errors


class LocalEpistasisModel(GenericModel):
        
    def __init__(self, wildtype, genotypes, phenotypes, phenotype_errors=None, log_phenotypes=False):
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
        super(LocalEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, phenotype_errors, log_phenotypes)
        self.order = self.length
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.X = generate_dv_matrix(self.bits, self.interaction_labels)
        self.X_inv = np.linalg.inv(self.X)
        
    def estimate_interactions(self):
        """ Estimate the values of all epistatic interactions using the expanded
            mutant cycle method to order=number_of_mutations.
        """
        self.interaction_values = np.dot(self.X_inv, self.bit_phenotypes)
        
    def estimate_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        if self.log_transform is True:
            # If log-transformed, fit assymetric errorbars correctly
            upper = np.sqrt(np.dot(self.X, self.bit_phenotype_errors[0]**2))
            lower = np.sqrt(np.dot(self.X, self.bit_phenotype_errors[1]**2))
            self.interaction_errors = np.array((lower,upper))
        else:
            # Errorbars are symmetric, so only one column for errors is necessary
            self.interaction_errors = np.sqrt(np.dot(self.X, self.bit_phenotype_errors**2))
    
class GlobalEpistasisModel(GenericModel):
    
    def __init__(self, wildtype, genotypes, phenotypes, phenotype_errors=None, log_phenotypes=False):
        """ Create a map of the global epistatic effects using Hadamard approach.
            This is the related to LocalEpistasisMap by the discrete Fourier 
            transform of mutant cycle approach. 
        """
        # Populate Epistasis Map
        super(GlobalEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, phenotype_errors, log_phenotypes)
        self.order = self.length
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.weight_vector = hadamard_weight_vector(self.bits)
        self.X = hadamard(2**self.length)
        
    def estimate_interactions(self):
        """ Estimate the values of all epistatic interactions using the hadamard
        matrix transformation.
        """
        self.interaction_values = np.dot(self.weight_vector,np.dot(self.X, self.bit_phenotypes))
        
    def estimate_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        if self.log_transform is True:
            # If log-transformed, fit assymetric errorbars correctly
            # upper and lower are unweighted tranformations
            upper = np.sqrt(np.dot(abs(self.X), self.bit_phenotype_errors[0]**2))
            lower = np.sqrt(np.dot(abs(self.X), self.bit_phenotype_errors[1]**2))
            self.interaction_errors = np.array((np.dot(self.weight_vector, lower), np.dot(self.weight_vector, upper)))
        else:
            unweighted = np.sqrt(np.dot(abs(self.X), self.bit_phenotype_errors**2))
            self.interaction_errors = np.dot(self.weight_vector, unweighted)
            
    
class ProjectedEpistasisModel(GenericModel):
    
    def __init__(self, wildtype, genotypes, phenotypes, regression_order, phenotype_errors=None, log_phenotypes=False):
        """ Create a map from local epistasis model projected into lower order
            order epistasis interactions. Requires regression to estimate values. 
        """
        # Populate Epistasis Map
        super(ProjectedEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, phenotype_errors, log_phenotypes)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.order = regression_order
       
        self.X = generate_dv_matrix(self.bits, self.interaction_labels)
        
        # Regression properties
        self.regression_model = LinearRegression(fit_intercept=False)
        self.error_model = LinearRegression(fit_intercept=False)
        self.score = None
        
        
    def estimate_interactions(self):
        """ Estimate the values of all epistatic interactions using the expanded
            mutant cycle method to any order<=number of mutations.
        """
        self.regression_model.fit(self.X, self.bit_phenotypes)
        self.score = self.regression_model.score(self.X, self.bit_phenotypes)
        self.interaction_values = self.regression_model.coef_
        
        
    def estimate_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        if self.log_transform is True:
            interaction_errors = np.empty((2,len(self.interaction_labels)), dtype=float)
            for i in range(len(self.interaction_labels)):
                n = len(self.interaction_labels[i])
                interaction_errors[0,i] = np.sqrt(n*self.bit_phenotype_errors[0,i]**2)
                interaction_errors[1,i] = np.sqrt(n*self.bit_phenotype_errors[1,i]**2)
            self.interaction_errors = interaction_errors        
        else:
            interaction_errors = np.empty(len(self.interaction_labels), dtype=float)
            for i in range(len(self.interaction_labels)):
                n = len(self.interaction_labels[i])
                interaction_errors[i] = np.sqrt(n*self.bit_phenotype_errors[i]**2)
            self.interaction_errors = interaction_errors
        