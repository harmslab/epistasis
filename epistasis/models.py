# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import itertools as it
import numpy as np
from scipy.linalg import hadamard
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from collections import OrderedDict

# ------------------------------------------------------------
# Local imports
# ------------------------------------------------------------
from epistasis.mapping.epistasis import EpistasisMap
from epistasis.regression_ext import generate_dv_matrix
from epistasis.utils import epistatic_order_indices, list_binary, enumerate_space, encode_mutations, construct_genotypes, build_model_params, farthest_genotype, binary_mutations_map

# ------------------------------------------------------------
# Unique Epistasis Functions
# ------------------------------------------------------------   

def hadamard_weight_vector(genotypes):
    """ Build the hadamard weigth vector. """
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
class BaseModel(EpistasisMap):
    
    def __init__(self, wildtype, genotypes, phenotypes, errors=None, log_transform=False, mutations=None):
        """ Populate an Epistasis mapping object. 
        
            Args:
            ----
            wildtype: str
                Wildtype genotype. Wildtype phenotype will be used as reference state.
            genotypes: array-like, dtype=str
                Genotypes in map. Can be binary strings, or not.
            phenotypes: array-like
                Quantitative phenotype values
            errors: array-like
                List of phenotype errors.
            log_transform: bool
                If True, log transform the phenotypes.
            mutations:
            
        """
        super(BaseModel, self).__init__()

        self.genotypes = genotypes
        self.wildtype = wildtype
        self.log_transform = log_transform
        self.phenotypes = phenotypes
        
        # Defaults to binary mapping if not specific mutations are named
        if mutations is None:
            mutant = farthest_genotype(self.wildtype, self.genotypes)
            self.mutations = binary_mutations_map(self.wildtype, mutant)
        else:
            self.mutations = mutations
            
        # Construct a binary representation of the map (method inherited from parent class)
        # and make it a subclass of the model.
        self._construct_binary()
      
        # Model error if given. 
        if errors is not None:
            self.errors = errors
        
            
    def get_order(self, order, errors=False, label="genotype"):
        """ Return a dict of interactions to values of a given order. """
        
        # get starting index of interactions
        if order > self.order:
            raise Exception("Order argument is higher than model's order")
            
        # Determine the indices of this order of interactions.
        start, stop = epistatic_order_indices(self.length,order)
        # Label type.
        if label == "genotype":
            keys = self.Interactions.genotypes
        elif label == "keys":
            keys = self.Interactions.keys
        else:
            raise Exception("Unknown keyword argument for label.")
        
        # Build dictionary of interactions
        stuff = OrderedDict(zip(keys[start:stop], self.Interactions.values[start:stop]))
        if errors:
            errors = OrderedDict(zip(keys[start:stop], self.Interactions.errors[start:stop]))
            return stuff, errors
        else:
            return stuff
            
    def fit(self):
        """ Fitting methods for epistasis models. """
        raise Exception("""Must be implemented in a subclass.""")
        
    def fit_error(self):
        """ Fitting method for errors in the epistatic parameters. """
        raise Exception("""Must be implemented in a subclass.""")
            


class LocalEpistasisModel(BaseModel):
        
    def __init__(self, wildtype, genotypes, phenotypes, errors=None, log_transform=False, mutations=None):
        """ Create a map of the local epistatic effects using expanded mutant 
            cycle approach.
            
            i.e.
            Phenotype = K_0 + sum(K_i) + sum(K_ij) + sum(K_ijk) + ...
            
            Args:
            ----
            wildtype: str
                Wildtype genotype. Wildtype phenotype will be used as reference state.
            genotypes: array-like, dtype=str
                Genotypes in map. Can be binary strings, or not.
            phenotypes: array-like
                Quantitative phenotype values
            errors: array-like
                List of phenotype errors.
            log_transform: bool
                If True, log transform the phenotypes.
        """
        # Populate Epistasis Map
        super(LocalEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, errors=errors, log_transform=log_transform, mutations=mutations)
        self.order = self.length
        
        # Construct the Interactions mapping -- Interactions Subclass is added to model
        self._construct_interactions()
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels)
        self.X_inv = np.linalg.inv(self.X)
        
    def fit(self):
        """ Estimate the values of all epistatic interactions using the expanded
            mutant cycle method to order=number_of_mutations.
        """
        self.Interactions.values = np.dot(self.X_inv, self.Binary.phenotypes)
        
    def fit_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        if self.log_transform is True:
            # If log-transformed, fit assymetric errorbars correctly
            upper = np.sqrt(np.dot(self.X, self.Binary.errors[0]**2))
            lower = np.sqrt(np.dot(self.X, self.Binary.errors[1]**2))
            self.Interactions.errors = np.array((lower,upper))
        else:
            # Errorbars are symmetric, so only one column for errors is necessary
            self.Interactions.errors = np.sqrt(np.dot(self.X, self.Binary.errors**2))
            
    
class GlobalEpistasisModel(BaseModel):
    
    def __init__(self, wildtype, genotypes, phenotypes, errors=None, log_transform=False):
        """ Create a map of the global epistatic effects using Hadamard approach.
            This is the related to LocalEpistasisMap by the discrete Fourier 
            transform of mutant cycle approach. 
            
            Args:
            ----
            wildtype: str
                Wildtype genotype. Wildtype phenotype will be used as reference state.
            genotypes: array-like, dtype=str
                Genotypes in map. Can be binary strings, or not.
            phenotypes: array-like
                Quantitative phenotype values
            errors: array-like
                List of phenotype errors.
            log_transform: bool
                If True, log transform the phenotypes.
        """
        # Populate Epistasis Map
        super(GlobalEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, errors, log_transform)
        self.order = self.length
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.weight_vector = hadamard_weight_vector(self.Binary.genotypes)
        self.X = hadamard(2**self.length)
        
    def fit(self):
        """ Estimate the values of all epistatic interactions using the hadamard
        matrix transformation.
        """
        self.Interactions.values = np.dot(self.weight_vector,np.dot(self.X, self.Binary.phenotypes))
        
    def fit_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        if self.log_transform is True:
            # If log-transformed, fit assymetric errorbars correctly
            # upper and lower are unweighted tranformations
            upper = np.sqrt(np.dot(abs(self.X), self.Binary.errors[0]**2))
            lower = np.sqrt(np.dot(abs(self.X), self.Binary.errors[1]**2))
            self.Interactions.errors = np.array((np.dot(self.weight_vector, lower), np.dot(self.weight_vector, upper)))
        else:
            unweighted = np.sqrt(np.dot(abs(self.X), self.Binary.errors**2))
            self.Interactions.errors = np.dot(self.weight_vector, unweighted)
            
    
class ProjectedEpistasisModel(BaseModel):
    
    def __init__(self, wildtype, genotypes, phenotypes, order=None, parameters=None, errors=None, log_transform=False, mutations=None):
        """ Create a map from local epistasis model projected into lower order
            order epistasis interactions. Requires regression to estimate values.
            
            Args:
            ----
            wildtype: str
                Wildtype genotype. Wildtype phenotype will be used as reference state.
            genotypes: array-like, dtype=str
                Genotypes in map. Can be binary strings, or not.
            phenotypes: array-like
                Quantitative phenotype values
            order: int
                Order of regression; if None, parameters must be passed in manually as parameters=<list of lists>
            parameters: dict
                interaction keys with their values expressed as lists.
            errors: array-like
                List of phenotype errors.
            log_transform: bool
                If True, log transform the phenotypes.
        """
        # Populate Epistasis Map
        super(ProjectedEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, errors, log_transform, mutations)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        if order is not None:
            self.order = order  
            # Construct the Interactions mapping -- Interactions Subclass is added to model
            self._construct_interactions()      
        elif parameters is not None:
            self._construct_interactions()      
            self.Interactions.labels = list(parameters.values())
        else:
            raise Exception("""Need to specify the model's `order` argument or manually 
                                list model parameters as `parameters` argument.""")
       
        self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels)
        
        # Regression properties
        self.regression_model = LinearRegression(fit_intercept=False)
        self.error_model = LinearRegression(fit_intercept=False)


    @property
    def score(self):
        """ Get the epistasis model score after estimating interactions. """
        return self._score


    def fit(self):
        """ Estimate the values of all epistatic interactions using the expanded
            mutant cycle method to any order<=number of mutations.
        """
        self.regression_model.fit(self.X, self.Binary.phenotypes)
        self._score = self.regression_model.score(self.X, self.Binary.phenotypes)
        self.Interactions.values = self.regression_model.coef_
        
        
    def fit_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
            
            CANNOT propagate error in regressed model.
        """
        pass
        
      
    def predict(self):
        """ Infer the phenotypes from model.
            
            Returns:
            -------
            genotypes: array
                array of genotypes -- in same order as phenotypes
            phenotypes: array
                array of quantitative phenotypes.
        """
        genotypes, binaries = enumerate_space(self.wildtype, self.mutant)
        X = generate_dv_matrix(binaries, self.Interactions.labels)
        phenotypes = self.regression_model.predict(X)
        return genotypes, phenotypes
        
        
class NonlinearEpistasisModel(BaseModel):
        
    def __init__(self, wildtype, genotypes, phenotypes, function, x, parameters, errors=None, log_transform=False, mutations=None):
        """ Fit a nonlinear epistasis model to a genotype-phenotype map. The function and parameters must
            specified prior. 
        
            Uses Scipy's curve_fit function.
            
            Args:
            ----
            wildtype: str
                Wildtype genotype. Wildtype phenotype will be used as reference state.
            genotypes: array-like, dtype=str
                Genotypes in map. Can be binary strings, or not.
            phenotypes: array-like
                Quantitative phenotype values
            function: callable
                Function to fit the linear model. Must have the arguments that correspond to the parameters.
            x: 2d array
                Array of dummy variables for epistasis model fitting (use `generate_dv_matrix` in regression_ext)
            parameters: dict
                interaction keys with their values expressed as lists.
            errors: array-like
                List of phenotype errors.
            log_transform: bool
                If True, log transform the phenotypes.
        """
        # Populate Epistasis Map
        super(NonlinearEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, errors, log_transform, mutations=mutations)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        if parameters is not None:
            self._construct_interactions()      
            self.Interactions.keys = list(parameters.values())
            self.Interactions.labels = list(parameters.values())
        else:
            raise Exception("""Need to specify the model's `order` argument or manually 
                                list model parameters as `parameters` argument.""")
   
        self.X = x
        self.function = function
    
    def fit(self):
        """ Fit the nonlinear function """
        values, cov = curve_fit(self.function, 
                                self.X.T, 
                                self.Binary.phenotypes, 
                                p0=0.1*np.ones(len(self.Interactions.labels), dtype=float), 
                                maxfev=1000000)
                                
        self.Interactions.values = values[:]
        # Setting error if covariance was estimated, else pass.
        try:
            self.error = cov[:]
        except:
            pass
                                                    
            