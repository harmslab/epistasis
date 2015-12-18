__doc__ = """ Submodule of linear epistasis models. Includes full local and global epistasis models and regression model for low order models."""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np

# ------------------------------------------------------------
# seqspace imports
# ------------------------------------------------------------

from seqspace.utils import list_binary, enumerate_space, encode_mutations, construct_genotypes
from seqspace.errors import StandardErrorMap, StandardDeviationMap

# ------------------------------------------------------------
# Local imports
# ------------------------------------------------------------

from epistasis.decomposition import generate_dv_matrix
from epistasis.utils import epistatic_order_indices, build_model_params
from epistasis.models.base import BaseModel

def add_error_map(method):
    """ Decorates methods where errors are being created
    """
    def wrapper(self, *args, **kwargs):
        self.Interactions.err = StandardErrorMap()
        self.Interactions.std = StandardDeviationMap()
        method(self, *args, **kwargs)
    return wrapper 

# ------------------------------------------------------------
# Epistasis Mapping Classes
# ------------------------------------------------------------

class LocalEpistasisModel(BaseModel):

    def __init__(self, wildtype, genotypes, phenotypes, 
                stdeviations=None, 
                log_transform=False, 
                mutations=None, 
                n_replicates=1):
                
        """ Create a map of the local epistatic effects using expanded mutant
            cycle approach.

            i.e.
            Phenotype = K_0 + sum(K_i) + sum(K_ij) + sum(K_ijk) + ...

            __Arguments__:

            `wildtype` [str] : Wildtype genotype. Wildtype phenotype will be used as reference state.

            `genotypes` [array-like, dtype=str] : Genotypes in map. Can be binary strings, or not.

            `phenotypes` [array-like] : Quantitative phenotype values

            `stdevs` [array-like] : List of phenotype errors.

            `log_transform` [bool] : If True, log transform the phenotypes.
        """
        # Populate Epistasis Map
        super(LocalEpistasisModel, self).__init__(wildtype, genotypes, phenotypes,  
                stdeviations=stdeviations, 
                log_transform=log_transform, 
                mutations=mutations, 
                n_replicates=n_replicates)
                
        self.order = self.length

        # Construct the Interactions mapping -- Interactions Subclass is added to model
        self._construct_interactions()

        # Generate basis matrix for mutant cycle approach to epistasis.
        self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding={"1": 1, "0": 0})
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
        upper = np.sqrt(np.dot(np.square(self.X_inv), self.Binary.std.upper**2))
        
        # If the space is log transformed, then the errorbars are assymmetric
        if self.log_transform is True:
            lower = np.sqrt(np.dot(np.square(self.X_inv), self.Binary.std.lower**2))
    
        # Else, the lower errorbar is just upper
        else:
            lower = upper

        self.Interactions.std = StandardDeviationMap(self.phenotypes, upper, lower=lower)
        self.Interactions.err = StandardErrorMap(self.phenotypes, upper, n_replicates=self.n_replicates, lower=lower)
           
        

class GlobalEpistasisModel(BaseModel):

    def __init__(self, wildtype, genotypes, phenotypes, 
                stdeviations=None, 
                log_transform=False, 
                mutations=None, 
                n_replicates=1):
                
        """ Create a map of the global epistatic effects using Hadamard approach (defined by XX)

            This is the related to LocalEpistasisMap by the discrete Fourier
            transform of mutant cycle approach.

            __Arguments__:

            `wildtype` [str] : Wildtype genotype. Wildtype phenotype will be used as reference state.

            `genotypes` [array-like, dtype=str] : Genotypes in map. Can be binary strings, or not.

            `phenotypes` [array-like] : Quantitative phenotype values

            `stdevs` [array-like] : List of phenotype errors.

            `log_transform` [bool] : If True, log transform the phenotypes.
        """
        # Populate Epistasis Map
        super(GlobalEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, 
                                                    stdeviations=stdeviations, 
                                                    log_transform=log_transform, 
                                                    mutations=mutations, 
                                                    n_replicates=n_replicates)
                
                
        self.order = self.length

        # Construct the Interactions mapping -- Interactions Subclass is added to model
        self._construct_interactions()

        # Generate basis matrix for mutant cycle approach to epistasis.
        #self.weight_vector = hadamard_weight_vector(self.Binary.genotypes)
        self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding={"1": -1, "0": 1})
        
        # Inverse of the hadamard matrix is the hadamard divided by its dimension
        self.X_inv = 1.0/self.n * self.X

    def fit(self):
        """ Estimate the values of all epistatic interactions using the hadamard
            matrix transformation.
        """
        self.Interactions.values = np.dot(self.X_inv, self.Binary.phenotypes)


    def fit_error(self):
        """ Estimate the error of each epistatic interaction by standard error
            propagation of the phenotypes through the model.
        """
        _upper = np.sqrt(np.dot(np.square(self.X_inv), self.Binary.std.upper**2))
        
        # If the space is log transformed, then the errorbars are assymmetric
        if self.log_transform is True:
            _lower = np.sqrt(np.dot(np.square(self.X_inv), self.Binary.std.lower**2))
    
        # Else, the lower errorbar is just upper
        else:
            _lower = _upper
            
        self.Interactions.std = StandardDeviationMap(self.phenotypes, _upper, lower=_lower)
        self.Interactions.err = StandardErrorMap(self.phenotypes, _upper, lower=_lower, n_replicates=self.n_replicates)
