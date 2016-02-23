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

# ------------------------------------------------------------
# Epistasis Mapping Classes
# ------------------------------------------------------------

class LinearEpistasisModel(BaseModel):
    
    def __init__(self, wildtype, genotypes, phenotypes, 
        stdeviations=None, 
        log_transform=False, 
        mutations=None, 
        n_replicates=1,
        model_type="local"):
                
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
        super(LinearEpistasisModel, self).__init__(wildtype, genotypes, phenotypes,  
            stdeviations=stdeviations, 
            log_transform=log_transform, 
            mutations=mutations, 
            n_replicates=n_replicates)
        
        # Define the encoding for different models
        model_types = {
            "local": {
                "encoding": {"1": 1, "0": 0},       # Decomposition matrix encoding
                "inverse": 1.0                      # Inverse functions coefficient
            }, 
            "global": {
                "encoding": {"1": -1, "0": 1},
                "inverse": 1.0/self.n
            }
        }
        
        # Set encoding from model_type given
        self.encoding = model_types[model_type]["encoding"]
                
        # Order of model is equal to number of mutations
        self.order = self.length

        # Generate basis matrix for mutant cycle approach to epistasis.
        self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding=self.encoding)
        
        # Calculate the inverse of the matrix
        self.X_inv = model_types[model_type]["inverse"] * np.linalg.inv(self.X)


    def fit(self):
        """ Estimate the values of all epistatic interactions using the expanded
            mutant cycle method to order=number_of_mutations.
        """
        self.Interactions.values = np.dot(self.X_inv, self.Binary.phenotypes)


    def fit_error(self):
        """ Estimate the error of each epistatic interaction by standard error
            propagation of the phenotypes through the model.
            
            
            For multiplicative: 
            ------------------
                f_x = phenotype x 
                sigma_f = standard deviation of phenotype x
                beta_i = epistatic coefficient i
            
                (sigma_beta_i)**2 = (beta_i ** 2) *  ( (sigma_f_x ** 2) / (f_x ** 2) + ... )
            
        """
        upper = np.sqrt(np.dot(np.square(self.X_inv), self.Binary.std.upper**2))
        
        # If the space is log transformed, then the errorbars are assymmetric
        if self.log_transform is True:
            # Get variables
            beta_i = self.Interactions.Raw.values
            sigma_f_x = self.Raw.std.upper
            f_x = self.Raw.phenotypes
            
            # Calculate unscaled terms
            upper = np.sqrt( (beta_i**2) * np.dot(np.square(self.X_inv),(sigma_f_x**2/f_x**2))) 
            
            # Create a raw map of the errors
            self.Interactions.Raw.std = StandardDeviationMap(self.Interactions.Raw.values, upper)
            self.Interactions.Raw.err = StandardErrorMap(self.Interactions.Raw.values, upper, n_replicates=self.n_replicates)
            
            values = beta_i
            
        # Else, the lower errorbar is just upper
        else:
            upper = np.sqrt(np.dot(np.square(self.X_inv), self.Binary.std.upper**2))
            values = self.Interactions.values

        self.Interactions.std = StandardDeviationMap(values, upper, log_transform=self.log_transform)
        self.Interactions.err = StandardErrorMap(values, upper, n_replicates=self.n_replicates, log_transform=self.log_transform)        
        


class LocalEpistasisModel(LinearEpistasisModel):

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
                n_replicates=n_replicates,
                model_type="local")
                
        
        
class GlobalEpistasisModel(LinearEpistasisModel):

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
                                                    n_replicates=n_replicates,
                                                    model_type="global")
                
                