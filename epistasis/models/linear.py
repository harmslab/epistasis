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
    """Construct a genotype-phenotype map and fit with a linear epistasis model,
    defined as follows:
    i.e.
    Phenotype = K_0 + sum(K_i) + sum(K_ij) + sum(K_ijk) + ...

    Parameters
    ----------
    wildtype : str
        Wildtype genotype. Wildtype phenotype will be used as reference state.
    genotypes : array-like, dtype=str
        Genotypes in map. Can be binary strings, or not.
    phenotypes : array-like
        Quantitative phenotype values
    stdeviations : array-like
        List of phenotype errors.
    log_transform : bool
        If True, log transform the phenotypes.
    mutations : dict
        Mapping sites to mutational alphabet.
    n_replicates : int
        number of replicates.
    model_type : str
        type of model to use.
    logbase : callable
        log spaces to transform phenotypes.


    Attributes
    ----------
    See seqspace package for more docs.

    Epistasis : EpistasisMap
        object containing epistatic interactions values
    """
    def __init__(self, wildtype, genotypes, phenotypes,
        stdeviations=None,
        log_transform=False,
        mutations=None,
        n_replicates=1,
        model_type="local",
        logbase=np.log10):
        # Populate Epistasis Map
        super(LinearEpistasisModel, self).__init__(wildtype, genotypes, phenotypes,
            stdeviations=stdeviations,
            log_transform=log_transform,
            mutations=mutations,
            n_replicates=n_replicates,
            logbase=logbase)

        # Define the encoding for different models
        model_types = {
            "local": {
                "encoding": {"1": 1, "0": 0},       # Decomposition matrix encoding
                "inverse": 1.0                      # Inverse functions coefficient
            },
            "global": {
                "encoding": {"1": 1, "0": -1},
                "inverse": 1.0
            }
        }
        # Set order of model.
        self.order = len(self.mutations)
        # Build EpistasisMap
        self.epistasis.order = self.order
        # Set encoding from model_type given
        self.encoding = model_types[model_type]["encoding"]
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.X = generate_dv_matrix(
            self.binary.genotypes,
            self.epistasis.labels,
            encoding=self.encoding
        )
        # Calculate the inverse of the matrix
        self.X_inv = np.linalg.inv(self.X) #* model_types[model_type]["inverse"]


    @property
    def metadata(self):
        """Return the metadata for this model."""
        return {
            "wildtype" : self.wildtype,
            "genotypes" : self.genotypes,
            "phenotypes" : self.Raw.phenotypes,
            "stdeviations" : self.stdeviations,
            "n_replicates" : self.n_replicates,
            "mutations" : self.mutations,
            "log_transform" : self.log_transform,
            "order" : self.order,
            "epistasis" : {
                "keys" : self.epistasis.keys,
                "values" : self.epistasis.values,
            }
        }


    def fit(self):
        """ Estimate the values of all epistatic interactions using the expanded
        mutant cycle method to order=number_of_mutations.
        """
        # Check if log transform is necessary.
        if self.log_transform:
            values = self.base**np.dot(self.X_inv, self.binary.log.phenotypes)
        else:
            values = np.dot(self.X_inv, self.binary.phenotypes)
        self.epistasis.values = values

    def fit_error(self):
        """ Estimate the error of each epistatic interaction by standard error
        propagation of the phenotypes through the model.

        Example
        -------
        f_x = phenotype x
        sigma_f = standard deviation of phenotype x
        beta_i = epistatic coefficient i

        (sigma_beta_i)**2 = (beta_i ** 2) *  ( (sigma_f_x ** 2) / (f_x ** 2) + ... )
        """
        # If the space is log transformed, then the errorbars are assymmetric
        if self.log_transform:
            # Get variables
            beta_i = self.epistasis.values
            sigma_f_x = self.std.upper
            f_x = self.phenotypes
            # Calculate unscaled terms
            stdeviations = np.sqrt( (beta_i**2) * np.dot(np.square(self.X_inv),(sigma_f_x**2/f_x**2)))
        # Else, the lower errorbar is just upper
        else:
            stdeviations = np.sqrt(np.dot(np.square(self.X_inv), self.binary.std.upper**2))
        #Set stdeviation for epistasis.
        self.epistasis.stdeviations = stdeviations


class LocalEpistasisModel(LinearEpistasisModel):
    """Construct a genotype-phenotype map and fit with a linear epistasis model,
    defined as follows
    P = K_0 + sum(K_i) + sum(K_ij) + sum(K_ijk) + ...

    Parameters
    ----------
    wildtype : str
        Wildtype genotype. Wildtype phenotype will be used as reference state.
    genotypes : array-like, dtype=str
        Genotypes in map. Can be binary strings, or not.
    phenotypes : array-like
        Quantitative phenotype values
    stdeviations : array-like
        List of phenotype errors.
    log_transform : bool
        If True, log transform the phenotypes.
    mutations : dict
        Mapping sites to mutational alphabet.
    n_replicates : int
        number of replicates.
    model_type : str
        type of model to use.
    logbase : callable
        log spaces to transform phenotypes.
    """
    def __init__(self, wildtype, genotypes, phenotypes,
                stdeviations=None,
                log_transform=False,
                mutations=None,
                n_replicates=1,
                logbase=np.log10):
        # Populate Epistasis Map
        super(LocalEpistasisModel, self).__init__(wildtype, genotypes, phenotypes,
                stdeviations=stdeviations,
                log_transform=log_transform,
                mutations=mutations,
                n_replicates=n_replicates,
                model_type="local",
                logbase=logbase)



class GlobalEpistasisModel(LinearEpistasisModel):
    """Construct a genotype-phenotype map and fit with a linear epistasis model,
    defined as follows
    Phenotype = K_0 + sum(K_i) + sum(K_ij) + sum(K_ijk) + ...

    Parameters
    ----------
    wildtype : str
        Wildtype genotype. Wildtype phenotype will be used as reference state.
    genotypes : array-like, dtype=str
        Genotypes in map. Can be binary strings, or not.
    phenotypes : array-like
        Quantitative phenotype values
    stdeviations : array-like
        List of phenotype errors.
    log_transform : bool
        If True, log transform the phenotypes.
    mutations : dict
        Mapping sites to mutational alphabet.
    n_replicates : int
        number of replicates.
    model_type : str
        type of model to use.
    logbase : callable
        log spaces to transform phenotypes.
    """
    def __init__(self, wildtype, genotypes, phenotypes,
                stdeviations=None,
                log_transform=False,
                mutations=None,
                n_replicates=1,
                logbase=np.log10):
        # Populate Epistasis Map
        super(GlobalEpistasisModel, self).__init__(wildtype, genotypes, phenotypes,
                                                    stdeviations=stdeviations,
                                                    log_transform=log_transform,
                                                    mutations=mutations,
                                                    n_replicates=n_replicates,
                                                    model_type="global",
                                                    logbase=logbase)
