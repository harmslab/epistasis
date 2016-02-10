__doc__ = """ Principal component analysis for genotype-phenotype maps in submodule."""

# -------------------------------------- -------------
# Module for Principal Component Analysis of Epistasis
# ----------------------------------------------------

from sklearn.decomposition import PCA
from epistasis.decomposition import generate_dv_matrix
from epistasis.models.base import BaseModel

class PCAStats(object):
    
    def __init__(self, model):
        """
        Object that holds different statistical readouts of PCA analysis 
        """
        self.model = model
    
    def cumulative_variance(self, n_components):
        """ 
        Return the cumulative explained variance for a given number of components.
        """
        return sum(self.model.explained_variance_ratio[:n_components])
        
    def variance_cutoff(self, variance_ratio):
        """
        Return the number of components whose cumulative sum explain the ratio of variance given.
        
        Arguments:
        ---------
        variance_fraction: float
            fraction of explained variance to find.
            
        Returns:
        -------
        n_components: int
            number of components that explain the given explained variance ratio
        """
        if variance_ratio > 1.0 or variance_ratio < 0.0: 
            raise Exception(""" variance_ratio must be between 0.0 and 1.0. """)
        
        # Init fractions
        fraction = 0 
        n_components = 0
        
        # Add principal components until their cumulative sum 
        while fraction < variance_ratio:
            n_components += 1
            # Calculate cumulative variance
            fraction = self.cumulative_variance(n_components)
        
        # Return the number of components
        return n_components 
        

class EpistasisPCA(BaseModel):

    def __init__(self, wildtype, genotypes, phenotypes, 
        order=1, 
        stdeviations=None, 
        log_transform=False, 
        mutations=None, 
        n_replicates=1, 
        n_components=None,
        model_type="local"):
        
        """ 
        
        Principal component analysis of the genotype-phenotype map.
        This module uses Scikit-learn's PCA class to perform the transformation.

        This model performs a transformation on the mutational coordinates
        dummy variable matrix, thereby finding the linear combination of mutations
        and their epistatic coordinates that best describe the variation in
        phenotype.
            
            
        """
        super(EpistasisPCA, self).__init__(wildtype, genotypes, phenotypes, stdeviations, log_transform, mutations=mutations, n_replicates=n_replicates)

        self.order = order
        self.n_components = n_components
        self.model = PCA(n_components=n_components)

        # Construct the Interactions mapping -- Interactions Subclass is added to model
        self._construct_interactions()
        
        # Select type of model
        self.model_type = model_type        
        model_types = {"local":  {"1": 1, "0": 0}, "global": {"1": -1, "0": 1}}
        encoding = model_types[self.model_type]

        # Construct a dummy variable matrix
        self.X = (self.Binary.phenotypes*generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding=encoding).T).T
        
        # Add statistics object
        self.Stats = PCAStats(self)

    def fit(self):
        """ 
        Estimate the principal components (i.e. maximum coordinates in phenotype variation.). 
        """
        self.X_new = self.model.fit_transform(self.X[:,:])
        self.explained_variance_ratio = self.model.explained_variance_ratio_
        self.components = self.model.components_