__doc__ = """ Principal component analysis for genotype-phenotype maps in submodule."""

# -------------------------------------- -------------
# Module for Principal Component Analysis of Epistasis
# ----------------------------------------------------

from sklearn.decomposition import PCA
from epistasis.decomposition import generate_dv_matrix
from epistasis.models.regression import EpistasisRegression

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
        

class EpistasisPCA(EpistasisRegression):

    def __init__(self, wildtype, genotypes, phenotypes, 
        order=1,
        n_components=None, 
        stdeviations=None, 
        log_transform=False, 
        mutations=None, 
        n_replicates=1, 
        model_type="local",
        coordinate_type="epistasis"):
        
        """ 
        
        Principal component analysis of the genotype-phenotype map.
        This module uses Scikit-learn's PCA class to perform the transformation.

        This model performs a transformation on the mutational coordinates
        dummy variable matrix, thereby finding the linear combination of mutations
        and their epistatic coordinates that best describe the variation in
        phenotype.
            
            
        Arguments:
        ---------
        wildtype: str
            Wildtype or ancestral genotype
        genotypes: array-like of str
            Array of genotypes
        phenotypes: array-like of floats
            Array of phenotypes
        order: int
            Order of epistasis for decomposition matrix
        stdeviations: array-like of floats [default=None]
            Standard deviations of the phenotype
        log_transform: bool [default = False]
            If True, log transform the phenotype
        mutations: dict [default=None]
            A mapping dictionary of mutations at each site
        n_replicates: int
            Number of replicate measurements of each phenotype
        n_components: int [default=None]
            Number of PCA components to include for model
        model_type: str [default='local']
            If 'local', use LocalEpistasisModel decomposition. If 'global', use GlobalEpistasisModel decomposition.
        coordinate_type: str [default='epistasis']
            If 'epistasis', project epistasis parameters onto decomposition matrix. If 'phenotypes', project
            phenotypes onto decomposition matrix.

        """
        # Inherent parent class (Epistasis Regression)
        super(EpistasisPCA, self).__init__(wildtype, genotypes, phenotypes, 
            order=order,
            stdeviations=stdeviations, 
            log_transform=log_transform,
            mutations=mutations, 
            n_replicates=n_replicates,
            model_type=model_type)
            
        self.order = order
        self.n_components = n_components
        self.model = PCA(n_components=n_components)

        # Construct the Interactions mapping -- Interactions Subclass is added to model
        self._construct_interactions()
        
        # Construct a dummy variable matrix based on user preferences
        if coordinate_type == "epistasis":
            # Must fit space with regression first, then use those coordinates
            super(EpistasisPCA, self).fit() 
            self.X = self.X * self.Interactions.values
            
        elif coordinate_type == "phenotypes":
            self.X = self.X * self.Binary.phenotypes
        
        # Add statistics object
        self.Stats = PCAStats(self)


    def fit(self):
        """ 
        Estimate the principal components (i.e. maximum coordinates in phenotype variation.). 
        """
        self.X_new = self.model.fit_transform(self.X[:,:])
        self.explained_variance_ratio = self.model.explained_variance_ratio_
        self.components = self.model.components_