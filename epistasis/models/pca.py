__doc__ = """ Principal component analysis for genotype-phenotype maps in submodule."""

# -------------------------------------- -------------
# Module for Principal Component Analysis of Epistasis
# ----------------------------------------------------

from sklearn.decomposition import PCA
from epistasis.decomposition import generate_dv_matrix
from epistasis.models.base import BaseModel

class EpistasisPCA(BaseModel):

    def __init__(self, wildtype, genotypes, phenotypes, 
                    order=1, 
                    stdeviations=None, 
                    log_transform=False, 
                    mutations=None, 
                    n_replicates=1, 
                    n_components=None,
                    model_type="local"):
        """ Principal component analysis of the genotype-phenotype map.
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

    def fit(self):
        """ Estimate the principal components (i.e. maximum coordinates in phenotype variation.). """
        self.X_new = self.model.fit_transform(self.X[:,:])
        self.explained_variance_ratio = self.model.explained_variance_ratio_
        self.components = self.model.components_
