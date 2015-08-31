# -------------------------------------- -------------
# Module for Principal Component Analysis of Epistasis
# ----------------------------------------------------

from sklearn.decomposition import PCA
from epistasis.base import BaseModel
from epistasis.regression_ext import generate_dv_matrix

class EpistasisPCA(BaseModel):

    def __init__(self, wildtype, genotypes, phenotypes, order=1, errors=None, log_transform=False, mutations=None):
        """ Principal component analysis of the genotype-phenotype map. """
        super(EpistasisPCA, self).__init__(wildtype, genotypes, phenotypes, errors, log_transform, mutations=mutations)
        
        self.order = order
        self.model = PCA()
        
        # Construct the Interactions mapping -- Interactions Subclass is added to model
        self._construct_interactions()
        
        # Construct a dummy variable matrix
        self.X = (self.Binary.phenotypes*generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels).T).T
        
    def fit(self):
        """ Estimate the principal components. """
        self.X_new = self.model.fit_transform(self.X)
        self.explained_variance_ratio = self.model.explained_variance_ratio_
        self.components = self.model.components_
        