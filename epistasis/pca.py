from sklearn.decomposition import PCA
from epistasis.models import GenericModel    
from epistasis.regression_ext import generate_dv_matrix

class EpistasisPCA(GenericModel):
    
    def __init__(self, wildtype, genotypes, phenotypes, regression_order=1, phenotype_errors=None, log_phenotypes=False):
        """ Principal component analysis of the genotype-phenotype map. """
        super(EpistasisPCA, self).__init__(wildtype, genotypes, phenotypes, phenotype_errors, log_phenotypes)
        self.order = regression_order
        self.model = PCA()
        self.X = (self.phenotypes*generate_dv_matrix(self.bits, self.interaction_labels).T).T
        
    def estimate_components(self):
        """ Estimate the principal components. """
        self.model.fit(self.X)
        self.explained_variance_ratio = self.model.explained_variance_ratio_
        self.components = self.model.components_
        return self.explained_variance_ratio