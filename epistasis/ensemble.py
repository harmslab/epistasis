import numpy as np
from .core.ensemble_mapping import EnsembleMap

class ModelEnsemble(EnsembleMap):
    
    def __init__(self, model, genotypes, phenotypes, order=None, phenotype_errors=None, log_phenotypes=True):
        """ Use an ensemble of epistasis models from different reference states to calculate 
            average properties about space (not dependent on starting state).
        """
        self.model = model
        self.genotypes = genotypes
        self.phenotypes = phenotypes
        self.log_phenotypes = log_phenotypes
        
        # set keywords arguments
        if order is None:
            self.order = len(self.genotypes[0])
        else:
            self.order = order
            
        if phenotype_errors is not None:
            self.phenotype_errors = phenotype_errors
        
    def build_ensemble(self, N, wildtypes=None):
        """ Build an ensemble of N models from different reference (wildtype) states."""
        self.N = N
        for i in range(s, )
        m =  model(wildtype, genotypes, phenotypes, phenotype_errors=self.phenotype_errors, log_phenotypes=log_phenotypes)
        m.estimate_interactions()
        self.ensemble = m.genotype2error
        
    def addto_ensemble(self):
        pass