# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
from .core.ensemble_mapping import EnsembleMap
    
# ------------------------------------------------------------
# EnsembleModel object for observing average of ensemble of 
# epistasis models.
# ------------------------------------------------------------

class EnsembleModel(EnsembleMap):
    
    def __init__(self, model, genotypes, phenotypes, order=None, phenotype_errors=None, log_phenotypes=True):
        """ Use an ensemble of epistasis models from different reference states to calculate 
            average properties about space (not dependent on starting state).
        """
        self.model = model
        self.genotypes = genotypes
        self.phenotypes = phenotypes
        self.log_phenotypes = log_phenotypes
        self.phenotype_errors = phenotype_errors
        self.log_phenotypes = log_phenotypes 
        
        # set keywords arguments
        if order is None:
            self.order = len(self.genotypes[0])
        else:
            self.order = order

                    
    def build_ensemble(self, N, wildtypes=None):
        """ Build an ensemble of N models from different reference (wildtype) states."""
        self.N = N
        # Build wildtypes array
        if wildtypes is None:
            self.wildtypes = np.random.choice(self.genotypes, size=self.N, replace=False)
              
        for i in range(self.N):
            # Create an instance with the first wildtype
            m =  self.model(self.wildtypes[i], self.genotypes, self.phenotypes, 
                            phenotype_errors=self.phenotype_errors, 
                            log_phenotypes=self.log_phenotypes)
            # Use the model to estimate epistasis from this reference             
            m.estimate_interactions()
            self.ensemble = m.genotype2value
        
    def addto_ensemble(self):
        """ Needs to be written. """
        pass
        
    def ensemble_averages(self):
        """ Get the ensemble averages of all interaction in the space. """
        averages = dict()
        variation = dict()
        try:
            for key, value in self.ensemble.items():
                averages[key] = np.mean(value)
                variation[key] = np.std(value)
            return averages, variation
        except AttributeError:
            raise AttributeError("Must build ensemble before trying to calculate averages.")