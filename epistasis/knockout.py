# Knockout epistatic interactions from a space from Epistasis map
import numpy as np

class KnockoutModel(object):
    
    def __init__(self, model):
        """ Generate a Genotype-Phenotype Map from a model after knocking out epistatic interactions. """
        self._X = model.X
        self._genotypes = model.Binary.genotypes
        self._phenotypes = model.Binary.phenotypes
        self._interactions = model.Interactions.keys
        self._indices = np.arange(len(self._interactions))
        self._pheno_indices = model.Binary.indices
        if model.log_transform:
            self._errors = model.Interactions.errors[0]
        else:
            self._errors = model.Interactions.errors
        self._values = model.Interactions.values
    
    @property
    def epistasis2index(self):
        """ Return epistasis to index dict."""
        return dict(zip(self._interactions, self._indices))
        
    def nonzero_terms(self, sigmas=2):
        """ Return all non zero epistatic interactions. """
        values = abs(self._values)
        errors = abs(self._errors)
        lowest = values - sigmas*errors
        terms = dict()
        for i in range(len(lowest)):
            if lowest[i] > 0:
                terms[self._interactions[i]] = self._values[i]
        return terms
        
    def generate_data(self, knockouts):
        """ Generate knockout data. """
        values = np.array(self._values)
        epistasis2index = self.epistasis2index
        for k in knockouts:
            values[epistasis2index[k]] = 0.0
        phenotypes = 10**(np.dot(self._X, values))
        return self._genotypes, phenotypes
    