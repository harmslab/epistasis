import itertools as it
import numpy as np
import scipy as sp
import pandas as pd
from regression_ext import generate_dv_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Functions for Model building
# ------------------------------------------------------------

def build_interaction_labels(genotypes):
    """ Create interaction labels for X Matrix. REQUIRES that genotypes are in
        ascending order by their binary value."""
    
    def find(s, ch):
        # simple function to find indices of mutations
        return sorted([i+1 for i, ltr in enumerate(s) if ltr == ch])
    
    vector = [find(g, '1') for g in genotypes]
    vector[0] = [0]
    return vector
    
def cut_interaction_labels(labels, order):
    """ Cut off interaction labels at certain order of interactions. """
    return [l for l in labels if len(l) <= order]


# ------------------------------------------------------------
# Epistasis Mapping Classes
# ------------------------------------------------------------


class EpistasisMap(object):
    
    def __init__(self, gpm, log_phenotypes=True):
        """ An empty placeholder class for easy navigating/mapping Epistasis 
            in a genotype-phenotype map. This class does not determine epistasis;
            it merely t
        """
        self.genotypes = gpm.phenotype_dict.keys()
        self.phenotypes = gpm.phenotype_dict.values()
        self.phenotype_errors = gpm.error_dict.values()
        self.length = len(self.genotypes[0])
        self.log_phenotypes = log_phenotypes
        self.epistasis_dataframe = None
        self.interaction_labels = build_interaction_labels(self.genotypes)
        
        # Make Y_vector with phenotypes logged
        if log_phenotypes is True:
            self.Y = np.log(self.phenotypes)
        else:
            self.Y = self.phenotypes
        
    def get_single_term(self, sites):
        """ Returns the value of interaction term. """
        if len(sites) > self.interaction_order:
            raise("Order of interation is higher than the model's regression interactions. ")
        sites.sort()
        location = self.dummy_variables.index(sites)
        self.interactions[location]
        
    def get_order_terms(self, order):
        """ Returns dictionary of all interactions with specified order. """
        terms = dict()
        for dv in range(len(self.dummy_variables)):
            dummy = self.dummy_variables[dv]
            if len(dummy) == order:
                label = ",".join([str(i) for i in dummy])
                terms[label] = self.interactions[dv]
        return terms
        
    def get_interactions(self):
        """ Returns a dictionary of interactions from regression. """
        terms = dict()
        for dv in range(len(self.dummy_variables)):
            dummy = self.dummy_variables[dv]
            label = ",".join([str(i) for i in dummy])
            terms[label] = self.interactions[dv]
        return terms
        
    def build_interaction_dataframe(self, include_error=False):
        """ Build a pandas dataframe of the interactions/error. """
        if include_error is True:
            uncertainty = self.get_interaction_error()
            interactions = self.get_interactions()
        
            data = list()
            for i in interactions:
                data.append([i,interactions[i],uncertainty[i],len(i)])

            self.epistasis_dataframe = pd.DataFrame(data=data,
                        columns=["Interactions", "Mean Value", "Standard Deviations", "Ordering"])        
        else:
            interactions = self.get_interactions()
        
            data = list()
            for i in interactions:
                data.append([i,interactions[i],len(i)])

            self.epistasis_dataframe = pd.DataFrame(data=data,
                        columns=["Interactions", "Mean Value", "Ordering"])

        self.epistasis_dataframe = self.epistasis_dataframe.sort("Ordering")

    def plot_interaction_dataframe(self, title, sigmas=3, figsize=[15,7],**kwargs):
        """ Plot the interactions sorted by their order. 
            
        Parameters:
        ----------
        title: str
            The title for the plot.
        sigmas: 
            Number of sigmas to represent the errorbars. If 0, no error bars will be included.
        """
        fig, ax = plt.subplots(1,1, figsize=figsize)
        
        if sigmas == 0:
            if self.epistasis_dataframe is None:
                self.build_interaction_dataframe()
            y = self.epistasis_dataframe["Mean Value"]
            xlabels = self.epistasis_dataframe["Interactions"]
            ax.plot(range(len(y)), y, linewidth=1.1, **kwargs)
        else:
            if self.epistasis_dataframe is None:
                self.build_interaction_dataframe(include_error=True)
        
            y = self.epistasis_dataframe["Mean Value"]
            xlabels = self.epistasis_dataframe["Interactions"]
            yerr = self.epistasis_dataframe["Standard Deviations"]
            ax.errorbar(range(len(y)), y, yerr=sigmas*yerr, linewidth=1.1, **kwargs)
            
        plt.xticks(range(len(y)), np.array(xlabels), rotation="vertical")
        ax.set_xlabel("Interaction term", fontsize=16)
        ax.set_ylabel("Interaction Value", fontsize=16) 
        ax.set_title(title, fontsize=20)
        ax.axis("tight")
        ax.grid('on')
        ax.hlines(0,0,len(y), linestyles="dashed")
        
        return fig, ax    

class LocalEpistasisMap(EpistasisMap):
        
    def __init__(self, gpm, log_phenotypes=True):
        """ Create a map of the local epistatic effects using expanded mutant 
            cycle approach.
            
            i.e.
            Phenotype = K_0 + sum(K_i) + sum(K_ij) + sum(K_ijk) + ...
            
            Parameters:
            ----------
            geno_pheno_dict: OrderedDict
                Dictionary with keys=ordered genotypes by their binary value, 
                mapped to their phenotypes.
            log_phenotypes: bool (default=True)
                Log transform the phenotypes for additivity.
        """
        # Inherit EpistasisMap
        EpistasisMap.__init__(self, gpm, log_phenotypes)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.X = generate_dv_matrix(self.genotypes, self.interaction_labels)
        self.X_inv = np.linalg.inv(self.X)
        self.interaction_values = None
        
    def estimate_interactions(self):
        """ Estimate the values of all epistatic interactions using the expanded
            mutant cycle method to order=number_of_mutations.
        """
        self.interaction_values = np.dot(self.X_inv, self.Y)
        return self.interaction_values
        
    def estimate_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        self.interaction_errors = np.dot(self.X, self.phenotype_errors)
        return self.interaction_errors
    
    
class GlobalEpistasisMap(EpistasisMap):
    
    def __init__(self, gpm, log_phenotypes=True):
        """ Create a map of the global epistatic effects using Hadamard approach.
            This is the related to LocalEpistasisMap by the discrete Fourier 
            transform of mutant cycle approach. 
        """
        # Inherit EpistasisMap
        EpistasisMap.__init__(self, gpm, log_phenotypes)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.X = sp.linalg.hadamard(2**self.length)
        self.interaction_values = None
        
    def estimate_interactions(self):
        """ Estimate the values of all epistatic interactions using the hadamard
        matrix transformation.
        """
        self.interaction_values = np.dot(self.X, self.Y)
        return self.interaction_values
        
    def estimate_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        self.interaction_errors = np.dot(self.X, self.phenotype_errors)
        return self.interaction_errors
    
class ProjectedEpistasisMap(EpistasisMap):
    
    def __init__(self, gpm, regression_order, log_phenotypes=True):
        """ Create a map from local epistasis model projected into lower order
            order epistasis interactions. Requires regression to estimate values. 
        """
        # Inherit EpistasisMap
        EpistasisMap.__init__(self, gpm, log_phenotypes)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        self.regression_order = regression_order
        self.interaction_labels = cut_interaction_labels(
                                            self.interaction_labels, 
                                            self.regression_order
                                            )
                                            
        self.X = generate_dv_matrix(self.genotypes, self.interaction_labels)
        self.interaction_values = None
        
        # Regression properties
        self.regression_model = LinearRegression(fit_intercept=False)
        self.r_squared = None
        
        
    def estimate_interactions(self):
        """ Estimate the values of all epistatic interactions using the expanded
            mutant cycle method to order=number_of_mutations.
        """
        self.regression_model.fit(self.X, self.Y)
        self.r_squared = self.regression_model.score(self.X, self.Y)
        self.interaction_values = self.regression_model.coef_
        return self.interaction_values
        
        
    def estimate_error(self):
        pass
        