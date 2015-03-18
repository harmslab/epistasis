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

def label_to_key(label):
    """ Take an interaction label list and convert to a string for dictionary 
        key (e.g. [1,2,6,8] = '1,2,6,8' )
    """
    return ",".join([str(i) for i in label])
    
def interaction_keyed_map(labels, values):
    """ Take ordered interaction labels ([1,2,3]) and values and create a dictionary
        mapping interaction key (i.e. '1,2,3') to interaction value
    """
    interaction_map = dict()
    for l in range(len(labels)):
        interaction_map[label_to_key(labels[l])] = values[l]
    return interaction_map
    
def nth_order_map(full_map, length, order):
    # Built list of labels with given order
    labels = [",".join(map(str,i)) for i in it.combinations(range(1,length+1), order)]
    nth_map = dict()
    for l in labels:
        nth_map[l] = full_map[l]
    return nth_map
    
# ------------------------------------------------------------
# Unique Epistasis Functions
# ------------------------------------------------------------   

def hadamard_weight_vector(genotypes):
    l = len(genotypes)
    n = len(genotypes[0])
    weights = np.zeros((l, l), dtype=float)
    for g in range(l):
        epistasis = float(genotypes[g].count("1"))
        weights[g][g] = ((-1)**epistasis)/(2**(n-epistasis))    
    return weights    

# ------------------------------------------------------------
# Epistasis Mapping Classes
# ------------------------------------------------------------


class EpistasisMap(object):
    
    def __init__(self, gpm, log_phenotypes=True):
        """ An empty placeholder class for easy navigating/mapping Epistasis 
            in a genotype-phenotype map. This class does not determine epistasis;
            it merely t
        """
        self.genotypes = gpm.gp_mapping.keys()
        self.phenotypes = gpm.gp_mapping.values()
        self.phenotype_errors = np.array(gpm.error_mapping.values())
        self.length = len(self.genotypes[0])
        self.log_phenotypes = log_phenotypes
        self.interaction_labels = build_interaction_labels(self.genotypes)
        self.interaction_values = None
        self.interaction_errors = None
        self.interaction_mapping = {}
        self.error_mapping = {}
        self.epistasis_dataframe = None
        
        # Log-transform phenotypes if specified
        if log_phenotypes is True:
            self.Y = np.log(self.phenotypes)
        else:
            self.Y = self.phenotypes
        
    def nth_order(self, order):
        """ Returns dictionary of all interactions with specified order. """
        return nth_order_map(self.interaction_mapping, self.length, order)
    
    def nth_error(self, order):
        return nth_order_map(self.error_mapping, self.length, order)
        
    def create_interaction_map(self):
        """ Take ordered interaction labels ([1,2,3]) and values and create a dictionary
            mapping interaction key (i.e. '1,2,3') to interaction value
        """
        self.interaction_mapping = interaction_keyed_map(
                                self.interaction_labels,
                                self.interaction_values)
        return self.interaction_mapping
        
    def create_error_map(self):
        """ Take ordered interaction labels ([1,2,3]) and values for error and create a dictionary
            mapping interaction key (i.e. '1,2,3') to interaction value
        """
        self.error_mapping = interaction_keyed_map(
                                self.interaction_labels, 
                                self.interaction_errors)
        
    def build_interaction_dataframe(self, include_error=False):
        """ Build a pandas dataframe of the interactions/error. """
        if include_error is True:
            uncertainty = self.error_mapping
            interactions = self.interaction_mapping
        
            data = list()
            for i in interactions:
                data.append([i,interactions[i],uncertainty[i],len(i)])

            self.epistasis_dataframe = pd.DataFrame(data=data,
                        columns=["Interactions", "Mean Value", "Standard Deviations", "Ordering"])        
        else:
            interactions = self.interaction_mapping
        
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
            ax.bar(range(len(y)), y, 0.9, alpha=0.4, align="center", **kwargs)
        else:
            if self.epistasis_dataframe is None:
                self.build_interaction_dataframe(include_error=True)
        
            y = self.epistasis_dataframe["Mean Value"]
            xlabels = self.epistasis_dataframe["Interactions"]
            yerr = self.epistasis_dataframe["Standard Deviations"]
            ax.bar(range(len(y)), y, 0.9, yerr=sigmas*yerr, alpha=0.4, align="center",**kwargs)
            
        plt.xticks(range(len(y)), np.array(xlabels), rotation="vertical")
        ax.set_xlabel("Interaction term", fontsize=16)
        ax.set_ylabel("Interaction Value", fontsize=16) 
        ax.set_title(title, fontsize=20)
        ax.axis("tight")
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
        self.create_interaction_map()
        return self.interaction_values
        
    def estimate_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        self.interaction_errors = np.sqrt(np.dot(self.X, self.phenotype_errors**2))
        self.create_error_map()
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
        self.weight_vector = hadamard_weight_vector(gpm.genotypes)
        self.X = sp.linalg.hadamard(2**self.length)
        self.interaction_values = None
        
    def estimate_interactions(self):
        """ Estimate the values of all epistatic interactions using the hadamard
        matrix transformation.
        """
        self.interaction_values = np.dot(self.weight_vector,np.dot(self.X, self.Y))
        self.create_interaction_map()
        return self.interaction_values
        
    def estimate_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        self.interaction_errors = np.dot(self.weight_vector, np.sqrt(np.dot(abs(self.X), self.phenotype_errors**2)))
        self.create_error_map()
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
        self.error_model = LinearRegression(fit_intercept=False)
        self.r_squared = None
        
        
    def estimate_interactions(self):
        """ Estimate the values of all epistatic interactions using the expanded
            mutant cycle method to any order<=number of mutations.
        """
        self.regression_model.fit(self.X, self.Y)
        self.r_squared = self.regression_model.score(self.X, self.Y)
        self.interaction_values = self.regression_model.coef_
        self.create_interaction_map()
        return self.interaction_values
        
        
    def estimate_error(self):
        """ Estimate the error of each epistatic interaction by standard error 
            propagation of the phenotypes through the model.
        """
        self.interaction_errors = np.empty(len(self.interaction_labels), dtype=float)
        for i in range(len(self.interaction_labels)):
            n = len(self.interaction_labels[i])
            self.interaction_errors[i] = np.sqrt(n*self.phenotype_errors[i]**2)
        self.create_error_map()
        return self.interaction_errors
        