import itertools as it
import numpy as np
import pandas as pd
from regression_ext import generate_dv_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Functions that are useful for regression
# ------------------------------------------------------------


def list_dummy_variables(length, order):
    dv = [[int(0)]]
    seq = range(1,length+1)
    for j in range(1,order+1):
        dv = dv + [list(i) for i in it.combinations(seq, j)]
    return dv

def sample_phenotypes(phenotypes, std, num_samples):
    """ Generate artificial data sampled from phenotype and percent error. """
    gen_phenotypes = list()
    gen_genotypes = list()
    for s in phenotypes:
        gen_genotypes += [s for i in range(num_samples)]
        gen_phenotypes += list(std[s] * np.random.randn(num_samples) + phenotypes[s])
    return gen_genotypes, gen_phenotypes

def generate_alpha(genotypes):
    """ Generate penalty parameter for Ridge Regressions. """
    alpha = np.empty(len(genotypes), dtype=float)
    for s in range(len(genotypes)):
        n = genotypes[s].count("1")
        alpha[s] = 2**n
    return alpha
    
def log_pheno(phenotypes):
    """ Generate a dictionary of the log(phenotype). """
    return np.log10(phenotypes)


# ------------------------------------------------------------
# Epistasis Regression Model
# ------------------------------------------------------------


class EpistasisRegression(object):
    
    def __init__(self, genotypes, phenotypes, interaction_order=2, log_phenotypes=True):
        self.genotypes = genotypes
        self.length = len(self.genotypes[0])
        self.interaction_order = interaction_order
        self.dummy_variables = list_dummy_variables(self.length, self.interaction_order)
        self.dummy_variables_matrix = generate_dv_matrix(self.genotypes, self.dummy_variables)
        self.model = LinearRegression(fit_intercept=False)
        self.error_model = LinearRegression()
        self.log_phenotypes = log_phenotypes
        self.epistasis_dataframe = None
            
        self.model_score = 0
        self.interactions = []
        
        if log_phenotypes is True:
            self.Y_vector = log_pheno(phenotypes)
        else:
            self.Y_vector = phenotypes
        
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
        
    def get_interaction_error(self):
        """ Returns a dictionary of the interaction errors from the regression. """
        error_terms = dict()
        for s in range(len(self.error_seq)):
            char = list(self.error_seq[s])
            indices = list()
            for i in range(len(char)):
                if char[i] == '1':
                    indices.append(str(i+1))
            label = ",".join(indices)
            if label == "":
                label = "0"
            error_terms[label] = self.uncertainty_window[s]
        return error_terms
    
    
    def fit(self, X=None,Y=None):
        """ Make EpistasisRegression instance learn from data. """
        if X is None or Y is None:
            self.model.fit(self.dummy_variables_matrix, self.Y_vector)
            self.model_score = self.model.score(self.dummy_variables_matrix, self.Y_vector)
        else:
            self.model.fit(X,Y)
            self.model_score = self.model.score(X,Y)
        self.interactions = self.model.coef_
        return self.model_score
        
    def fit_error(self, errors):
        """ 
        A method for creating a model that handles error propagation in intereaction terms. 
        
        Parameters:
        ----------
        error_vector: array or dict
            If array, will reuse the dummy_variable_matrix stored in this class from __init__.
            Else, a dictionary will rebuild the dummy_variable_matrix.
            
        """
        if type(errors) == dict:
            self.error_seq = errors.keys()
            error_vector = np.array(errors.values())
            X = generate_dv_matrix(self.error_seq, self.dummy_variables)
        else:
            self.error_seq = self.genotypes
            error_vector = np.array(errors)
            X = self.dummy_variables_matrix
        
        # Check if the phenotypes are log transforms to transform the error if necessary.
        
        if self.log_phenotypes is True:
            error_vector = log_pheno(1.0+error_vector)
            
        # Build the error model from linear regression.
        self.error_vector = error_vector**2
        self.error_model = LinearRegression(fit_intercept=False)
        self.error_model.coef_ = self.error_vector
        self.error_model.intercept_ = 0
        self.uncertainty_window = np.sqrt(np.array(self.error_model.predict(X)))
        
        
    def predict(self,X):
        """ Predict from epistasis model instance. """
        return self.model.predict(X)
        
    def score(self, X,Y):
        """ Score X and Y's fit to the model. """
        return self.model.score(X,Y)
        
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


class EpistasisRidgeRegression(EpistasisRegression):
    
    def __init__(self, genotypes, phenotypes, interaction_order=2):
        EpistasisRegression.__init__(self, genotypes, phenotypes, interaction_order)
        self.alphas = 1
        self.model = Ridge(alpha=self.alphas)
    