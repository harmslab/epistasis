__doc__ = """ Submodule with nonlinear epistasis models for estimating epistatic interactions in nonlinear genotype-phenotype maps."""

import numpy as np
import itertools as it
from scipy.optimize import curve_fit, basinhopping, minimize
import matplotlib.pyplot as plt

from epistasis.stats import r_squared
from epistasis.base import BaseModel
from epistasis.utils import label_to_lmfit
from epistasis.regression_ext import generate_dv_matrix

# ------------------------------------------
# LMFIT imports
# ------------------------------------------

import lmfit

# ------------------------------------------
# Possible functions for nonlinear epistasis
# ------------------------------------------

def two_state_func(x, *args):
    """ Two state boltzmann weighted nonlinear function."""
    beta = args[-1]
    params = args[:-1]
    length = len(params)
    params1 = np.array(params[0:int(length/2)])
    params2 = np.array(params[int(length/2):])
    X1 = np.exp(-np.dot(x[0:int(length/2),:].T, params1)/beta)
    X2 = np.exp(-np.dot(x[int(length/2):-1,:].T, params2)/beta)
    return -beta*np.log(X1 + X2)

def threshold_func(params, x, y_obs):
    """ LMFIT Threshold function. 
    
        P(p) = \theta (1 - exp(-\nu * p))
        
        where p is a high order linear epistasis model. 
        
        __Arguments__:
        
        `params` : LMFIT Parameters object
    """
    # Check parameters
    if isinstance(params, lmfit.Parameters) is not True:
        raise Exception(""" params must be LMFIT's Parameters object. """) 
    # Assign parameters
    paramdict = params.valuesdict()
    theta = paramdict["theta"]
    del paramdict["theta"]
    nu = paramdict["nu"]
    del paramdict["nu"]
    interactions = list(paramdict.values())
    
    ###########   Model to minimize   #############
    # ln[ln(\theta) - ln(\theta - P)] = ln(\nu) + ln(p)
    y_pred = np.log(nu)+np.dot(x,interactions)          # right side
    P = np.log(np.log(theta) - np.log(theta-y_obs))     # left side
    
    # Residuals to minimize
    residuals = P - y_pred
    return residuals


# -------------------------------------------------------------------------------
# Classes for NonLinear modelling
# -------------------------------------------------------------------------------

class NonlinearEpistasisModel(BaseModel):
        
    def __init__(self, wildtype, genotypes, phenotypes, function, parameters, x, errors=None, log_transform=False, mutations=None):
        """ Fit a nonlinear epistasis model to a genotype-phenotype map. The function and parameters must
            specified prior. 
        
            Uses Scipy's curve_fit method.
            
            __Arguments__:
            
            `wildtype` [str] : Wildtype genotype. Wildtype phenotype will be used as reference state.
            
            `genotypes` [array-like, dtype=str] : Genotypes in map. Can be binary strings, or not.
            
            `phenotypes` [array-like] : Quantitative phenotype values
            
            `function` [callable] : Function to fit the linear model. Must have the arguments that correspond to the parameters.
            
            `x` [2d array] : Array of dummy variables for epistasis model fitting (use `generate_dv_matrix` in regression_ext)
            
            `parameters` [dict] : interaction keys with their values expressed as lists.
            
            `errors` [array-like] : List of phenotype errors.
            
            `log_transform` [bool] : If True, log transform the phenotypes.
        """
        # Populate Epistasis Map
        super(NonlinearEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, errors, log_transform, mutations=mutations)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        if parameters is not None:
            self._construct_interactions()      
            self.Interactions.labels = parameters
        else:
            raise Exception("""Need to specify the model's `order` argument or manually 
                                list model parameters as `parameters` argument.""")
   
        self.X = x
        self.function = function
    
    def fit(self, p0=None, *args):
        """ Fit the nonlinear function """
        
        # Try initial guess 
        if p0 == None:
            p0 = 0.1*np.ones(len(self.Interactions.labels), dtype=float)

        self.results = minimize(self.function,
                                p0,
                                args=(self.X, self.Binary.phenotypes),
                                method="L-BFGS-B",
                                options={ "maxiter":100000, "ftol":1.e-25})
                                
        self.Interactions.values = self.results.x[:]
            
            
class LMFITEpistasisModel(BaseModel):
    
    def __init__(self, wildtype, genotypes, phenotypes, function, x, parameters, errors=None, log_transform=False, mutations=None):
        """ Fit a nonlinear epistasis model to a genotype-phenotype map. The function and parameters must
            specified prior. 
        
            Uses LMFIT's pa
            
            __Arguments__:
            
            `wildtype` [str] : Wildtype genotype. Wildtype phenotype will be used as reference state.
            
            `genotypes` [array-like, dtype=str] : Genotypes in map. Can be binary strings, or not.
            
            `phenotypes` [array-like] : Quantitative phenotype values
            
            `function` [callable] : Function to fit the linear model. Must have the arguments that correspond to the parameters.
            
            `x` [2d array] : Array of dummy variables for epistasis model fitting (use `generate_dv_matrix` in regression_ext)
            
            `parameters` [lmfit.Parameters class] : Parameters as specified by lmfit
            
            `errors` [array-like] : List of phenotype errors.
            
            `log_transform` [bool] : If True, log transform the phenotypes.
        """
        # Populate Epistasis Map
        super(LMFITEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, errors, log_transform, mutations=mutations)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        if parameters is not None:
            self._construct_interactions()
            self.parameters = parameters
            params = parameters.valuesdict()    
            self.Interactions.keys = list(params.keys())
            self.Interactions.labels = list(params.keys())
            self.Interactions.values = list(params.values())
        else:
            raise Exception("""Need to specify the model's `order` argument or manually 
                                list model parameters as `parameters` argument.""")
   
        self.X = x
        self.function = function
        
        
    def fit(self, method="leastsq", **kwargs):
        """ Fit the nonlinear function """
        self.minimizer = lmfit.minimize(self.function, self.parameters, args=(self.X, self.Binary.phenotypes,), method=method, **kwargs)
        self.Interactions.values = np.array(list(self.minimizer.params.valuesdict().values()))
        
    def update_param_value(self, **kwargs):
        """ Update a LMFIT parameters guess value and propagate through mapping. """
        mapping = self.Interactions.key2index
        for k in kwargs:
            self.parameters[k].value = kwargs[k]
            self.parameters[k].vary = False
            self.Interactions.values[mapping[k]] = kwargs[k]
            
        
class GlobalNonlinearEpistasisModel(BaseModel):
    
    def __init__(self, wildtype, genotypes, phenotypes, function, x, parameters, errors=None, log_transform=False, mutations=None):
        """ Fit a nonlinear epistasis model to a genotype-phenotype map. The function and parameters must
            specified prior. 
        
            Uses Scipy's basinhopping method.
            
            __Arguments__:
            
            `wildtype` [str] : Wildtype genotype. Wildtype phenotype will be used as reference state.
            
            `genotypes` [array-like, dtype=str] : Genotypes in map. Can be binary strings, or not.
            
            `phenotypes` [array-like] : Quantitative phenotype values
            
            `function` [callable] : Function to fit the linear model. Must have the arguments that correspond to the parameters.
            
            `x` [2d array] : Array of dummy variables for epistasis model fitting (use `generate_dv_matrix` in regression_ext)
            
            `parameters` [dict] : interaction keys with their values expressed as lists.
            
            `errors` [array-like] : List of phenotype errors.
            
            `log_transform` [bool] : If True, log transform the phenotypes.
        """
        # Populate Epistasis Map
        super(GlobalNonlinearEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, errors, log_transform, mutations=mutations)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        if parameters is not None:
            self._construct_interactions()      
            self.Interactions.keys = list(parameters.values())
            self.Interactions.labels = list(parameters.values())
        else:
            raise Exception("""Need to specify the model's `order` argument or manually 
                                list model parameters as `parameters` argument.""")
   
        self.X = x
        self.function = function
        
    def fit(self, p0=None):
        """ Fit the nonlinear function """
        
        # Try initial guess 
        if p0 == None:
            p0 = 0.1*np.ones(len(self.Interactions.labels), dtype=float)
        
        
        results = basinhopping(self.function, p0, niter=1000,
                                minimizer_kwargs={"args": (self.Binary.phenotypes,self.X.T)})

        self.Interactions.values = results.x
        # Setting error if covariance was estimated, else pass.
        try:
            self.errors = cov[:]
        except:
            pass


# -------------------------------------------------------------------------------
# Examples of nonlinear models
# -------------------------------------------------------------------------------

class ThresholdingEpistasisModel(LMFITEpistasisModel, BaseModel):
    
    def __init__(self, wildtype, genotypes, phenotypes, order, errors=None, log_transform=False, mutations=None):
        """ """
        # Construct initial base map
        BaseModel.__init__(self, wildtype, genotypes, phenotypes, errors=errors, log_transform=log_transform, mutations=mutations)
        
        # Construct the linear epistasis model
        self.order = order
        self._construct_interactions()
        
        # Construct LMFIT parameters properly
        labels = self.Interactions.labels
        x = generate_dv_matrix(self.Binary.genotypes, labels)
        keys = [label_to_lmfit(l) for l in labels]

        params = lmfit.Parameters()
        params.add("K0", value=0, vary=False)
        for i in range(1,len(keys)):
            params.add(keys[i], value=1, vary=True)

        params.add("theta", value=100, vary=True)
        params.add("nu", value=1, vary=True)
        
        # Construct nonlinear modelling map
        LMFITEpistasisModel.__init__(self, wildtype, genotypes, phenotypes, 
                                            threshold_func, 
                                            x, 
                                            params, 
                                            errors=errors, 
                                            log_transform=log_transform, 
                                            mutations=mutations)
        
        
    def fit(self, **kwargs):
        """ Fit nonlinear thresholding epistasis model."""        
        # Use LMFIT fit method above.
        self.update_param_value(**kwargs)
        LMFITEpistasisModel.fit(self)
        
    def linear_phenotypes(self):
        """ Get phenotypes after removing thresholding effect. """
        return np.exp(np.dot(self.X, self.Interactions.values[:-2]))
        
    def plot_interactions(self, sigmas=0, title="Epistatic interactions", string_labels=False, ax=None, color='b', figsize=[6,4]):
        """ """
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=figsize)
        else:
            fig = ax.get_figure()
    
        y = self.Interactions.values[:-2]
        if string_labels is True:
            xtick = self.Interactions.genotypes[:-2]
        else:
            xtick = self.Interactions.keys[:-2]
            xlabel = "Interaction Indices"
    
        # plot error if sigmas are given.
        if sigmas == 0:
            ax.bar(range(len(y)), y, 0.9, alpha=0.4, align="center", color=color) #, **kwargs)
        else:
            yerr = self.Interactions.errors[:-2]
            ax.bar(range(len(y)), y, 0.9, yerr=sigmas*yerr, alpha=0.4, align="center", color=color) #,**kwargs)
    
        # vertically label each interaction by their index
        plt.xticks(range(len(y)), np.array(xtick), rotation="vertical", family='monospace',fontsize=7)
        ax.set_ylabel("Interaction Value", fontsize=14) 
        try:
            ax.set_xlabel(xlabel, fontsize=14)
        except:
            pass
        ax.set_title(title, fontsize=12)
        ax.axis([-.5, len(y)-.5, -max(abs(y)), max(abs(y))])
        ax.hlines(0,0,len(y), linestyles="dashed")
        return fig, ax
        
        