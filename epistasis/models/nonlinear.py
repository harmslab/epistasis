__doc__ = """ Submodule with nonlinear epistasis models for estimating epistatic interactions in nonlinear genotype-phenotype maps."""

import numpy as np
from scipy.optimize import curve_fit

#from epistasis.stats import r_squared
from epistasis.decomposition import generate_dv_matrix
from epistasis.models.base import BaseModel
from epistasis.stats import pearson

# -------------------------------------------------------------------------------
# Classes for Nonlinear modelling
# -------------------------------------------------------------------------------

from epistasis.models.regression import EpistasisRegression

import inspect

# -------------------------------------------------------------------------------
# Classes for Nonlinear modelling
# -------------------------------------------------------------------------------

class Parameters:
    
    def __init__(self, **kwargs):
        """ Extra non epistasis parameters in nonlinear epistasis models. """
        self._n_params = 0
        self._mapping = {}
        for kw in kwargs:
            self._mapping[self._n_params] = kw
            setattr(self, kw, kwargs[kw])
            self._n_params += 1
            
    def set_param(self, param, value):
        """ Set attribute"""
        
        # If param is an index, get name from mappings
        if type(param) == int or type(param) == float:
            param = self._mapping[param]
        
        setattr(self, param, value)
        
    def get_params(self):
        """ Get an ordered list of the parameters"""
        return [getattr(self, self._mapping[i]) for i in range(len(self._mapping))]
            

class NonlinearStats(object):
    
    def __init__(self, model):
        self.model = model

    @property
    def score(self):
        """ Get the epistasis model score after estimating interactions. """
        return self.model._score
        
    def predict(self):
        """ Infer the phenotypes from model.

            __Returns__:

            `genotypes` [array] : array of genotypes -- in same order as phenotypes

            `phenotypes` [array] : array of quantitative phenotypes.
        """
        phenotypes = np.zeros(len(self.model.complete_genotypes), dtype=float)
        binaries = self.model.Binary.complete_genotypes
        X = generate_dv_matrix(binaries, self.model.Interactions.labels, encoding=self.model.encoding)
        
        popt = list(self.model.Interactions.values) + self.model.Parameters.get_params()
        phenotypes = self.model._wrapped_function(X, *popt)
        
        return phenotypes
        

class NonlinearEpistasisModel(EpistasisRegression):
    
    
    def __init__(self, wildtype, genotypes, phenotypes, function, 
        order=None, 
        parameters=None, 
        stdeviations=None, 
        log_transform=False, 
        mutations=None, 
        n_replicates=1, 
        model_type="local"):
        
        """
        Performs minimization on a non-linear epistasis model function.
        
        `function` must be a callable function whose first argument is a decomposition matrix.
        Other arguments are extra coefficients/parameters in the nonlinear function
        
        Example:
        -------
        
        # nonlinear function to minimize
        def func 
        
        """
        
        super(NonlinearEpistasisModel, self).__init__(wildtype, genotypes, phenotypes,
            order=order, 
            parameters=parameters, 
            stdeviations=stdeviations, 
            log_transform=log_transform, 
            mutations=mutations, 
            n_replicates=n_replicates, 
            model_type=model_type)
        
        # Set the nonlinear function    
        self.function = function
        
        # Get the parameters from the nonlinear function argument list
        function_sign = inspect.signature(self.function)
        parameters = list(function_sign.parameters.keys())
        
        # Check that the first argument is epistasis
        if parameters[0] != "x":
            raise Exception(""" First argument of the nonlinear function must be `x`. """)
        else:
            # Build kwargs dict with all parameters set to zero as default.
            parameters_kw = dict([(p, 0) for p in parameters[1:]])
            
        # Construct parameters object
        self.Parameters = Parameters(**parameters_kw)
        self.Stats = NonlinearStats(self)
        
        
    def _nonlinear_function_wrapper(self, function):
        """ 
            Nonlinear function wrapper adding epistasis as an argument to user defined function
        """
        
        def inner(*args):
            """ 
                Convert the user defined function to a new function which fits epistasis
                coefficients as well
            """
            
            ###### Deconstruct the arguments in user's function
            # X is the first argument
            x = args[0]
            
            # The epistasis coefficients are the next set of arguments
            n_coeffs = len(self.Interactions.labels)
            betas = args[1:1+n_coeffs]

            # The user defined arguments from nonlinear function are the final set.
            other_args = args[n_coeffs+1:]

            return function(np.dot(x,betas), *other_args)

        return inner


    def fit(self, guess=None, **kwargs):
        """ """
        # Create an initial guess array
        if guess is None:
            guess = np.ones(len(self.Interactions.labels) + self.Parameters._n_params)
                    
        # Wrap user function to add epistasis parameters
        self._wrapped_function = self._nonlinear_function_wrapper(self.function)
        
        # Curve fit the data using a nonlinear least squares fit
        popt, pcov = curve_fit(self._wrapped_function, self.X, self.phenotypes, p0=guess, **kwargs)
        
        # Set the Interaction values
        self.Interactions.values = popt[0:len(self.Interactions.labels)]
        
        # Set the other parameters from nonlinear function to fit results
        for i in range(0, self.Parameters._n_params):
            self.Parameters.set_param(i, popt[len(self.Interactions.labels)+i])
            
        y_pred = self._wrapped_function(self.X, *popt)
        
        self._score = pearson(self.phenotypes, y_pred)