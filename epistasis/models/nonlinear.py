__doc__ = """ Submodule with nonlinear epistasis models for estimating epistatic interactions in nonlinear genotype-phenotype maps."""

import inspect
import numpy as np
from scipy.optimize import curve_fit

from epistasis.decomposition import generate_dv_matrix
from epistasis.stats import pearson
from epistasis.models.regression import EpistasisRegression
from epistasis.plotting import NonlinearPlotting

# decorators for catching errors
from seqspace.utils import ipywidgets_missing
from seqspace.plotting import mpl_missing

# Try to import ipython specific tools
try:
    import ipywidgets
except ImportError:
    pass

# -------------------------------------------------------------------------------
# Classes for Nonlinear modelling
# -------------------------------------------------------------------------------

class Parameters:
    
    def __init__(self, params):
        """ Extra non epistasis parameters in nonlinear epistasis models. 
        
            Sets the params to 0 initially
        """
        self._param_list = params
        self.n = len(self._param_list)
        self._mapping, self._mapping_  = {}, {}
        
        for i in range(self.n):
            setattr(self, self._param_list[i], 0)
            self._mapping_[i] = self._param_list[i]
            self._mapping[self._param_list[i]] = i
            
    def _set_param(self, param, value):
        """ Set Parameter value. 
        
            Method is not exposed to user.
            
            `param` can be either the name of the parameter or its index in this object.
        """
        
        # If param is an index, get name from mappings
        if type(param) == int or type(param) == float:
            param = self._mapping_[param]
        
        setattr(self, param, value)
        
    def get_params(self):
        """ Get an ordered list of the parameters."""
        return [getattr(self, self._mapping_[i]) for i in range(len(self._mapping_))]
            

class NonlinearStats(object):
    
    def __init__(self, model):
        self._model = model

    @property
    def score(self):
        """ Get the epistasis model score after estimating interactions. """
        return self._model._score
        
    def predict(self):
        """ Infer the phenotypes from model.

            __Returns__:

            `genotypes` [array] : array of genotypes -- in same order as phenotypes

            `phenotypes` [array] : array of quantitative phenotypes.
        """
        phenotypes = np.zeros(len(self._model.complete_genotypes), dtype=float)
        binaries = self._model.Binary.complete_genotypes
        X = generate_dv_matrix(binaries, self._model.Interactions.labels, encoding=self._model.encoding)
        
        popt = list(self._model.Interactions.values) + self._model.Parameters.get_params()
        phenotypes = self._model._wrapped_function(X, *popt)
        
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
            
        # Construct parameters object
        self.Parameters = Parameters(parameters[1:])
        self.Stats = NonlinearStats(self)
        
        # Add a plotting object if matplotlib exists
        try:
            self.Plot = NonlinearPlotting(self)
        except Warning:
            pass
        
        
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
        """ 
            Fit using nonlinear least squares regression.
            
            Arguments:
            ---------
            guess: array of guesses
            
            kwargs are passed directly to scipy's curve_fit function
        """
        # Create an initial guess array
        if guess is None:
            guess = np.ones(len(self.Interactions.labels) + self.Parameters.n)
                    
        # Wrap user function to add epistasis parameters
        self._wrapped_function = self._nonlinear_function_wrapper(self.function)
        
        # Curve fit the data using a nonlinear least squares fit
        popt, pcov = curve_fit(self._wrapped_function, self.X, self.phenotypes, p0=guess, **kwargs)
        
        # Set the Interaction values
        self.Interactions.values = popt[0:len(self.Interactions.labels)]
        
        # Set the other parameters from nonlinear function to fit results
        for i in range(0, self.Parameters.n):
            self.Parameters._set_param(i, popt[len(self.Interactions.labels)+i])
            
        y_pred = self._wrapped_function(self.X, *popt)
        
        self._score = pearson(self.phenotypes, y_pred)
    
    @ipywidgets_missing
    def fit_widget(self, show_stats=True, **kwargs):
        """
            Simple IPython widget for trying initial guesses of the nonlinear parameters.
        
            This works by, first, fitting the coefficients using a linear epistasis model as initial
            guesses (along with user defined kwargs) for the nonlinear model. 
            
            kwargs should be ranges of guess values for each parameter. They are are turned into 
            slider widgets for varying these guesses easily. The kwarg needs to match the name of
            the parameter in the nonlinear fit.
            
        """
        # Build fitting method to pass into widget box
        def fitting(**kwargs):
            """ Callable to be controlled by widgets. """
            # Fit with a linear epistasis model first, and use those values as initial guesses.
            super(NonlinearEpistasisModel, self).fit()
        
            # Construct an array of guesses, using the scale specified by user.
            guess = np.ones(self.Interactions.n + self.Parameters.n)
            
            # Add linear guesses to guess array
            guess[:self.Interactions.n] = self.Interactions.values
        
            # Add guesses to input array
            for kw in kwargs:
                index = self.Parameters._mapping[kw] 
                guess[self.Interactions.n + index] = kwargs[kw]
            
            # Fit the nonlinear least squares fit
            self.fit(guess=guess)
            
            if show_stats:
                # Print score
                print("R-squared of fit: " + str(self.Stats.score))
                
                # Print parameters
                for kw in self.Parameters._mapping:
                    print(kw + ": " + str(getattr(self.Parameters, kw)))
            
            
            # Plot if available
            if hasattr(self, "Plot"):
                self.Plot.predicted_phenotypes()
        
        # Construct and return the widget box
        widgetbox = ipywidgets.interactive(fitting, **kwargs)
        return widgetbox
        
        
        
        