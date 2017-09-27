import inspect
import numpy as np
import pandas as pd
import json
from functools import wraps
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.base import BaseEstimator, RegressorMixin

# Import epistasis modules.
from .base import BaseModel
from .utils import  X_fitter, X_predictor, FittingError
from .linear import EpistasisLinearRegression
from epistasis.stats import pearson
# decorators for catching errors
from gpmap.utils import ipywidgets_missing

# Try to import ipython specific tools
try:
    import ipywidgets
except ImportError:
    pass

# Suppress an annoying error
import warnings
#warnings.filterwarnings(action="ignore", category=RuntimeWarning)

class Parameters(object):
    """A container object for parameters extracted from a nonlinear fit.
    """
    def __init__(self, params):
        self._param_list = params
        self.n = len(self._param_list)
        self._mapping, self._mapping_  = {}, {}
        for i in range(self.n):
            setattr(self, self._param_list[i], 0)
            self._mapping_[i] = self._param_list[i]
            self._mapping[self._param_list[i]] = i

    def to_json(self, filename):
        """Write parameters to json
        """
        with open(filename, "w") as f:
            json.dump(f, self())

    def __call__(self):
        """Return parameters if the instance is called."""
        return dict(zip(self.keys, self.values))

    @property
    def keys(self):
        """Get ordered list of params"""
        return self._param_list

    @property
    def values(self):
        """Get ordered list of params"""
        vals = []
        for p in self._param_list:
            vals.append(getattr(self, p))
        return vals

    def _set_param(self, param, value):
        """ Set Parameter value. Method is not exposed to user.
        param can be either the name of the parameter or its index in this object.
        """
        # If param is an index, get name from mappings
        if type(param) == int or type(param) == float:
            param = self._mapping_[param]
        setattr(self, param, value)

    def get_params(self):
        """ Get an ordered list of the parameters."""
        return [getattr(self, self._mapping_[i]) for i in range(len(self._mapping_))]

class EpistasisNonlinearRegression(RegressorMixin, BaseEstimator, BaseModel):
    """Use nonlinear least-squares regression to estimate epistatic coefficients
    and nonlinear scale in a nonlinear genotype-phenotype map.

    This models has three steps:
        1. Fit an additive, linear regression to approximate the average effect of
            individual mutations.
        2. Fit the nonlinear function to the observed phenotypes vs. the additive
            phenotypes estimated in step 1.
        3. Transform the phenotypes to this linear scale and fit leftover variation
            with high-order epistasis model.

    Methods are described in the following publication:
        Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in Nonlinear
        Genotype-Phenotype Maps'. Genetics 205, 1079-1088 (2017).

    Parameters
    ----------
    function : callable
        Nonlinear function between Pobs and Padd
    reverse : callable
        The inverse of the nonlinear function used to back transform from nonlinear
        phenotypic scale to linear scale.
    order : int
        order of epistasis to fit.
    model_type : str (default: global)
        type of epistasis model to use. See paper above for more information.

    Keyword Arguments
    -----------------
    Keyword arguments are interpreted as intial guesses for the nonlinear function
    parameters. Must have the same name as parameters in the nonlinear function

    Attributes
    ----------
    epistasis : EpistasisMap
        Mapping object containing high-order epistatic coefficients
    Linear : EpistasisLinearRegression
        Linear regression object for fitting high-order epistasis model
    Additive : EpistasisLinearRegression
        Linear regression object for fitting additive model
    parameters : Parameters object
        Mapping object for nonlinear coefficients
    """
    def __init__(self,
        function,
        reverse,
        order=1,
        model_type="global",
        **p0):

        # Do some inspection to
        # Get the parameters from the nonlinear function argument list
        function_sign = inspect.signature(function)
        parameters = list(function_sign.parameters.keys())
        if parameters[0] != "x":
            raise Exception(""" First argument of the nonlinear function must be `x`.""")

        # Set up the function for fitting.
        self.function = function
        self.reverse = reverse

        # Construct parameters object
        self.parameters = Parameters(parameters[1:])
        self.set_params(order=order,
            model_type=model_type)

        # Initial parameters guesses
        self.p0 = p0
        
        # Set up additive and high-order linear model
        self.Additive = EpistasisLinearRegression(order=1, model_type=self.model_type)
        self.Linear = EpistasisLinearRegression(order=self.order, model_type=self.model_type)

    @property
    def thetas(self):
        """Get all parameters in the model as a single array. This concatenates
        the nonlinear parameters and high-order epistatic coefficients. Nonlinear
        parameters are first in the array; linear coefficients are second.
        """
        return np.concatenate((self.parameters.values, self.Linear.coef_))

    def fit(self, X='obs', y='obs', sample_weight=None, use_widgets=False, plot_fit=True, **kwargs):
        """Fit nonlinearity in genotype-phenotype map.

        Parameters
        ----------
        X : 2-d array
            independent data; samples.
        y : array
            dependent data; observations.
        sample_weight : array (default: None, assumed even weight)
            weights for fit.

        Notes
        -----
        Also, will create IPython widgets to sweep through initial parameters guesses.

        This works by, first, fitting the coefficients using a linear epistasis model as initial
        guesses (along with user defined kwargs) for the nonlinear model.

        kwargs should be ranges of guess values for each parameter. They are are turned into
        slider widgets for varying these guesses easily. The kwarg needs to match the name of
        the parameter in the nonlinear fit.
        """
        if hasattr(self, 'gpm') is False:
            raise Exception("This model will not work if a genotype-phenotype "
                "map is not attached to the model class. Use the `add_gpm` method")

        # ----------------------------------------------------------------------
        # Part 1: Estimate average, independent mutational effects and fit
        #         nonlinear scale.
        # ----------------------------------------------------------------------
        # Get pobs for nonlinear fit.
        if type(y) is str and y in ["obs", "complete"]:            
            pobs = self.gpm.binary.phenotypes
        # Else, numpy array or dataframe
        elif type(y) == np.array or type(y) == pd.Series:
            pobs = y
        else:
            raise FittingError("y is not valid. Must be one of the following: 'obs', 'complete', "
                           "numpy.array, pandas.Series. Right now, its {}".format(type(y)))    
        
        # Fit with an additive model
        self.Additive.add_gpm(self.gpm)
        self.Additive.add_epistasis()
        
        # Use a first order matrix only.
        if type(X) == np.ndarray or type(X) == pd.DataFrame:
            Xadd = X[:,:self.Additive.epistasis.n]
        else:
            Xadd = X
        
        # Fit Additive model
        self.Additive.fit(X=Xadd, y=pobs)
        self.Additive.epistasis.values = self.Additive.coef_
        
        # Linearize phenotypes
        padd = self.Additive.predict(X=Xadd)
        
        # If true, make a plot of the
        #if plot_fit:
        #    fig, ax = self.plot_fit()

        # ----------------------------------------------------------------------
        # Part 2: Estimate nonlinear function.
        # ----------------------------------------------------------------------
        
        # Prepare a high-order model
        self.Linear.add_gpm(self.gpm)
        self.Linear.add_epistasis()

        # Call fit one time on nonlinear space to built X matrix
        self.Linear.add_X(X=X, key="fit")

        ## Use widgets to guess the value?
        if use_widgets:
            import matplotlib.pyplot as plt
            import epistasis.plot

            # Build fitting method to pass into widget box
            def fitting(**parameters):
                """Callable to be controlled by widgets."""
                # Fit the nonlinear least squares fit
                self._fit_(padd, pobs, sample_weight=sample_weight, **parameters)

                # Print score
                #print("R-squared of fit: " + str(self.score(X=Xadd, y=padd)))
                # Print parameters
                for kw in self.parameters._mapping:
                    print(kw + ": " + str(getattr(self.parameters, kw)))

                # Plot the nonlinear fit!
                ylin = self.Additive.predict(X=Xadd)
                epistasis.plot.corr_resid(ylin, y, figsize=(3,5))
                plt.show()

            # Construct and return the widget box
            widgetbox = ipywidgets.interactive(fitting, **kwargs)
            return widgetbox

        # Don't use widgets to fit data
        else:
            self._fit_(padd, pobs, sample_weight=sample_weight, **kwargs)
        return self

    def _fit_(self, x, y, sample_weight=None, **kwargs):
        """Estimate the scale of multiple mutations in a genotype-phenotype map."""
        
        # Set up guesses for parameters
        self.p0.update(**kwargs)
        kwargs = self.p0
        guesses = np.ones(self.parameters.n)
                
        for kw in kwargs:
            index = self.parameters._mapping[kw]
            guesses[index] = kwargs[kw]

        # Convert weights to variances on fit parameters.
        if sample_weight is None:
            sigma = None
        else:
            sigma = 1 / np.sqrt(sample_weight)

        # Fit with curve_fit, using
        popt, pcov = curve_fit(self.function, x, y, p0=guesses, sigma=sigma, method="trf")
        for i in range(0, self.parameters.n):
            self.parameters._set_param(i, popt[i])

        # ----------------------------------------------------------------------
        # Part 3: Fit high-order, linear model.
        # ----------------------------------------------------------------------

        # Construct a linear epistasis model.
        if self.order > 1:
            Xlin = self.Linear.Xbuilt["fit"]
            ylin = self.reverse(y, *self.parameters.values)
            # Now fit with a linear epistasis model.
            self.Linear.fit(X=Xlin, y=ylin)
        else:
            self.Linear = self.Additive
        # Map to epistasis.
        self.Linear.epistasis.values = self.Linear.coef_

    def plot_fit(self):
        """Plots the observed phenotypes against the additive model phenotypes"""
        padd = self.Additive.predict()
        pobs = self.gpm.phenotypes
        fig, ax = plt.subplots(figsize=(3,3))
        ax.plot(padd, pobs, '.b')
        plt.show()
        return fig, ax        

    def predict(self, X='complete'):
        """Infer phenotypes from model coefficients and nonlinear function."""
        x = self.Linear.predict(X)
        y = self.function(x, *self.parameters.values)
        return y

    def score(self, X='obs', y='obs'):
        """Calculates the squared-pearson coefficient for the nonlinear fit.

        Returns
        -------
        r_nonlinear : float
            squared pearson coefficient between phenotypes and nonlinear function.
        r_linear : float
            squared pearson coefficient between linearized phenotypes and linear epistasis model
            described by epistasis.values.
        """
        xlin = self.Additive.predict(X=X)
        ypred = self.function(xlin, *self.parameters.get_params())
        yrev = self.reverse(y, *self.parameters.get_params())
        return pearson(y, ypred)**2, self.Linear.score(X=X, y=yrev)

    @X_predictor
    def hypothesis(self, X='complete', thetas=None):
        """Given a set of parameters, compute a set of phenotypes. Does not predict. This is method
        can be used to test a set of parameters (Useful for bayesian sampling).
        """
        # ----------------------------------------------------------------------
        # Part 0: Break up thetas
        # ----------------------------------------------------------------------
        # Get thetas from model.
        if thetas is None:
            thetas = self.thetas

        i, j = self.parameters.n, self.Linear.epistasis.n
        parameters = thetas[:i]
        epistasis = thetas[i:i+j]

        # Part 1: Linear portion
        ylin = np.dot(X, epistasis)

        # Part 2: Nonlinear portion
        ynonlin = self.function(ylin, *parameters)

        return ynonlin

    def lnlike_of_data(self, X='obs', y='obs', yerr='obs', thetas=None):
        """Calculate the log likelihoods of each data point, given a set of model coefficients.

        Parameters
        ----------
        X : 2d array
            model matrix
        y : array
            data to calculate the likelihood
        yerr: array
            uncertainty in data
        thetas : array
            array of model coefficients

        Returns
        -------
        lnlike : np.ndarray
            log-likelihood of each data point given a model.
        """
        ####### Prepare input #########
        # If no model parameters are given, use the model fit.
        if thetas is None:
            thetas = self.thetas

        ######## Handle y.
        # Get pobs for nonlinear fit.
        if type(y) is str and y in ["obs", "complete"]:            
            ydata = self.gpm.binary.phenotypes
        # Else, numpy array or dataframe
        elif type(y) == np.array or type(y) == pd.Series:
            ydata = y
        else:
            raise FittingError("y is not valid. Must be one of the following: 'obs', 'complete', "
                           "numpy.array, pandas.Series. Right now, its {}".format(type(y)))    

        ######## Handle yerr.
        # Check if yerr is string
        if type(yerr) is str and yerr in ["obs", "complete"]:
            yerr = self.gpm.binary.std.upper

        # Else, numpsy array or dataframe
        elif type(y) != np.array and type(y) != pd.Series:
            raise FittingError("yerr is not valid. Must be one of the following: 'obs', 'complete', "
                           "numpy.array, pandas.Series. Right now, its {}".format(type(yerr)))    

        ####### Calculate likelihood #########
        # Calculate ymodel
        ymodel = self.hypothesis(X=X, thetas=thetas)

        # Likelihood of data given model
        return np.log(2*np.pi*yerr**2) + ((ydata - ymodel)/yerr)**2 
    
    def lnlikelihood(self, X='obs', y='obs', yerr='obs', thetas=None):
        """Calculate the log likelihood of the data, given a set of model coefficients.

        Parameters
        ----------
        X : 2d array
            model matrix
        y : array
            data to calculate the likelihood
        yerr: array
            uncertainty in data
        thetas : array
            array of model coefficients

        Returns
        -------
        lnlike : float
            log-likelihood of data given a model.
        """    
        lnlike = -0.5 * np.sum( self.lnlike_of_data(X=X, y=y, yerr=yerr, thetas=thetas) )
        # If log-likelihood is infinite, set to negative infinity.
        if np.isinf(lnlike):
            return -np.inf
        
        elif np.isnan(lnlike):
            raise FittingError("Got an NaN in the likelihood.")
        return lnlike
