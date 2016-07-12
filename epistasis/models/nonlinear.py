__doc__ = """Submodule with nonlinear epistasis models for estimating epistatic
interactions in nonlinear genotype-phenotype maps.
"""

import inspect
import numpy as np
from scipy.optimize import curve_fit

from epistasis.decomposition import generate_dv_matrix
from epistasis.stats import pearson
from epistasis.models.regression import EpistasisRegression
from epistasis.models.base import BaseModel
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
    """Object that returns useful statistics and tranformations from a nonlinear
    Epistasis model.
    """
    def __init__(self, model):
        self._model = model

    @property
    def score(self):
        """ Get the epistasis model score after estimating interactions. """
        return self._model._score

    def _subtract_function(self):
        """Returns phenotypes without the nonlinear function.

        Not developed yet.
        """
        pass

    def linear(self):
        """Return phenotypes composed of only the Interaction values determined by
        fit. Removes the nonlinear function from the data.
        """
        if self._model.linear.log_transform:
            return 10**np.dot(self._model.X, self._model.epistasis.values)
        else:
            return np.dot(self._model.X, self._model.epistasis.values)

    def predict(self):
        """ Infer the phenotypes from model.

            __Returns__:

            `genotypes` [array] : array of genotypes -- in same order as phenotypes

            `phenotypes` [array] : array of quantitative phenotypes.
        """
        phenotypes = np.zeros(len(self._model.complete_genotypes), dtype=float)
        binaries = self._model.binary.complete_genotypes

        X = generate_dv_matrix(binaries, self._model.epistasis.labels, encoding=self._model.encoding)

        popt = list(self._model.epistasis.values) + self._model.parameters.get_params()
        phenotypes = self._model._wrapped_function(X, *popt)
        return phenotypes

class NonlinearEpistasisModel(EpistasisRegression):
    """ Runs a nonlinear least squares fit to regress epistatic coefficients from
    a genotype-phenotype map which exhibits global nonlinearity in the phenotype.

    This fitting class works in two-steps:
        1. Fit the phenotypes with a nonlinear function (provided by you) that
        captures global structure/scale in the data.
        2. Fit the leftover variation with a linear epistasis model (via linear regression),
        calculating the deviation from additivity/multiplicity in the genotypes.

    The construction of this object, then, has a self.linear object within, which
    holds the output/results of step 2.

    If the expected effects of mutations are multiplicative, a log transformation
    is applied to the data after step 1 to make fitting the epistatic coefficients
    a simple linear regression. Note: `self.log_transform` will return False still;
    however, self.linear.log_transform will return True.


    Parameters
    ----------
    wildtype : str
        wildtype sequence to be used as the reference state.
    genotypes : array-like
        list of genotypes
    phenotypes : array-like
        list of the phenotypes in same order as their genotype
    function : callable
        nonlinear function for scaling phenotypes
    order : int
        order of epistasis model
    stdeviations : array-like
        standard deviations
    log_transform : bool
        if true, log transform the linear space. Note: this does not transform the
        nonlinear feature of this space.
    mutations : dict
        mapping sites to their mutation alphabet
    n_replicates : int
        number of replicate measurements for each phenotypes
    model_type : str
        model type (global or local)
    logbase : callable
        logarithm function for transforming phenotypes.

    Attributes
    ----------
    see seqspace for attributes from GenotypePhenotype.

    parameters : Parameters object
        store output from the nonlinear function parameters
    linear : EpistasisRegression
        linear epistasis regression for calculating specific interactions.
    """
    def __init__(self, wildtype, genotypes, phenotypes, function,
        order=None,
        stdeviations=None,
        log_transform=False,
        mutations=None,
        n_replicates=1,
        model_type="local",
        logbase=np.log10):

        # Inherit parent class __init__
        super(NonlinearEpistasisModel, self).__init__(wildtype, genotypes, phenotypes,
            order=order,
            stdeviations=stdeviations,
            log_transform=False,    # Set this log transformation to False
            mutations=mutations,
            n_replicates=n_replicates,
            model_type=model_type,
            logbase=logbase)

        # Initialize the linear function
        self.linear = EpistasisRegression.from_gpm(self,
            log_transform=log_transform,
            order=order,
        )

        # Set the nonlinear function
        self.function = function

        # Get the parameters from the nonlinear function argument list
        function_sign = inspect.signature(self.function)
        parameters = list(function_sign.parameters.keys())

        # Check that the first argument is epistasis
        if parameters[0] != "x":
            raise Exception(""" First argument of the nonlinear function must be `x`. """)

        # Construct parameters object
        self.parameters = Parameters(parameters[1:])
        self.statistics = NonlinearStats(self)

        # Add a plotting object if matplotlib exists
        try:
            self.plot = NonlinearPlotting(self)
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
            n_coeffs = len(self.epistasis.labels)
            betas = args[1:1+n_coeffs]

            # The user defined arguments from nonlinear function are the final set.
            other_args = args[n_coeffs+1:]

            # If the underlying genotype map is to be log_transformed
            if self.linear.log_transform:
                new_x = 10**(np.dot(x,betas))
            else:
                new_x = np.dot(x, betas)
            return function(new_x, *other_args)

        return inner


    def fit(self, guess_coeffs=None, fit_kwargs={}, **kwargs):
        """
            Fit using nonlinear least squares regression.

            Arguments:
            ---------
            guess: array of guesses

            fit_kwargs are passed to scipy's curvefit function

            **kwargs are used as parameters.
        """
        # Construct an array of guesses, using the scale specified by user.
        guess = np.ones(self.epistasis.n + self.parameters.n)
        # Add model's extra parameter guesses to input array
        for kw in kwargs:
            index = self.parameters._mapping[kw]
            guess[self.epistasis.n + index] = kwargs[kw]
        # Create an initial guess array using fit values
        if guess_coeffs is None:
            # Fit with a linear epistasis model first, and use those values as initial guesses.
            super(NonlinearEpistasisModel, self).fit()
            # Add linear guesses to guess array
            guess[:self.epistasis.n] = self.epistasis.values
        else:
            # Add user guesses for interactions
            guess[:self.epistasis.n] = guess_coeffs
        # Wrap user function to add epistasis parameters
        self._wrapped_function = self._nonlinear_function_wrapper(self.function)
        # Curve fit the data using a nonlinear least squares fit
        popt, pcov = curve_fit(self._wrapped_function, self.X, self.phenotypes, p0=guess, **fit_kwargs)
        # Set the Interaction values
        self.epistasis.values = popt[0:len(self.epistasis.labels)]
        # Set the other parameters from nonlinear function to fit results
        for i in range(0, self.parameters.n):
            self.parameters._set_param(i, popt[len(self.epistasis.labels)+i])
        # Expose the linear stuff to user and fill in important stuff.
        self.linear.epistasis.values = self.epistasis.values
        linear_phenotypes = np.dot(self.X, self.epistasis.values)
        if self.linear.log_transform:
            linear_phenotypes = 10**linear_phenotypes
        self.linear.phenotypes = linear_phenotypes
        # Score the fit
        y_pred = self._wrapped_function(self.X, *popt)
        self._score = pearson(self.phenotypes, y_pred)**2

    @ipywidgets_missing
    def fit_widget(self, print_stats=True, **kwargs):
        """Simple IPython widget for trying initial guesses of the nonlinear parameters.

        This works by, first, fitting the coefficients using a linear epistasis model as initial
        guesses (along with user defined kwargs) for the nonlinear model.

        kwargs should be ranges of guess values for each parameter. They are are turned into
        slider widgets for varying these guesses easily. The kwarg needs to match the name of
        the parameter in the nonlinear fit.

        """
        # Build fitting method to pass into widget box
        def fitting(**kwargs):
            """ Callable to be controlled by widgets. """
            # Fit the nonlinear least squares fit
            self.fit(**kwargs)

            if print_stats:
                # Print score
                print("R-squared of fit: " + str(self.statistics.score))

                # Print parameters
                for kw in self.Parameters._mapping:
                    print(kw + ": " + str(getattr(self.parameters, kw)))

            # Plot if available
            if hasattr(self, "Plot"):
                self.plot.predicted_phenotypes()

        # Construct and return the widget box
        widgetbox = ipywidgets.interactive(fitting, **kwargs)
        return widgetbox
