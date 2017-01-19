import inspect
import numpy as np
from functools import wraps
from scipy.optimize import curve_fit

from sklearn.base import BaseEstimator, RegressorMixin
from ..base import BaseModel, X_fitter, X_predictor

from ..linear.regression import EpistasisLinearRegression
from epistasis.stats import pearson
# decorators for catching errors
from gpmap.utils import ipywidgets_missing

# Try to import ipython specific tools
try:
    import ipywidgets
except ImportError:
    pass

class Parameters(object):
    """ A container object for parameters extracted from a nonlinear fit.
    """
    def __init__(self, params):
        self._param_list = params
        self.n = len(self._param_list)
        self._mapping, self._mapping_  = {}, {}
        for i in range(self.n):
            setattr(self, self._param_list[i], 0)
            self._mapping_[i] = self._param_list[i]
            self._mapping[self._param_list[i]] = i

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
    """Fit a nonlinear epistasis model.
    """
    def __init__(self, function, reverse,
        order=1,
        model_type="global",
        fix_linear=False,
        **kwargs):
        # Get the parameters from the nonlinear function argument list
        function_sign = inspect.signature(function)
        parameters = list(function_sign.parameters.keys())
        if parameters[0] != "x":
            raise Exception(""" First argument of the nonlinear function must be `x`. """)
        # Add parameters to kwargs
        self.parameters = Parameters(parameters[1:])

        # Set up the function for fitting.
        self._function = function
        @wraps(self._function)
        def function(*args, **kwargs): return self._function(*args, **kwargs)
        self.function = function

        self.reverse = reverse
        # Construct parameters object
        self.set_params(order=order,
            model_type=model_type,
            fix_linear=fix_linear,
        )

    @ipywidgets_missing
    def fit(self, X=None, y=None, sample_weight=None, use_widgets=False, **parameters):
        """Fit nonlinearity in genotype-phenotype map.

        Also, can use IPython widget for trying initial guesses of the nonlinear parameters.

        This works by, first, fitting the coefficients using a linear epistasis model as initial
        guesses (along with user defined kwargs) for the nonlinear model.

        kwargs should be ranges of guess values for each parameter. They are are turned into
        slider widgets for varying these guesses easily. The kwarg needs to match the name of
        the parameter in the nonlinear fit.
        """
        # Fitting data
        ## Use widgets to guesst the value?
        if use_widgets:
            # Build fitting method to pass into widget box
            def fitting(**parameters):
                """ Callable to be controlled by widgets. """
                # Fit the nonlinear least squares fit
                if self.fix_linear:
                    self._fit_(X=X, y=y, sample_weight=sample_weight, **parameters)
                else:
                    self._fit_float_linear(X=X, y=y, sample_weight=sample_weight, **parameters)
                #if print_stats:
                # Print score
                print("R-squared of fit: " + str(self.score()))
                # Print parameters
                for kw in self.parameters._mapping:
                    print(kw + ": " + str(getattr(self.parameters, kw)))
                # Plot if available
                #if hasattr(self, "plot"):
                #    self.plot.best_fit()
            # Construct and return the widget box
            widgetbox = ipywidgets.interactive(fitting, **parameters)
            return widgetbox
        # Don't use widgets to fit data
        else:
            if self.fix_linear:
                self._fit_(X=X, y=y, sample_weight=sample_weight, **parameters)
            else:
                raise Exception("This is currently broken, fixing in next release.")
                self._fit_float_linear(X=X, y=y, sample_weight=sample_weight, **parameters)
        self._score = self.score(X=X, y=y)


    def _function_generator(self, X, y, parameters):
        """Nonlinear function wrapper adding epistasis as an argument to user defined function
        """
        def inner(*args):
            """Replace `x` in the user defined function with the decomposition
            matrix `X`. The new `X` maps all possible high order epistasis interactions
            to the phenotypes they model.
            """
            x = X[:,:-self.parameters.n]
            n_samples, n_features = x.shape
            ###### Deconstruct the arguments in user's function
            # X is the first argument
            x = X[:, n_features]

            # The epistasis coefficients are the next set of arguments
            n_coeffs = len(coeffs)
            betas = args[1:1+n_coeffs]
            # The user defined arguments from nonlinear function are the final set.
            other_args = args[n_coeffs+1:]
            new_x = np.dot(x, betas)
            return function(new_x, *other_args)
        return inner

    def _guess_model(self, X, y, sample_weight=None):
        """Initializes and fits a Linear Regression to guess epistatic coefficients.
        """
        # Construct the linear portion of the fit.
        linear = EpistasisLinearRegression(self.order, self.model_type)
        linear.fit(X, y, sample_weight=None)
        return linear

    @X_fitter
    def _fit_(self, X=None, y=None, sample_weight=None, **kwargs):
        """Fit the genotype-phenotype map for epistasis.
        """
        # Fit coeffs
        guess = self._guess_model(X, y, sample_weight=sample_weight)
        self.coef_ = guess.coef_
        # Get the scale of the map.
        x = guess.predict(X)
        # Construct an array of guesses, using the scale specified by user.
        guess = np.ones(self.parameters.n)
        # Add model's extra parameter guesses to input array
        for kw in kwargs:
            index = self.parameters._mapping[kw]
            guess[index] = kwargs[kw]
        # create a sigma array for weighted fits with curve fit.
        if sample_weight is None:
            sigma = None
        else:
            sigma = 1 / np.sqrt(sample_weight)
        # Curve fit the data using a nonlinear least squares fit
        popt, pcov = curve_fit(self._function, x, y, p0=guess, sigma=sigma)
        for i in range(0, self.parameters.n):
            self.parameters._set_param(i, popt[i])

    @X_fitter
    def _fit_float_linear(self, X=None, y=None, **kwargs):
        """Fit using nonlinear least squares regression.

        Parameters
        ----------
        """
        # Guess coeffs
        guess = self._guess_model(X, y)
        n_coef = len(guess.coef)
        # Construct an array of guesses, using the scale specified by user.
        guesses = np.ones(n_coef + self.parameters.n)
        # Add model's extra parameter guesses to input array
        for kw in kwargs:
            index = self.parameters._mapping[kw]
            guesses[n_coef + index] = kwargs[kw]
        # Add linear guesses to guess array
        guesses[:n_coef] = guess.coef_
        # Transform user function to readable model function
        self._wrapped_function = self._function_generator(self._function)
        # Curve fit the data using a nonlinear least squares fit
        popt, pcov = curve_fit(self._wrapped_function, X, y, p0=guess)
        # Set the Interaction values
        self.coef_ = popt[0:n_coef]
        # Set the other parameters from nonlinear function to fit results
        for i in range(0, self.parameters.n):
            self.parameters._set_param(i, popt[n_coef+i])

    def transform_target(self, y=None):
        """Use the inverse of the function to transform y (the target) to it's
        linear scale. Only works if a reverse function is given.
        """
        if self.reverse is None:
            raise AttributeError("Reverse method is not given.")
        if y is None:
            y = self.gpm.phenotypes
        y_transformed = self.reverse(y, *self.parameters.values)
        return y_transformed

    @X_predictor
    def predict_linear(self, X=None):
        """Use the inverse of the function to transform y (the target) to it's
        linear scale. Only works if a reverse function is given.
        """
        return np.dot(X, self.coef_)

    @X_predictor
    def predict(self, X=None):
        """Predict new targets from model."""
        x = self.predict_linear(X)
        y = self.function(x, *self.parameters.values)
        return y

    @X_fitter
    def score(self, X=None, y=None):
        """ Calculates the squared-pearson coefficient between the model's
        predictions and data.
        """
        y_pred = self.predict(X)
        return pearson(y, y_pred)**2
