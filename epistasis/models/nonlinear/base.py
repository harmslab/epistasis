import inspect
import numpy as np
from scipy.optimize import curve_fit

from sklearn.base import RegressorMixin
from ..base import BaseModel

from ..linear.regression import EpistasisLinearRegression
from epistasis.stats import pearson
# decorators for catching errors
from seqspace.utils import ipywidgets_missing

# Try to import ipython specific tools
try:
    import ipywidgets
except ImportError:
    pass

class Parameters(object):

    def __init__(self, model, param_list):
        self.list = param_list
        self.model = model

    @property
    def n(self):
        return len(self.list)

    @property
    def keys(self):
        return list

    @property
    def values(self):
        modelparams = self.model.get_params()
        return [modelparams[p] for p in self.list]

class EpistasisNonlinearRegression(RegressorMixin, BaseModel):
    """Fit a nonlinear epistasis model.
    """
    def __init__(self, function, reverse,
        order=1,
        model_type="global",
        fix_linear=False,
        **kwargs):
        # Get the parameters from the nonlinear function argument list
        function_sign = inspect.signature(self.function)
        parameters = list(function_sign.parameters.keys())
        if parameters[0] != "x":
            raise Exception(""" First argument of the nonlinear function must be `x`. """)
        # Add parameters to kwargs
        for p in parameters[1:]:
            kwargs[p] = 0
        self.parameters = Parameters(parameters)
        # Construct parameters object
        self.set_params(order=order,
            model_type=model_type,
            fix_linear=fix_linear,
            function=function,
            reverse=reverse,
            **kwargs)

    @ipywidgets_missing
    def fit(self, X=None, y=None, use_widgets=False, **parameters):
        """Fit nonlinearity in genotype-phenotype map.

        Also, can use IPython widget for trying initial guesses of the nonlinear parameters.

        This works by, first, fitting the coefficients using a linear epistasis model as initial
        guesses (along with user defined kwargs) for the nonlinear model.

        kwargs should be ranges of guess values for each parameter. They are are turned into
        slider widgets for varying these guesses easily. The kwarg needs to match the name of
        the parameter in the nonlinear fit.
        """
        # Prepare data, and fit matrix
        if y is None:
                y = self.gpm.phenotypes
        if X is None:
            # Build X AND EpistasisMap attachment.
            X = self.X_helper(
                genotypes=self.gpm.binary.genotypes,
                **self.get_params())
            self.X = X
        # Fitting data
        ## Use widgets to guesst the value?
        if use_widgets:
            # Build fitting method to pass into widget box
            def fitting(**parameters):
                """ Callable to be controlled by widgets. """
                # Fit the nonlinear least squares fit
                if self.fix_linear:
                    self._fit_(**parameters)
                else:
                    self._fit_float_linear(**parameters)
                if print_stats:
                    # Print score
                    print("R-squared of fit: " + str(self.statistics.score))
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
                self._fit_(**parameters)
            else:
                self._fit_float_linear(**parameters)

    def _kernel_generator(self, X, y, parameters):
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

    def _fit_(self, X=None, y=None, **kwargs):
        """Fit the genotype-phenotype map for epistasis.
        """
        # Construct the linear portion of the fit.
        try:
            self.linear = EpistasisLinearRegression.from_gpm(self.gpm, **self.get_params())
        except:
            self.linear = EpistasisLinearRegression(**self.get_params)
        # Fit the linear portion
        self.linear.fit()
        # Get the scale of the map.
        linear_phenotypes = self.linear.predict()
        # Construct an array of guesses, using the scale specified by user.
        guess = np.ones(self.parameters.n)
        # Add model's extra parameter guesses to input array
        for kw in kwargs:
            index = self.parameters._mapping[kw]
            guess[index] = kwargs[kw]
        # Curve fit the data using a nonlinear least squares fit
        popt, pcov = curve_fit(self.function, linear_phenotypes,
            self.phenotypes,
            p0=guess)
        for i in range(0, self.parameters.n):
            self.parameters._set_param(i, popt[i])

    def _fit_float_linear(self, X=None, y=None, guess_coeffs=None, fit_kwargs={}, **kwargs):
        """Fit using nonlinear least squares regression.

        Parameters
        ----------
        guess : array-like
            array of guesses
        fit_kwargs : dict
            specific keyword arguments for scipy's curve_fit function. This is
            not for parameters.
        **kwargs :
            guess values for `parameters` in the nonlinear function passed in as
            keyword arguments
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

        # Transform user function to readable model function
        self._wrapped_function = self._nonlinear_function_wrapper(self.function)

        # Curve fit the data using a nonlinear least squares fit
        popt, pcov = curve_fit(self._wrapped_function, self.X, self.phenotypes,
            p0=guess,
            **fit_kwargs)

        # Set the Interaction values
        self.epistasis.values = popt[0:len(self.epistasis.labels)]

        # Set the other parameters from nonlinear function to fit results
        for i in range(0, self.parameters.n):
            self.parameters._set_param(i, popt[len(self.epistasis.labels)+i])

        # Expose the linear stuff to user and fill in important stuff.
        self.linear.epistasis.values = self.epistasis.values
        linear_phenotypes = np.dot(self.X, self.epistasis.values)
        self.linear.phenotypes = linear_phenotypes

        # Score the fit
        y_pred = self._wrapped_function(self.X, *popt)
        self._score = pearson(self.phenotypes, y_pred)**2

    def predict(self, X=None):
        """
        """

    def score(self, X=None, y=None):
        """ Calculates the squared-pearson coefficient between the model's
        predictions and data.
        """
        if y is None:
            y = self.gpm.phenotypes
        y_pred = self.predict(X)
        return pearson(y, y_pred)**2
