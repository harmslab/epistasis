__doc__ = """ Submodule with nonlinear epistasis models for estimating epistatic interactions in nonlinear genotype-phenotype maps."""

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
        if self._model.Linear.log_transform:
            return 10**np.dot(self._model.X, self._model.Interactions.values)
        else:
            return np.dot(self._model.X, self._model.Interactions.values)

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


class EnsembleStats(object):

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

        Xs = self._model.X
        popt = list()
        for s in self._model.states:
            popt += list(s.Interactions.values)
        popt += self._model.Parameters.get_params()
        phenotypes = self._model._wrapped_function(Xs, *popt)
        return phenotypes



class NonlinearEpistasisModel(EpistasisRegression):
    """ Runs a nonlinear least squares fit to regress epistatic coefficients from
    a genotype-phenotype map which exhibits global nonlinearity in the phenotype.

    This fitting class works in two-steps:
        1. Fit the phenotypes with a nonlinear function (provided by you) that
        captures global structure/scale in the data.
        2. Fit the leftover variation with a linear epistasis model (via linear regression),
        calculating the deviation from additivity/multiplicity in the genotypes.

    The construction of this object, then, has a self.Linear object within, which
    holds the output/results of step 2.

    If the expected effects of mutations are multiplicative, a log transformation
    is applied to the data after step 1 to make fitting the epistatic coefficients
    a simple linear regression. Note: `self.log_transform` will return False still;
    however, self.Linear.log_transform will return True.
    """
    def __init__(self, wildtype, genotypes, phenotypes, function,
        order=None,
        parameters=None,
        stdeviations=None,
        log_transform=False,
        mutations=None,
        n_replicates=1,
        model_type="local",
        logbase=np.log10):

        # Inherit parent class __init__
        super(NonlinearEpistasisModel, self).__init__(wildtype, genotypes, phenotypes,
            order=order,
            parameters=parameters,
            stdeviations=stdeviations,
            log_transform=False,    # Set this log transformation to False
            mutations=mutations,
            n_replicates=n_replicates,
            model_type=model_type,
            logbase=logbase)

        # Initialize the linear function
        self._Linear = EpistasisRegression.from_gpm(self,
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

            # If the underlying genotype map is to be log_transformed
            if self._Linear.log_transform:
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
        guess = np.ones(self.Interactions.n + self.Parameters.n)

        # Add model's extra parameter guesses to input array
        for kw in kwargs:
            index = self.Parameters._mapping[kw]
            guess[self.Interactions.n + index] = kwargs[kw]

        # Create an initial guess array using fit values
        if guess_coeffs is None:
            # Fit with a linear epistasis model first, and use those values as initial guesses.
            super(NonlinearEpistasisModel, self).fit()

            # Add linear guesses to guess array
            guess[:self.Interactions.n] = self.Interactions.values

        else:
            # Add user guesses for interactions
            guess[:self.Interactions.n] = guess_coeff

        # Wrap user function to add epistasis parameters
        self._wrapped_function = self._nonlinear_function_wrapper(self.function)

        # Curve fit the data using a nonlinear least squares fit
        popt, pcov = curve_fit(self._wrapped_function, self.X, self.phenotypes, p0=guess, **fit_kwargs)

        # Set the Interaction values
        self.Interactions.values = popt[0:len(self.Interactions.labels)]

        # Set the other parameters from nonlinear function to fit results
        for i in range(0, self.Parameters.n):
            self.Parameters._set_param(i, popt[len(self.Interactions.labels)+i])

        y_pred = self._wrapped_function(self.X, *popt)

        # Expose the Linear stuff to user and fill in important stuff.
        self.Linear = self._Linear
        self.Linear.phenotypes = y_pred
        self.Linear.Interactions.values = self.Interactions.values

        self._score = pearson(self.phenotypes, y_pred)**2

    @ipywidgets_missing
    def fit_widget(self, print_stats=True, **kwargs):
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
            # Fit the nonlinear least squares fit
            self.fit(**kwargs)

            if print_stats:
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


class EnsembleEpistasisModel(BaseModel):


    def __init__(self, wildtype, genotypes, phenotypes, function,
        order=None,
        parameters=None,
        stdeviations=None,
        log_transform=False,
        mutations=None,
        n_replicates=1,
        model_type="local",
        logbase=np.log10):

        """
        Performs minimization on a nonlinear, ensemble epistasis model function.

        `function` must be a callable function whose first argument is a decomposition matrix.
        Other arguments are extra coefficients/parameters in the nonlinear function

        Example:
        -------

        # nonlinear function to minimize
        def func

        """
        super(EnsembleEpistasisModel, self).__init__(wildtype, genotypes, phenotypes,
            stdeviations=stdeviations,
            log_transform=log_transform,
            mutations=mutations,
            n_replicates=n_replicates,
            logbase=logbase)

        # Set the nonlinear function
        self.function = function

        # Get the parameters from the nonlinear function argument list
        function_sign = inspect.signature(self.function)
        parameters = list(function_sign.parameters.keys())

        # First check that at least one state is given.
        if parameters[0][0] != "x":
            raise Exception(""" First argument of the nonlinear function must be `x`. """)

        # Count the number of states in the ensemble
        self.n_states = 0
        for i in range(len(parameters)):

            # Iterate through parameters in callable function
            if parameters[i][0] == "x":
                self.n_states += 1

        # Construct a list of all the states.
        self.states = list()

        Xs = tuple()

        # Construct a set of models for each state specified
        for i in range(self.n_states):

            # Add linear epistais models to states attribute
            model = EpistasisRegression(wildtype, genotypes, phenotypes,
                order=order,
                parameters=parameters,
                stdeviations=stdeviations,
                log_transform=log_transform,
                mutations=mutations,
                n_replicates=n_replicates,
                model_type=model_type,
                logbase=logbase)

            Xs += (model.X,)
            self.states.append(model)

        # Construct one large X
        self.X = np.concatenate(Xs, axis=1)

        # Construct parameters object
        self.Parameters = Parameters(parameters[self.n_states:])
        self.Stats = EnsembleStats(self)

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

            new_args = tuple()

            X = args[0]
            for i in range(self.n_states):
                # Get number of coeffs in state i
                n_coeffs = self.states[i].Interactions.n

                # arg[i] is the ith state
                xi = X[:,n_coeffs*i:n_coeffs*(i+1)]

                # Convert to epistasis for ith state
                betas_i = np.array(args[1 + n_coeffs*i:1 + n_coeffs*(i+1)])



                # Append all these epistasis coeffs to new argument array
                new_args += ( np.dot(xi, betas_i), )

            # The user defined arguments from nonlinear function are the final set.

            total_coeffs = sum([s.Interactions.n for s in self.states])

            other_args = args[total_coeffs+1:]

            return function(*new_args, *other_args)

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
        # Useful things
        n_coeffs = sum([s.Interactions.n for s in self.states])


        # Construct an array of guesses, using the scale specified by user.
        guess = np.zeros(n_coeffs + self.Parameters.n)

        # Add model's extra parameter guesses to input array
        for kw in kwargs:
            index = self.Parameters._mapping[kw]
            guess[n_coeffs + index] = kwargs[kw]

        # Create an initial guess array using fit values
        if guess_coeffs is None:
            pass
            # Fit with a linear epistasis model first, and use those values as initial guesses.
            #super(NonlinearEpistasisModel, self).fit()

            # Add linear guesses to guess array
            #guess[:self.Interactions.n] = self.Interactions.values

        else:
            # Add user guesses for interactions
            guess[:n_coeffs] = guess_coeff

        # Wrap user function to add epistasis parameters
        self._wrapped_function = self._nonlinear_function_wrapper(self.function)

        # Curve fit the data using a nonlinear least squares fit
        popt, pcov = curve_fit(self._wrapped_function, self.X, self.phenotypes, p0=guess, **fit_kwargs)

        # Set the Interaction values for all states
        for i in range(self.n_states):
            n = self.states[i].Interactions.n
            self.states[i].Interactions.values = popt[i*n:i*n+n]

        # Set the other parameters from nonlinear function to fit results
        for i in range(0, self.Parameters.n):
            self.Parameters._set_param(i, popt[n_coeffs+i])

        # Predict from fit values
        y_pred = self._wrapped_function(self.X, *popt)

        # Calculate the score
        self._score = pearson(self.phenotypes, y_pred)**2

    @ipywidgets_missing
    def fit_widget(self, print_stats=True, **kwargs):
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
            # Fit the nonlinear least squares fit
            self.fit(**kwargs)

            if print_stats:
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
