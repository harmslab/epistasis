# Suppress an annoying error
import warnings
# warnings.filterwarnings(action="ignore", category=RuntimeWarning)

# Standard library imports
import sys
import json
import inspect

# Scipy stack imports
import numpy as np
import pandas as pd
import lmfit
from lmfit import Parameter, Parameters

# Scikit learn imports
from sklearn.base import BaseEstimator, RegressorMixin

# GPMap import
from gpmap import GenotypePhenotypeMap

# Epistasis imports.
from ..mapping import EpistasisMap
from .base import BaseModel
from .utils import (arghandler, FittingError)
from .linear import (EpistasisLinearRegression, EpistasisLasso)
from ..stats import pearson


class EpistasisNonlinearRegression(BaseModel):
    """Use nonlinear least-squares regression to estimate epistatic coefficients
    and nonlinear scale in a nonlinear genotype-phenotype map.

    This models has two steps:
        1. Fit an additive, linear regression to approximate the average effect
        of individual mutations.
        2. Fit the nonlinear function to the observed phenotypes vs. the
        additive phenotypes estimated in step 1.

    Methods are described in the following publication:
        Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in
        Nonlinear Genotype-Phenotype Maps'. Genetics 205, 1079-1088 (2017).

    Parameters
    ----------
    function : callable
        Nonlinear function between Pobs and Padd

    reverse : callable
        The inverse of the nonlinear function used to back transform from
        nonlinear phenotypic scale to linear scale.

    model_type : str (default: global)
        type of epistasis model to use. See paper above for more information.

    Keyword Arguments
    -----------------
    Keyword arguments are interpreted as intial guesses for the nonlinear
    function parameters. Must have the same name as parameters in the
    nonlinear function.

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
                 model_type="global",
                 **p0):

        # Do some inspection to get the parameters from the nonlinear
        # function argument list.
        func_signature = inspect.signature(function)
        func_params = list(func_signature.parameters.keys())

        if func_params[0] != "x":
            raise Exception("First argument of the nonlinear function must "
                            "be `x`.")

        # Construct lmfit parameters object
        self.parameters = Parameters()
        for p in func_params[1:]:
            # Get starting value of parameter if given.
            val = None
            if p in p0:
                val = p0[p]
            # Add parameter.
            self.parameters.add(name=p, value=val)

        # Set up the function for fitting.
        self.function = function
        self.reverse = reverse
        self.order = 1
        self.Xbuilt = {}

        # Construct parameters object
        self.set_params(model_type=model_type)

        # Store model specs.
        self.model_specs = dict(
            function=function,
            reverse=reverse,
            model_type=self.model_type,
            **p0)

        # Set up additive and high-order linear model
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)

    def add_gpm(self, gpm):
        super(EpistasisNonlinearRegression, self).add_gpm(gpm)
        # Add gpm to other models.
        self.Additive.add_gpm(gpm)
        return self

    @property
    def thetas(self):
        return np.concatenate((list(self.parameters.values()),
                               self.Additive.coef_))

    @property
    def num_of_params(self):
        n = 0
        n += len(self.parameters) + len(self.Additive.coef_)
        return n

    def fit(self,
            X=None,
            y=None,
            use_widgets=False,
            plot_fit=True,
            **kwargs):
        # Fit linear portion
        self._fit_additive(X=X, y=y)

        # Use widgets to guess the value?
        if use_widgets is False:
            # Step 2: fit nonlinear function
            self._fit_nonlinear(X=X, y=y, **kwargs)
            return self

        # Don't use widgets to fit data
        else:
            import matplotlib.pyplot as plt
            import epistasis.pyplot
            import ipywidgets

            # Build fitting method to pass into widget box
            def fitting(**parameters):
                """Callable to be controlled by widgets."""
                # Fit the nonlinear least squares fit
                self._fit_nonlinear(X=X, y=y, **parameters)

                # Print model parameters.
                self.parameters.pretty_print()

            # Construct and return the widget box
            widgetbox = ipywidgets.interactive(fitting, **kwargs)
            return widgetbox

    def _fit_additive(self, X=None, y=None, **kwargs):

        if hasattr(self, 'gpm') is False:
            raise Exception("This model will not work if a genotype-phenotype "
                            "map is not attached to the model class. Use the "
                            "`add_gpm` method")

        # Fit with an additive model
        self.Additive.epistasis = EpistasisMap(
            sites=self.Additive.Xcolumns,
            order=self.Additive.order,
            model_type=self.Additive.model_type
        )

        # Use a first order matrix only.
        if type(X) == np.ndarray or type(X) == pd.DataFrame:
            Xadd = X[:, :self.Additive.epistasis.n]
        else:
            Xadd = X

        # Fit Additive model
        self.Additive.fit(X=Xadd, y=y)
        self.Additive.epistasis.values = self.Additive.coef_

        return self

    @arghandler
    def _fit_nonlinear(self, X=None, y=None, **kwargs):
        """Estimate the scale of multiple mutations in a genotype-phenotype
        map."""
        # Use a first order matrix only.
        if type(X) == np.ndarray or type(X) == pd.DataFrame:
            Xadd = X[:, :self.Additive.epistasis.n]
        else:
            Xadd = X

        # Predict additive phenotypes.
        x = self.Additive.predict(X=Xadd)

        # Set guesses
        for key, value in kwargs.items():
            self.parameters[key].set(value=value)

        # Store residual steps in case fit fails.
        last_residual_set = None

        # Residual function to minimize.
        def residual(params, func, x, y=None):
            # Fit model
            parvals = list(params.values())
            ymodel = func(x, *parvals)

            # Store items in case of error.
            nonlocal last_residual_set
            last_residual_set = (params, ymodel)
            return y - ymodel

        # Minimize the above residual function.
        try:
            self.Nonlinear = lmfit.minimize(
                residual, self.parameters,
                args=[self.function, x],
                kws={'y': y})

        # If fitting fails, print what happened
        except Exception as e:
            # if e is ValueError
            print("ERROR! Some of the transformed phenotypes are invalid.")
            print("\nParameters:")
            print("----------")
            print(last_residual_set[0].pretty_print())
            print("\nTransformed phenotypes:")
            print("----------------------")
            print(last_residual_set[1])
            raise

        # Point to nonlinear.
        self.parameters = self.Nonlinear.params

    @arghandler
    def fit_transform(self, X=None, y=None, **kwargs):
        self.fit(X=X, y=y, **kwargs)

        linear_phenotypes = self.reverse(y, *self.parameters.values())

        # Transform map.
        gpm = GenotypePhenotypeMap.read_dataframe(
            dataframe=self.gpm.data,
            wildtype=self.gpm.wildtype,
            mutations=self.gpm.mutations
        )

        gpm.data['phenotypes'] = linear_phenotypes
        return gpm

    def predict(self, X=None):
        x = self.Additive.predict(X=X)
        y = self.function(x, *self.parameters.values())
        return y

    def predict_transform(self, X=None, y=None):
        if y is None:
            x = self.Additive.predict(X=X)
        else:
            x = y
        return self.function(x, *self.parameters.values())

    @arghandler
    def hypothesis(self, X=None, thetas=None):
        # ----------------------------------------------------------------------
        # Part 0: Break up thetas
        # ----------------------------------------------------------------------
        i, j = len(self.parameters.valuesdict()), self.Additive.epistasis.n
        parameters = thetas[:i]
        epistasis = thetas[i:i + j]

        # Part 1: Linear portion
        ylin = np.dot(X, epistasis)

        # Part 2: Nonlinear portion
        ynonlin = self.function(ylin, *parameters)

        return ynonlin

    def hypothesis_transform(self, X=None, y=None, thetas=None):
        # Part 2: Nonlinear portion
        if y is None:
            x = self.Additive.predict(X=X)
        else:
            x = y
        y_transform = self.reverse(x, *self.parameters.values())
        return y_transform

    @arghandler
    def score(self, X=None, y=None):
        xlin = self.Additive.predict(X=X)
        ypred = self.function(xlin, *self.parameters.values())
        return pearson(y, ypred)**2

    @arghandler
    def lnlike_of_data(self, X=None, y=None, yerr=None, thetas=None):
        # ###### Calculate likelihood #########
        # Calculate ymodel
        ymodel = self.hypothesis(X=X, thetas=thetas)

        # Likelihood of data given model
        return (- 0.5 * np.log(2 * np.pi * yerr**2) -
                (0.5 * ((y - ymodel)**2 / yerr**2)))

    @arghandler
    def lnlike_transform(
            self,
            X=None,
            y=None,
            yerr=None,
            lnprior=None,
            thetas=None):
        # Update likelihood.
        lnlike = self.lnlike_of_data(X=X, y=y, yerr=yerr, thetas=thetas)
        return lnlike + lnprior

class EpistasisNonlinearLasso(EpistasisNonlinearRegression):
    """Use nonlinear regression with a linear lasso epistasis model
    to estimate epistatic coefficients and nonlinear scale in a nonlinear
    genotype-phenotype map.

    This models has two steps:
        1. Fit an additive, linear regression to approximate the average effect
        of individual mutations.
        2. Fit the nonlinear function to the observed phenotypes vs. the
        additive phenotypes estimated in step 1.

    Methods are described in the following publication:
        1. Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in
        Nonlinear Genotype-Phenotype Maps'. Genetics 205, 1079-1088 (2017).
        2. Poelwijk FJ, Socolich M, and Ranganathan R. 'Learning the pattern of
        epistasis linking enotype and phenotype in a protein'.
        bioRxiv. (2017).

    Parameters
    ----------
    function : callable
        Nonlinear function between Pobs and Padd

    reverse : callable
        The inverse of the nonlinear function used to back transform from
        nonlinear phenotypic scale to linear scale.

    model_type : str (default: global)
        type of epistasis model to use. See paper above for more information.

    Keyword Arguments
    -----------------
    Keyword arguments are interpreted as intial guesses for the nonlinear
    function parameters. Must have the same name as parameters in the
    nonlinear function.

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
    def __init__(function,
                 reverse,
                 model_type="global",
                 alpha=1.0,
                 **p0):
        # initialize model
        super(EpistasisNonlinearLasso, self).__init__(function, reverse,
                                                      model_type=model_type)

        # Set up additive and high-order linear model
        self.Additive = EpistasisLasso(
            alpha=alpha,
            order=1, model_type=self.model_type)

    @arghandler
    def lnlike_of_data(self, X=None, y=None, yerr=None, thetas=None):
        # ###### Calculate likelihood #########
        # Calculate ymodel
        ymodel = self.hypothesis(X=X, thetas=thetas)

        # Likelihood of data given model
        return (- 0.5 * np.log(2 * np.pi * yerr**2) -
                (0.5 * ((y - ymodel)**2 / yerr**2)) -
                (self.alpha * sum(abs(thetas))))
