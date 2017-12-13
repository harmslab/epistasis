# Suppress an annoying error
import warnings
# warnings.filterwarnings(action="ignore", category=RuntimeWarning)

# Standard library imports
import sys
import json
import inspect
from functools import wraps

# Scipy stack imports
import numpy as np
import pandas as pd
import lmfit
from lmfit import Parameter, Parameters

# Scikit learn imports
from sklearn.base import BaseEstimator, RegressorMixin

# Epistasis imports.
from .base import BaseModel
from .utils import (X_fitter, X_predictor, FittingError)
from .linear import (EpistasisLinearRegression, EpistasisLasso)
from ..stats import pearson


class EpistasisNonlinearRegression(RegressorMixin, BaseEstimator, BaseModel):
    """Use nonlinear least-squares regression to estimate epistatic coefficients
    and nonlinear scale in a nonlinear genotype-phenotype map.

    This models has three steps:
        1. Fit an additive, linear regression to approximate the average effect
        of individual mutations.
        2. Fit the nonlinear function to the observed phenotypes vs. the
        additive phenotypes estimated in step 1.
        3. Transform the phenotypes to this linear scale and fit leftover
        variation with high-order epistasis model.

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
    order : int
        order of epistasis to fit.
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
                 order=1,
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

        # Construct parameters object
        self.set_params(order=order,
                        model_type=model_type)

        # Store model specs.
        self.model_specs = dict(
            function=function,
            reverse=reverse,
            order=self.order,
            model_type=self.model_type,
            **p0)

        # Set up additive and high-order linear model
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)
        self.Linear = EpistasisLinearRegression(
            order=self.order, model_type=self.model_type)

    @wraps(BaseModel.add_gpm)
    def add_gpm(self, gpm):
        super(EpistasisNonlinearRegression, self).add_gpm(gpm)
        # Add gpm to other models.
        self.Additive.add_gpm(gpm)
        self.Linear.add_gpm(gpm)

    @property
    def thetas(self):
        """Get all parameters in the model as a single array. This concatenates
        the nonlinear parameters and high-order epistatic coefficients.
        Nonlinear parameters are first in the array; linear coefficients are
        second.
        """
        return np.concatenate((list(self.parameters.values()),
                               self.Linear.coef_))

    def fit(self, X='obs', y='obs',
            sample_weight=None,
            use_widgets=False,
            plot_fit=True,
            **kwargs):
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
        Also, will create IPython widgets to sweep through initial parameters
        guesses.

        This works by, first, fitting the coefficients using a linear epistasis
        model as initial guesses (along with user defined kwargs) for the
        nonlinear model.

        kwargs should be ranges of guess values for each parameter. They are
        turned into slider widgets for varying these guesses easily. The kwarg
        needs to match the name of the parameter in the nonlinear fit.
        """
        # Get pobs for nonlinear fit.
        if type(y) is str and y in ["obs", "complete"]:
            y = self.gpm.binary.phenotypes
        # Else, numpy array or dataframe
        elif type(y) == np.array or type(y) == pd.Series:
            pass
        else:
            raise FittingError("y is not valid. Must be one of the following: "
                               "'obs', 'complete', numpy.array, pandas.Series."
                               " Right now, its {}".format(type(y)))

        # Fit linear portion
        self._fit_additive(X=X, y=y, sample_weight=sample_weight)

        # Use widgets to guess the value?
        if use_widgets is False:
            # Step 2: fit nonlinear function
            self._fit_nonlinear(X=X, y=y, sample_weight=sample_weight,
                                **kwargs)

            # Step 3: fit linear, high-order model.
            self._fit_linear(X=X, y=y, sample_weight=sample_weight)
            return self

        # Don't use widgets to fit data
        else:
            import matplotlib.pyplot as plt
            import epistasis.plot
            import ipywidgets

            # Build fitting method to pass into widget box
            def fitting(**parameters):
                """Callable to be controlled by widgets."""
                # Fit the nonlinear least squares fit
                self._fit_nonlinear(
                    padd, pobs, sample_weight=sample_weight, **parameters)

                # Print score
                # print("R-squared of fit: " + str(self.score(X=Xadd, y=padd)))
                # Print parameters
                for kw in self.parameters.valuesdict():
                    print(kw + ": " + str(getattr(self.parameters, kw)))

                # Plot the nonlinear fit!
                ylin = self.Additive.predict(X=Xadd)
                # epistasis.plot.corr_resid(ylin, y, figsize=(3,5))
                # plt.show()

            # Construct and return the widget box
            widgetbox = ipywidgets.interactive(fitting, **kwargs)
            return widgetbox

    def _fit_additive(self, X='obs', y='obs', sample_weight=None, **kwargs):
        """
        """
        if hasattr(self, 'gpm') is False:
            raise Exception("This model will not work if a genotype-phenotype "
                            "map is not attached to the model class. Use the "
                            "`add_gpm` method")

        # Fit with an additive model
        self.Additive.add_epistasis()

        # Use a first order matrix only.
        if type(X) == np.ndarray or type(X) == pd.DataFrame:
            Xadd = X[:, :self.Additive.epistasis.n]
        else:
            Xadd = X

        # Fit Additive model
        self.Additive.fit(X=Xadd, y=y, sample_weight=sample_weight)
        self.Additive.epistasis.values = self.Additive.coef_

        return self

    def _fit_nonlinear(self, X='obs', y='obs', sample_weight=None, **kwargs):
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

        # Residual function to minimize.
        def residual(params, func, x, y=None):
            parvals = list(params.values())
            ymodel = func(x, *parvals)
            return y - ymodel

        # Minimize the above residual function.
        self.Nonlinear = lmfit.minimize(residual, self.parameters,
                                        args=[self.function, x], kws={'y': y})

        # Point to nonlinear.
        self.parameters = self.Nonlinear.params

    def _fit_linear(self, X='obs', y='obs', sample_weight=None):
        """"""
        # Prepare a high-order model
        self.Linear.add_epistasis()

        # Construct a linear epistasis model.
        if self.order > 1:
            xadd = self.Additive.predict(X='obs')
            ylin = self.reverse(y, *self.parameters.values(), data=xadd)
            # Now fit with a linear epistasis model.
            self.Linear.fit(X=X, y=ylin)
        else:
            self.Linear = self.Additive
        # Map to epistasis.
        self.Linear.epistasis.values = self.Linear.coef_
        return self

    def plot_fit(self):
        """Plots the observed phenotypes against the additive model
        phenotypes"""
        padd = self.Additive.predict()
        pobs = self.gpm.phenotypes
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(padd, pobs, '.b')
        plt.show()
        return fig, ax

    def predict(self, X='complete'):
        """Infer phenotypes from model coefficients and nonlinear function."""
        x = self.Linear.predict(X=X)
        y = self.function(x, *self.parameters.values())
        return y

    def score(self, X='obs', y='obs', sample_weight=None):
        """Calculates the squared-pearson coefficient for the nonlinear fit.

        Returns
        -------
        r_nonlinear : float
            squared pearson coefficient between phenotypes and nonlinear
            function.
        r_linear : float
            squared pearson coefficient between linearized phenotypes and
            linear epistasis model described by epistasis.values.
        """
        # Get pobs for nonlinear fit.
        if type(y) is str and y in ["obs", "complete"]:
            pobs = self.gpm.binary.phenotypes
        # Else, numpy array or dataframe
        elif type(y) == np.array or type(y) == pd.Series:
            pobs = y

        xlin = self.Additive.predict(X=X)
        ypred = self.function(xlin, *self.parameters.values())
        yrev = self.reverse(pobs, *self.parameters.values())
        return (pearson(pobs, ypred)**2,
                self.Linear.score(X=X, y=yrev, sample_weight=sample_weight))

    def contributions(self, X='obs', y='obs', sample_weight=None):
        """Calculate the contributions from nonlinearity and epistasis to
        the variation in phenotype.

        Returns
        -------
        contribs
        """
        # Calculate various pearson coeffs.
        add_score = self.Additive.score(X=X, y=y, sample_weight=sample_weight)
        scores = self.score(X=X, y=y, sample_weight=sample_weight)

        # Calculate the nonlinear contribution
        nonlinear_contrib = scores[0] - add_score

        # Calculate the contribution from epistasis
        epistasis_contrib = 1 - scores[0]

        # Build output dict.
        contrib = {'nonlinear': nonlinear_contrib,
                   'epistasis': epistasis_contrib}
        return contrib

    @X_predictor
    def hypothesis(self, X='complete', thetas=None):
        """Given a set of parameters, compute a set of phenotypes. Does not
        predict. This is method can be used to test a set of parameters
        (Useful for bayesian sampling).
        """
        # ----------------------------------------------------------------------
        # Part 0: Break up thetas
        # ----------------------------------------------------------------------
        # Get thetas from model.
        if thetas is None:
            thetas = self.thetas

        i, j = len(self.parameters.valuesdict()), self.Linear.epistasis.n
        parameters = thetas[:i]
        epistasis = thetas[i:i + j]

        # Part 1: Linear portion
        ylin = np.dot(X, epistasis)

        # Part 2: Nonlinear portion
        ynonlin = self.function(ylin, *parameters)

        return ynonlin

    def lnlike_of_data(self, X='obs', y='obs', yerr='obs',
                       sample_weight=None, thetas=None):
        """Calculate the log likelihoods of each data point, given a set of
        model coefficients.

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
        # ###### Prepare input #########
        # If no model parameters are given, use the model fit.
        if thetas is None:
            thetas = self.thetas

        # Handle y.
        # Get pobs for nonlinear fit.
        if type(y) is str and y in ["obs", "complete"]:
            ydata = self.gpm.binary.phenotypes
        # Else, numpy array or dataframe
        elif type(y) == np.array or type(y) == pd.Series:
            ydata = y
        else:
            raise FittingError("y is not valid. Must be one of the following:"
                               "'obs', 'complete', numpy.array, pandas.Series."
                               " Right now, its {}".format(type(y)))

        # Handle yerr.
        # Check if yerr is string
        if type(yerr) is str and yerr in ["obs", "complete"]:
            yerr = self.gpm.binary.std.upper

        # Else, numpsy array or dataframe
        elif type(y) != np.array and type(y) != pd.Series:
            raise FittingError("yerr is not valid. Must be one of the "
                               "following: 'obs', 'complete', numpy.array, "
                               "pandas.Series. Right now, its "
                               "{}".format(type(yerr)))

        # ###### Calculate likelihood #########
        # Calculate ymodel
        ymodel = self.hypothesis(X=X, thetas=thetas)

        # Likelihood of data given model
        return (- 0.5 * np.log(2 * np.pi * yerr**2) -
                (0.5 * ((ydata - ymodel)**2 / yerr**2)))


class EpistasisNonlinearLasso(EpistasisNonlinearRegression):
    """Use nonlinear regression with a linear lasso epistasis model
    to estimate epistatic coefficients and nonlinear scale in a nonlinear
    genotype-phenotype map.

    This models has three steps:
        1. Fit an additive, linear regression to approximate the average effect
        of individual mutations.
        2. Fit the nonlinear function to the observed phenotypes vs. the
        additive phenotypes estimated in step 1.
        3. Transform the phenotypes to this linear scale and fit leftover
        variation with EpistasisLasso.

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
    order : int
        order of epistasis to fit.
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
                 order=1,
                 model_type="global",
                 alpha=1.0,
                 **p0):
        # initialize model
        super(EpistasisNonlinearLasso, self).__init__(function, reverse,
                                                      order=order,
                                                      model_type=model_type)

        # Add lasso model for linear fit.
        self.Linear = EpistasisLasso(
            order=self.order, model_type=self.model_type)

    def lnlike_of_data(self, X='obs', y='obs', yerr='obs',
                       sample_weight=None, thetas=None):
        """Calculate the log likelihoods of each data point, given a set of
        model coefficients.

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
        # ###### Prepare input #########
        # If no model parameters are given, use the model fit.
        if thetas is None:
            thetas = self.thetas

        # Handle y.
        # Get pobs for nonlinear fit.
        if type(y) is str and y in ["obs", "complete"]:
            ydata = self.gpm.binary.phenotypes
        # Else, numpy array or dataframe
        elif type(y) == np.array or type(y) == pd.Series:
            ydata = y
        else:
            raise FittingError("y is not valid. Must be one of the following:"
                               "'obs', 'complete', numpy.array, pandas.Series."
                               " Right now, its {}".format(type(y)))

        # Handle yerr.
        # Check if yerr is string
        if type(yerr) is str and yerr in ["obs", "complete"]:
            yerr = self.gpm.binary.std.upper

        # Else, numpsy array or dataframe
        elif type(y) != np.array and type(y) != pd.Series:
            raise FittingError("yerr is not valid. Must be one of the "
                               "following: 'obs', 'complete', numpy.array, "
                               "pandas.Series. Right now, its "
                               "{}".format(type(yerr)))

        # ###### Calculate likelihood #########
        # Calculate ymodel
        ymodel = self.hypothesis(X=X, thetas=thetas)

        # Likelihood of data given model
        return (- 0.5 * np.log(2 * np.pi * yerr**2) -
                (0.5 * ((ydata - ymodel)**2 / yerr**2)) -
                (self.alpha * sum(abs(thetas))))
