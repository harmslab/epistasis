import inspect
import json

import scipy
import numpy as np
import pandas as pd

import lmfit
from lmfit import Parameter, Parameters

from .utils import X_fitter, X_predictor
from ..stats import gmean, pearson
from .linear import EpistasisLinearRegression, EpistasisLasso
from .nonlinear import (EpistasisNonlinearRegression,
                        EpistasisNonlinearLasso,
                        Parameters)

# Suppress an annoying error
import warnings
# warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def power_transform(x, lmbda, A, B, data=None):
    """Transform x according to a power transformation.

    Note, the power transform calculates the geometric mean of x
    to center the curve on that point. If you'd like to calculate
    the geometric mean on a different array than x (perhaps some
    real data) pass that ohter array to the data keyword argument.

    .. math::
        y = \\frac{ x^{\\lambda} - 1 }{\\lambda [GM(x)]^{\\lambda - 1}}

    Parameters
    ----------
    x : array-like
        data to transform.
    lmbda : float
        power parameter.
    A : float
        horizontal translation constant.
    B : float
        vertical translation constant.
    data : array-like (default=None)
        data to calculate the geometric mean.
    """
    # Calculate the GMean on the data
    if data is None:
        gm = gmean(x + A)
    else:
        gm = gmean(data + A)

    # Check for zeros
    if lmbda == 0:
        return gm * np.log(x + A)
    else:
        first = (x + A)**lmbda
        out = (first - 1.0) / (lmbda * gm**(lmbda - 1)) + B
    return out


def reverse_power_transform(y, lmbda, A, B, data):
    """The reverse of ``power_transform``.

    Note, the power transform calculates the geometric mean of x
    to center the curve on that point. If you'd like to calculate
    the geometric mean on a different array than x (perhaps some
    real data) pass that ohter array to the data keyword argument.
    """
    # Calculate the GMean on the data
    gm = gmean(data + A)
    return (gm**(lmbda - 1) * lmbda * (y - B) + 1)**(1 / lmbda) - A


class EpistasisPowerTransform(EpistasisNonlinearRegression):
    """Use power-transform function, via nonlinear least-squares regression,
    to estimate epistatic coefficients and the nonlinear scale in a nonlinear
    genotype-phenotype map.

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
    order : int
        order of epistasis to fit.
    model_type : str (default: global)
        type of epistasis model to use. See paper above for more information.

    Keyword Arguments
    -----------------
    Keyword arguments are interpreted as intial guesses for the nonlinear
    function parameters. Must have the same name as parameters in the
    nonlinear function

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

    def __init__(self, order=1, model_type="global", **p0):
        # Construct parameters object
        self.parameters = Parameters()
        for p in ['lmbda', 'A', 'B']:
            # Get starting value of parameter if given.
            val = None
            if p in p0:
                val = p0[p]
            # Add parameter.
            self.parameters.add(name=p, value=val)

        # Save functions
        self.function = power_transform
        self.reverse = reverse_power_transform

        # Construct parameters object
        self.set_params(order=order,
                        model_type=model_type)

        # Store model specs.
        self.model_specs = dict(
            order=self.order,
            model_type=self.model_type,
            **p0)

        # Set up additive and high-order linear model
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)
        self.Linear = EpistasisLinearRegression(
            order=self.order, model_type=self.model_type)

    def _fit_nonlinear(self, X='obs', y='obs', sample_weight=None, **kwargs):
        """Estimate the scale of multiple mutations in a genotype-phenotype
        map."""
        # Use a first order matrix only.
        if type(X) == np.ndarray or type(X) == pd.DataFrame:
            Xadd = X[:, :self.Additive.epistasis.n]
        else:
            Xadd = X

        # Predict additive phenotypes that passed the classifier.
        x = self.Additive.predict(X=Xadd)

        # Get data used to approximate x_add
        xadd = self.Additive.predict(X='fit')

        # Set guesses
        for key, value in kwargs.items():
            self.parameters[key].set(value=value)

        # Set the lower bound on B.
        self.parameters['A'].set(min=-min(x))

        # Residual function to minimize.
        def residual(params, func, x, y=None, data=None):
            parvals = list(params.values())
            ymodel = func(x, *parvals, data=data)
            return y - ymodel

        # Minimize the above residual function.
        self.Nonlinear = lmfit.minimize(residual, self.parameters,
                                        args=[self.function, x],
                                        kws={'y': y, 'data': xadd})

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

    def predict(self, X='complete'):
        """Infer phenotypes from model coefficients and nonlinear function."""
        xadd = self.Additive.predict(X='fit')
        x = self.Linear.predict(X=X)
        y = self.function(x, *self.parameters.values(), data=xadd)
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

        xadd = self.Additive.predict(X='fit')
        ypred = self.function(xadd, *self.parameters.values(), data=xadd)
        yrev = self.reverse(pobs, *self.parameters.values(), data=xadd)
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

        # Get the data that was used to estimate the geometric mean.
        xadd = self.Additive.predict(X='fit')

        # Part 1: Linear portion
        ylin = np.dot(X, epistasis)

        # Part 2: Nonlinear portion
        ynonlin = self.function(ylin, *parameters, data=xadd)

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


class EpistasisPowerTransformLasso(EpistasisPowerTransform):
    """Use power-transform function, via nonlinear least-squares regression,
    and an epistasis lasso model to estimate epistatic coefficients and the
    nonlinear scale in a nonlinear genotype-phenotype map.

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
    order : int
        order of epistasis to fit.
    model_type : str (default: global)
        type of epistasis model to use. See paper above for more information.

    Keyword Arguments
    -----------------
    Keyword arguments are interpreted as intial guesses for the nonlinear
    function parameters. Must have the same name as parameters in the
    nonlinear function

    Attributes
    ----------
    epistasis : EpistasisMap
        Mapping object containing high-order epistatic coefficients
    Linear : EpistasisLasso
        Linear regression object for fitting high-order epistasis model
    Additive : EpistasisLinearRegression
        Linear regression object for fitting additive model
    parameters : Parameters object
        Mapping object for nonlinear coefficients
    """
    def __init__(self, order=1, model_type="global", alpha=1.0, **p0):
        super(EpistasisPowerTransformLasso, self).__init__(
            order=order, model_type=model_type, **p0)

        # Set lasso.
        self.Linear = EpistasisLasso(
            order=self.order, model_type=self.model_type, alpha=alpha)

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
                (self.Linear.alpha * sum(abs(thetas))))
