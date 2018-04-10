import inspect
import json

import scipy
import numpy as np
import pandas as pd

import lmfit
from lmfit import Parameter, Parameters

from .utils import arghandler
from ..stats import gmean, pearson
from .linear import EpistasisLinearRegression, EpistasisLasso
from .nonlinear import (EpistasisNonlinearRegression,
                        EpistasisNonlinearLasso,
                        Parameters)

from gpmap import GenotypePhenotypeMap

# Suppress an annoying error
import warnings
# warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def power_transform(x, lmbda, A, B, data=None):
    """Transform x according to a power transformation.

    Note, this functions calculates the geometric mean of x
    to center the power transform on the data. If you'd like to calculate
    the geometric mean on a different array than x (perhaps some
    other data) pass that ohter array to the data keyword argument.

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
    #print(lmbda, A, B)
    return out


def reverse_power_transform(y, lmbda, A, B, data=None):
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
    def __init__(self, model_type="global", **p0):
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
        self.order = 1
        self.Xbuilt = {}

        # Construct parameters object
        self.set_params(model_type=model_type)

        # Store model specs.
        self.model_specs = dict(
            model_type=self.model_type,
            **p0)

        # Set up additive and high-order linear model
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)

    @arghandler
    def _fit_nonlinear(self, X=None, y=None, **kwargs):
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
        xadd = self.Additive.predict(X=X)

        # Set guesses
        for key, value in kwargs.items():
            self.parameters[key].set(value=value)

        # Set the lower bound on B.
        self.parameters['A'].set(min=-min(x))

        # Store residual steps in case fit fails.
        last_residual_set = None

        # Residual function to minimize.
        def residual(params, func, x, y=None, data=None):
            # Fit model
            parvals = list(params.values())
            ymodel = func(x, *parvals, data=data)

            # Store items in case of error.
            nonlocal last_residual_set
            last_residual_set = (params, ymodel)

            return y - ymodel

        # Minimize the above residual function.
        try:
            self.Nonlinear = lmfit.minimize(residual, self.parameters,
                                            args=[self.function, x],
                                            kws={'y': y, 'data': xadd})
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
        # Fit method.
        self.fit(X=X, y=y, **kwargs)

        xdata = self.Additive.predict(X='fit')

        linear_phenotypes = self.reverse(y, *self.parameters.values(), data=xdata)

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
        xadd = self.Additive.predict(X='fit')
        y = self.function(x, *self.parameters.values(), data=xadd)
        return y

    def predict_transform(self, X=None, y=None):
        xdata = self.Additive.predict(X='fit')
        if y is None:
            x = self.Additive.predict(X=X)
        else:
            x = y
        return self.function(x, *self.parameters.values(), data=xdata)

    @arghandler
    def score(self, X=None, y=None):
        xadd = self.Additive.predict(X=X)
        ypred = self.function(xadd, *self.parameters.values(), data=xadd)
        return pearson(y, ypred)**2

    @arghandler
    def hypothesis(self, X=None, thetas=None):
        # Break up thetas
        i, j = len(self.parameters.valuesdict()), self.Additive.epistasis.n
        parameters = thetas[:i]
        epistasis = thetas[i:i + j]

        # Get the data that was used to estimate the geometric mean.
        xdata = self.Additive.predict(X='fit')

        # Part 1: Linear portion
        x = self.Additive.hypothesis(X=X, thetas=epistasis)

        # Part 2: Nonlinear portion
        ynonlin = self.function(x, *parameters, data=xdata)

        return ynonlin

    def hypothesis_transform(self, X=None, y=None, thetas=None):
        # Break up thetas
        i, j = len(self.parameters.valuesdict()), self.Additive.epistasis.n
        parameters = thetas[:i]
        epistasis = thetas[i:i + j]

        # Estimate additive coefficients
        xdata = self.Additive.predict(X='fit')
        # Part 2: Nonlinear portion
        if y is None:
            x = self.Additive.hypothesis(X=X, thetas=epistasis)
        else:
            x = y
        y_transform = self.function(x, *parameters, data=xdata)
        return y_transform

    @arghandler
    def lnlike_of_data(self, X=None, y=None, yerr=None, thetas=None):
        # ###### Calculate likelihood #########
        # Calculate ymodel
        ymodel = self.hypothesis(X=X, thetas=thetas)

        # Likelihood of data given model
        return (- 0.5 * np.log(2 * np.pi * yerr**2) -
               (0.5 * ((y - ymodel)**2 / yerr**2)))


class EpistasisPowerTransformLasso(EpistasisPowerTransform):
    """Use power-transform function, via nonlinear least-squares regression,
    and an epistasis lasso model to estimate epistatic coefficients and the
    nonlinear scale in a nonlinear genotype-phenotype map.

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

    Additive : EpistasisLinearRegression
        Linear regression object for fitting additive model

    parameters : Parameters object
        Mapping object for nonlinear coefficients
    """
    def __init__(self, model_type="global", alpha=1.0, **p0):
        super(EpistasisPowerTransformLasso, self).__init__(
            model_type=model_type, **p0)

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
                (self.Linear.alpha * sum(abs(thetas))))
