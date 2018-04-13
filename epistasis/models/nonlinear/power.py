import inspect
from functools import wraps

import scipy
import numpy as np
import pandas as pd

import lmfit
from lmfit import Parameter, Parameters

from epistasis.stats import gmean, pearson
from epistasis.models.utils import arghandler
from epistasis.models.linear.ordinary import EpistasisLinearRegression
from epistasis.models.nonlinear.ordinary import EpistasisNonlinearRegression

from .minimizer import FunctionMinimizer

# -------------------- Power Transform Function -----------------------

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

    return out

# --------------------- Power transform Minizer object -----------------------

class PowerTransformMinizer(FunctionMinimizer):
    """Minimizer class for power transform.
    """
    def __init__(self, **p0):
        # Construct parameters object
        self.parameters = Parameters()
        for p in ['lmbda', 'A', 'B']:
            # Get starting value of parameter if given.
            val = None
            if p in p0:
                val = p0[p]
            # Add parameter.
            self.parameters.add(name=p, value=val)

        # Set function
        self._function = power_transform

    def function(self, x, lmbda, A, B):
        """Execute the function."""
        return self._function(x, lmbda=lmbda, A=A, B=B, data=self.data)

    def predict(self, x):
        return self._function(x, **self.parameter, data=self.data)

    def fit(self, x, y):
        self.data = x

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
            self.minimizer = lmfit.minimize(residual, self.parameters,
                                            args=[self._function, x],
                                            kws={'y': y, 'data': self.data})
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

        self.parameters = self.minimizer.params


# -------------------- Epistasis Model -----------------------


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
    Additive : EpistasisLinearRegression
        Linear regression object for fitting additive model

    parameters : Parameters object
        Mapping object for nonlinear coefficients
    """
    def __init__(self, model_type="global", **p0):
        # Set up the function for fitting.
        self.function = power_transform
        self.minimizer = PowerTransformMinizer(**p0)
        self.parameters = self.minimizer.parameters
        self.order = 1
        self.Xbuilt = {}

        # Construct parameters object
        self.set_params(model_type=model_type)

        # Store model specs.
        self.model_specs = dict(
            function=self.function,
            model_type=self.model_type,
            **p0)

        # Set up additive and high-order linear model
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)
