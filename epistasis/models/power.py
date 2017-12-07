import scipy
import numpy as np
import inspect
import json
import pandas as pd
from scipy.optimize import curve_fit

from .utils import X_fitter
from epistasis.stats import gmean
from .linear import EpistasisLinearRegression, EpistasisLasso
from .nonlinear import (EpistasisNonlinearRegression,
                        EpistasisNonlinearLasso,
                        Parameters)

# Suppress an annoying error
import warnings
# warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def power_transform(x, lmbda, A, B):
    """Power transformation function. Ignore zeros in gmean calculation"""
    # Check for zeros
    gm = gmean(x + A)
    if lmbda == 0:
        return gm * np.log(x + A)
    else:
        first = (x + A)**lmbda
        out = (first - 1.0) / (lmbda * gm**(lmbda - 1)) + B
    return out


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
        self.parameters = Parameters(["lmbda", "A", "B"])
        self.set_params(order=order,
                        model_type=model_type,
                        )
        # Initial parameters guesses
        self.p0 = p0

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

    def function(self, x, lmbda, A, B):
        """Power transformation function. Exposed to the user for transforming
        test data (not training data.)
        """
        # Check for zero
        if lmbda == 0:
            return self.gmean * np.log(x + A)
        else:
            first = (x + A)**lmbda
            out = (first - 1.0) / (lmbda * self.gmean**(lmbda - 1)) + B
        return out

    def _function(self, x, lmbda, A, B):
        """Internal power transformation function. Note that this method actually
        changes the geometric mean while fitting. This not exposed to the user,
        and should only be used by the `fit` method.

        This will fit the geometric mean of the training set and will be used
        in `function` to transform test data.
        """
        # Check for zeros
        self._gmean = gmean(x + A)
        if lmbda == 0:
            return self._gmean * np.log(x + A)
        else:
            first = (x + A)**lmbda
            out = (first - 1.0) / (lmbda * self._gmean**(lmbda - 1)) + B
        return out

    @property
    def gmean(self):
        try:
            return self._gmean
        except AttributeError:
            raise AttributeError("The geometric mean is unknown. Call `fit` "
                                 "before looking for this method.")

    def reverse(self, y, lmbda, A, B):
        """reverse transform"""
        gmean = self.gmean
        return (gmean**(lmbda - 1) * lmbda * (y - B) + 1)**(1 / lmbda) - A

    def hypothesis(self, X='obs', thetas=None):
        """Given a set of parameters, compute a set of phenotypes. Does not
        predict. This is method can be used to test a set of parameters
        (Useful for bayesian sampling).
        """
        y = super(EpistasisPowerTransform, self).hypothesis(
            X=X, thetas=thetas)
        # NOTE: sets nan values to the saturation point.
        # y[np.isnan(y)==True] = self.parameters.B
        return y

    def predict(self, X='complete'):
        """Predict new targets from model."""
        y = super(EpistasisPowerTransform, self).predict(X=X)
        # NOTE: sets nan values to the saturation point.
        # y[np.isnan(y)==True] = self.parameters.B
        return y

    def _fit_nonlinear(self, X='obs', y='obs', sample_weight=None,
                       fit_gmean=True, **kwargs):
        """Estimate the scale of multiple mutations in a genotype-phenotype
        map.
        """
        # Use a first order matrix only.
        if type(X) == np.ndarray or type(X) == pd.DataFrame:
            Xadd = X[:, :self.Additive.epistasis.n]
        else:
            Xadd = X

        x = self.Additive.predict(X=Xadd)

        # Set up guesses
        self.p0.update(**kwargs)
        kwargs = self.p0

        guesses = np.ones(self.parameters.n)
        for kw in kwargs:
            index = self.parameters._mapping[kw]
            guesses[index] = kwargs[kw]

        # Convert weights to variances on fit parameters.
        if sample_weight is None:
            sigma = None
        elif type(sample_weight) == str and sample_weight == 'relative':
            sigma = 1 / sample_weight
        else:
            sigma = 1 / sample_weight

        # Fit with curve_fit, using
        if fit_gmean:
            f = self._function
        else:
            try:
                f = self.function
            except AttributeError:
                raise AttributeError("gmean needs to be calculated. Set "
                                     "`fit_gmean` to True.")

        popt, pcov = curve_fit(f, x, y, p0=guesses, sigma=sigma,
                               bounds=([-np.inf, -np.inf, -np.inf],
                                       [np.inf, np.inf,
                                        min(self.gpm.phenotypes)]))

        for i in range(0, self.parameters.n):
            self.parameters._set_param(i, popt[i])

        return self


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
