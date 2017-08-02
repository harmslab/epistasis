import scipy
import numpy as np
import inspect
import json
from scipy.optimize import curve_fit

from .nonlinear import EpistasisNonlinearRegression, Parameters
from .linear import EpistasisLinearRegression
from .utils import X_fitter
from epistasis.stats import gmean

# Suppress an annoying error
import warnings
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def power_transform(x, lmbda, A, B):
    """Power transformation function. Ignore zeros in gmean calculation"""
    # Check for zeros
    gm = gmean(x+A)
    if lmbda == 0:
        return gm*np.log(x + A)
    else:
        first = (x+A)**lmbda
        out = (first - 1.0)/(lmbda * gm**(lmbda-1)) + B
    return out

class EpistasisPowerTransform(EpistasisNonlinearRegression):
    """Use power-transform function, via nonlinear least-squares regression,
    to estimate epistatic coefficients and the nonlinear scale in a nonlinear
    genotype-phenotype map.

    This models has three steps:
        1. Fit an additive, linear regression to approximate the average effect of
            individual mutations.
        2. Fit the nonlinear function to the observed phenotypes vs. the additive
            phenotypes estimated in step 1.
        3. Transform the phenotypes to this linear scale and fit leftover variation
            with high-order epistasis model.

    Methods are described in the following publication:
        Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in Nonlinear
        Genotype-Phenotype Maps'. Genetics 205, 1079-1088 (2017).

    Parameters
    ----------
    order : int
        order of epistasis to fit.
    model_type : str (default: global)
        type of epistasis model to use. See paper above for more information.

    Keyword Arguments
    -----------------
    Keyword arguments are interpreted as intial guesses for the nonlinear function
    parameters. Must have the same name as parameters in the nonlinear function

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
    def __init__(self, order=1, model_type="global", fix_linear=True, **p0):
        # Construct parameters object
        self.parameters = Parameters(["lmbda","A","B"])
        self.set_params(order=order,
            model_type=model_type,
            fix_linear=fix_linear,
        )
        # Initial parameters guesses
        self.p0 = p0

    def function(self, x, lmbda, A, B):
        """Power transformation function. Exposed to the user for transforming
        test data (not training data.)
        """
        # Check for zeros
        if lmbda == 0:
            return self.gmean*np.log(x + A)
        else:
            first = (x+A)**lmbda
            out = (first - 1.0)/(lmbda * self.gmean**(lmbda-1)) + B
        return out

    def _function(self, x, lmbda, A, B):
        """Internal power transformation function. Note that this method actually
        changes the geometric mean while fitting. This not exposed to the user,
        and should only be used by the `fit` method.

        This will fit the geometric mean of the training set and will be used
        in `function` to transform test data.
        """
        # Check for zeros
        self._gmean = gmean(x+A)
        if lmbda == 0:
            return self._gmean*np.log(x + A)
        else:
            first = (x+A)**lmbda
            out = (first - 1.0)/(lmbda * self._gmean**(lmbda-1)) + B
        return out

    @property
    def gmean(self):
        try:
            return self._gmean
        except AttributeError:
            raise AttributeError("The geometric mean is unknown. Call `fit` before"
            " looking for this method.")

    def reverse(self, y, lmbda, A, B):
        """reverse transform"""
        gmean = self.gmean
        return (gmean**(lmbda-1)*lmbda*(y - B) + 1)**(1/lmbda) - A

    def hypothesis(self, X=None, thetas=None):
        """Given a set of parameters, compute a set of phenotypes. Does not predict. This is method
        can be used to test a set of parameters (Useful for bayesian sampling).
        """
        y = super(EpistasisPowerTransform, self).hypothesis(X=X, thetas=thetas)
        # NOTE: sets nan values to the saturation point.
        y[np.isnan(y)==True] = self.parameters.B
        return y

    def predict(self, X=None):
        """Predict new targets from model."""
        y = super(EpistasisPowerTransform, self).predict(X=X)
        # NOTE: sets nan values to the saturation point.
        y[np.isnan(y)==True] = self.parameters.B
        return y

    def _fit_(self, X=None, y=None, sample_weight=None, **kwargs):
        """Estimate the scale of multiple mutations in a genotype-phenotype map."""
        # ----------------------------------------------------------------------
        # Part 1: Estimate average, independent mutational effects and fit
        #         nonlinear scale.
        # ----------------------------------------------------------------------
        self.Additive.fit(y=y)
        x = self.Additive.predict(X=self.Additive.Xfit)

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
        else:
            sigma = 1 / np.sqrt(sample_weight)

        # Fit with curve_fit, using
        popt, pcov = curve_fit(self._function, x, y, p0=guesses, sigma=sigma,
            bounds=([-np.inf, -np.inf, -np.inf],[np.inf, np.inf, min(self.gpm.phenotypes)]))

        for i in range(0, self.parameters.n):
            self.parameters._set_param(i, popt[i])

        # ----------------------------------------------------------------------
        # Part 3: Fit high-order, linear model.
        # ----------------------------------------------------------------------

        # Construct a linear epistasis model.
        if self.order > 1:
            linearized_y = self.reverse(y, *self.parameters.values)
            # Now fit with a linear epistasis model.
            self.Linear.fit(X=self.Linear.Xfit, y=linearized_y)
        else:
            self.Linear = self.Additive
        self.coef_ = self.Linear.coef_
