import scipy
import numpy as np
import inspect
import json
from scipy.optimize import curve_fit
from .regression import EpistasisNonlinearRegression, Parameters
from ..linear.regression import EpistasisLinearRegression
from epistasis.stats import gmean
from ..base import X_fitter

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
    """Fit a power transform to linearize a genotype-phenotype map.
    """
    def __init__(self, order=1, model_type="global", fix_linear=True, **kwargs):
        # Construct parameters object
        self.parameters = Parameters(["lmbda","A","B"])
        self.set_params(order=order,
            model_type=model_type,
            fix_linear=fix_linear,
        )

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

    def _fit_(self, X=None, y=None, sample_weight=None, **kwargs):
        """Estimate the scale of multiple mutations in a genotype-phenotype map."""
        # ----------------------------------------------------------------------
        # Part 1: Estimate average, independent mutational effects and fit
        #         nonlinear scale.
        # ----------------------------------------------------------------------
        self.Additive.fit()
        x = self.Additive.predict(X=self.Additive.X)

        # Set up guesses
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
            self.Linear.fit(X=self.Linear.X, y=linearized_y)
        else:
            self.Linear = self.Additive
        self.coef_ = self.Linear.coef_
