import scipy
import numpy as np
import inspect
import json
from scipy.optimize import curve_fit
from .regression import EpistasisNonlinearRegression, Parameters
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

    def _params_to_json(self, filename):
        """Temporary method. will be deprecated soon!!!!!
        """
        params = self.parameters()
        params.update(gmean=self.gmean)
        with open(filename, "w") as f:
            json.dump(params, f)

    @X_fitter
    def _fit_(self, X=None, y=None, sample_weight=None, **kwargs):
        """Fit the genotype-phenotype map for epistasis.
        """
        # Fit coeffs
        guess = self._guess_model(X, y, sample_weight=sample_weight)
        self.coef_ = guess.coef_
        # Get the scale of the map.
        x = guess.predict(X)
        # Construct an array of guesses, using the scale specified by user.
        lmbda = 1
        A = 0
        B = min(self.gpm.phenotypes)
        guess = [lmbda, A, B]
        # Add model's extra parameter guesses to input array
        for kw in kwargs:
            index = self.parameters._mapping[kw]
            guess[index] = kwargs[kw]
        # create a sigma array for weighted fits with curve fit.
        if sample_weight is None:
            sigma = None
        else:
            sigma = 1 / np.sqrt(sample_weight)
        # Curve fit the data using a nonlinear least squares fit
        popt, pcov = curve_fit(self._function, x, y, p0=guess, sigma=sigma,
            bounds=([-np.inf, -np.inf, -np.inf],[np.inf, np.inf, min(self.gpm.phenotypes)]))
        for i in range(0, self.parameters.n):
            self.parameters._set_param(i, popt[i])
