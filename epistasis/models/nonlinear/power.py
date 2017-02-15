import scipy
import numpy as np
import inspect
import json
from scipy.optimize import curve_fit
from .regression import EpistasisNonlinearRegression
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
    """"""
    def __init__(self, order=1, model_type="global", fix_linear=False, **kwargs):
        super(EpistasisPowerTransform, self).__init__(
            function=power_transform,
            reverse=self.reverse,
            order=order,
            model_type=model_type,
            fix_linear=fix_linear,
            **kwargs)

        def function(x, lmbda, A, B):
            """Power transformation function. Ignore zeros in gmean calculation"""
            # Check for zeros
            gm = gmean(x+A)
            if lmbda == 0:
                return gm*np.log(x + A)
            else:
                first = (x+A)**lmbda
                out = (first - 1.0)/(lmbda * gm**(lmbda-1)) + B
            return out

        self.function = function

    @property
    def gmean(self):
        linear = np.dot(self.X, self.coef_)
        return gmean(linear + self.parameters.A)

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
        guess = np.ones(self.parameters.n)
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
