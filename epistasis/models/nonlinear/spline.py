import numpy as np

from .minimizer import Minimizer
from .ordinary import EpistasisNonlinearRegression
from epistasis.models import EpistasisLinearRegression
from epistasis.models.utils import (arghandler, FittingError)
from scipy.interpolate import UnivariateSpline
from lmfit import Parameter, Parameters


# -------------------- Minimizer object ------------------------

class SplineMinizer(Minimizer):
    """Spline Fitter.
    """
    def __init__(self, k=3, s=None):
        self.k = k
        self.s = s
        # Set initalize parameters to zero.
        self.parameters = Parameters()
        for i in range(self.k+1):
            self.parameters.add(name='c{}'.format(i), value=0)

    def _sorter(self, x, y=None, tol=1e-5):
        """sort x (and y) according to x values

        The spline call requires that x must be increasing. This function
        sorts x. If there are non-unique values, it adds a small numerical
        difference using tol.
        """
        # Copy x to leave it unchanged.
        x_ = np.copy(x)

        # Get non-unique terms
        idx = np.arange(len(x_))
        u, u_idx = np.unique(x, return_index=True)
        idx = np.delete(idx, u_idx)

        # Add noise to non-unique
        x_[idx] = x_[idx] + np.random.uniform(1,9.9, size=len(idx))*tol

        # Now sort x
        idx = np.argsort(x_)

        if y is None:
            return x_[idx]
        else:
            return x_[idx], y[idx]

    def function(self, x, *coefs):
        # Order of polynomial
        k = self.k

        # Number of coefficients
        n = k + 1

        # Knot positions
        t_arr = np.zeros(n*2, dtype=float)
        t_arr[:n] = min(x)
        t_arr[n:] = max(x)

        # Coefficients
        c_arr = np.zeros(n*2, dtype=float)
        c_arr[:n] = coefs

        # Build Spline function
        tck = [t_arr, c_arr, k]
        model = UnivariateSpline._from_tck(tck)

        # Return predicted function
        return model(x)

    def predict(self, x):
        return self._spline(x)

    def transform(self, x, y):
        ymodel = self.predict(x)
        return (y - ymodel) + x

    def fit(self, x, y):
        # Sort values for fit
        x_, y_ = self._sorter(x, y)

        # Fit spline.
        self._spline = UnivariateSpline(
            x=x_,
            y=y_,
            k=self.k,
            s=self.s
        )

        for i, coef in enumerate(self._spline.get_coeffs()):
            self.parameters['c{}'.format(i)].value = coef

# -------------------- Minimizer object ------------------------

class EpistasisSpline(EpistasisNonlinearRegression):
    """Estimate nonlinearity in a genotype-phenotype map using a spline.

    Parameters
    ----------
    k : int
        order of spline.

    s : float, optional
        smoothness factor.
    """
    def __init__(self, k=3, s=None, model_type="global"):
        # Set atributes
        self.k = k
        self.s = s

        # Set up the function for fitting.
        self.minimizer = SplineMinizer(k=self.k, s=self.s)
        self.order = 1
        self.Xbuilt = {}

        # Construct parameters object
        self.set_params(model_type=model_type)

        # Store model specs.
        self.model_specs = dict(model_type=self.model_type)

        # Set up additive and high-order linear model
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)
