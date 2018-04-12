import numpy as np

from .minimizer import Minimizer
from .ordinary import EpistasisNonlinearRegression
from epistasis.models import EpistasisLinearRegression
from epistasis.models.utils import (arghandler, FittingError)
from scipy.interpolate import UnivariateSpline

# -------------------- Minimizer object ------------------------

class SplineMinizer(Minimizer):
    """Spline Fitter.
    """
    def __init__(self, k=3, s=None):
        self.k = k
        self.s = s
        self.parameters = None

    def _sorter(self, x, y=None):
        """sort x (and y) according to x values"""
        idx = np.argsort(x)
        if y is None:
            return x[idx]
        else:
            return x[idx], y[idx]

    def predict(self, x):
        return self._spline(x)

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

    def transform(self, x, y):
        ymodel = self.predict(x)
        return (y - ymodel) + x

# -------------------- Minimizer object ------------------------

class EpistasisSpline(EpistasisNonlinearRegression):
    """Epistasis Spline method.
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
