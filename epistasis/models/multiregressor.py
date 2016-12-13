from functools import wraps

from .base import BaseModel
from .linear.classification import EpistasisLogisticRegression
from .linear.regression import EpistasisLinearRegression
from .nonlinear.power import EpistasisPowerTransform

def gpm_check(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "gpm") is False:
            raise Exception("A GenotypePhenotypeMap object must be attached")
        return method(*args, **kwargs)
    return wrapper

class EpistasisPhenotypePredictor(BaseModel):
    """

    Fitting
    -------
    1. Regress for zero-inflated data.

    2. Fit scale in non-zero phenotypes

    3. Regress epistasis.

    Predicting
    ----------

    1. new x --> no scale

    2. Add scale

    3.
    """
    def __init__(self, order=1, model_type="global", **kwargs):
        """
        """
        self.LogisticRegression = EpistasisLogisticRegression(order=1, model_type="global")
        self.PowerTransform = EpistasisPowerTransform(order=1, model_type="global")
        self.LinearRegression = EpistasisLinearRegression(model_type="global")

    @gpm_check
    def fit_class(self, X=None, y=None):
        y = self.gpm.phenotypes
        self.LogisticRegression.fit(X=X, y=y)

    @gpm_check
    def fit_power_transform(self, X=None, y=None, **kwargs):
        self.Power

    @gpm_check
    def fit_epistasis(self):
