from functools import wraps
import numpy as np

from .base import BaseModel
from .linear.classification import EpistasisLogisticRegression
from .linear.regression import EpistasisLinearRegression
from .nonlinear.power import EpistasisPowerTransform

def gpm_check(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "gpm") is False:
            raise Exception("A GenotypePhenotypeMap object must be attached")
        return method(self, *args, **kwargs)
    return wrapper

class EpistasisTwoStepRegression(BaseModel):
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
    def __init__(self, threshold, order=1, model_type="global", fix_linear=True, remove_zeros=True, **kwargs):
        self.remove_zeros = remove_zeros
        self.threshold = threshold
        self.order = order
        self.model_type = model_type
        self.LogisticRegression = EpistasisLogisticRegression(threshold, order=order, model_type=model_type)
        self.PowerTransform = EpistasisPowerTransform(order=1, model_type=model_type, fix_linear=fix_linear)
        self.LinearRegression = EpistasisLinearRegression(order=order, model_type=model_type)

    @gpm_check
    def fit(self, X=None, y=None, sample_weight=None, **kwargs):
        # If no weights are given, determine weights from logistic regression.
        y_ = np.ones(self.gpm.n)
        y = self.gpm.phenotypes
        y_[ self.gpm.phenotypes < self.threshold ] = 0
        X = self.X_constructor(self.gpm.binary.genotypes)
        X_ = self.PowerTransform.X_constructor(self.gpm.binary.genotypes)
        self.LogisticRegression.fit(X, y_)
        
        if sample_weight is None:
            # Fit the logistic regression
            sample_weight = self.LogisticRegression.predict_proba(X)[:,1]

        if self.remove_zeros:
            subset = np.where( y_ == 1 )[0]
            y = y[subset]
            X = X[subset]
            X_ = X_[subset]
            sample_weight = sample_weight[subset]

        # Fit a nonlinear model.
        self.PowerTransform.fit(X=X_, y=y, sample_weight=sample_weight, **kwargs)
        self.parameters = self.PowerTransform.parameters

        # Fit linear epistasis model
        ylin = self.PowerTransform.transform_target(y)
        self.LinearRegression.fit(X, ylin, sample_weight=sample_weight)
        self.epistasis.values = self.LinearRegression.coef_

    @gpm_check
    def predict(self, X=None):
        if X is None:
            X = self.LinearRegression.X_constructor(self.gpm.binary.complete_genotypes)
        y_linear = self.LinearRegression.predict(X)
        y_nonlinear = self.PowerTransform.function(y_linear, *self.parameters.get_params())
        y_predicted = y_nonlinear * self.LogisticRegression.predict(X)
        return y_predicted
