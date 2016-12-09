import numpy as np
from sklearn.linear_model import LogisticRegression
from ..base import BaseModel, X_fitter, X_predictor

class EpistasisLogisticRegression(LogisticRegression, BaseModel):
    """Logistic Regression used to categorize phenotypes as either alive or dead.
    """
    def __init__(self, threshold, order=1, model_type="global", n_jobs=1, **kwargs):
        # Set Linear Regression settings.
        super(EpistasisLogisticRegression, self).__init__()
        self.set_params(threshold=threshold, model_type=model_type, order=order, **kwargs)

    @X_fitter
    def fit(self, X=None, y=None):
        # Build input linear regression.
        super(self.__class__, self).fit(X,y)

    @X_predictor
    def predict(self, X=None):
        return super(self.__class__, self).predict(X)

    @X_predictor
    def predict_log_proba(self, X=None):
        return super(self.__class__, self).predict_log_proba(X)

    @X_predictor
    def predict_proba(self, X=None):
        return super(self.__class__, self).predict_proba(X)

    @X_fitter
    def score(self, X=None, y=None):
        return super(self.__class__, self).score(X, y)
