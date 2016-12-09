import numpy as np
from sklearn.linear_model import LogisticRegression
from ..base import BaseModel

def X_helper(method):
    def preprocess(self, X=None):

        # Build input to

        if X is None:
            X = self.X_helper(
                genotypes=self.gpm.binary.complete_genotypes,
                **self.get_params())



        return method(self, X)
    return preprocess


class EpistasisLogisticRegression(LogisticRegression, BaseModel):
    """ Logistic Regression used to categorize phenotypes as either alive or dead.
    """
    def __init__(self, threshold, order=1, model_type="global", n_jobs=1, **kwargs):
        # Set Linear Regression settings.
        super(EpistasisLogisticRegression, self).__init__()
        self.set_params(threshold=threshold, model_type=model_type, order=order, **kwargs)

    def fit(self, X=None, y=None):
        # Build input linear regression.
        if y is None:
                y = np.array(self.gpm.phenotypes)
                y[ y < self.threshold ] = 0
                y[ y >= self.threshold ] = 1
        if X is None:
            # Build X AND EpistasisMap attachment.
            X = self.X_helper(
                genotypes=self.gpm.binary.genotypes,
                **self.get_params())
            self.X = X
            # fit linear regression.
            super(self.__class__, self).fit(X, y)
            self.epistasis.values = self.coef_ # point epistasis map to coef
        else:
            super(self.__class__, self).fit(X, y)

    @X_helper
    def predict(self, X=None):
        return super(self.__class__, self).predict(X)

    @X_helper
    def predict_log_proba(self, X=None):
        return super(self.__class__, self).predict_log_proba(X)

    @X_helper
    def predict_proba(self, X=None):
        return super(self.__class__, self).predict_proba(X)

    def score(self, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.gpm.phenotypes
        return super(self.__class__, self).score(X, y)
