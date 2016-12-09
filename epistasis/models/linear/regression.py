import numpy as _np
from sklearn.linear_model import LinearRegression as _LinearRegression

from epistasis.models.base import BaseModel as _BaseModel

class EpistasisLinearRegression(_LinearRegression, _BaseModel):
    """ Ordinary least-squares regression of epistatic interactions.
    """
    def __init__(self, order=1, model_type="global", n_jobs=1, **kwargs):
        # Set Linear Regression settings.
        self.fit_intercept = False
        self.normalize = False
        self.copy_X = False
        self.n_jobs = n_jobs
        self.set_params(model_type=model_type, order=order, **kwargs)

    def fit(self, X=None, y=None):
        # Build input linear regression.
        if y is None:
                y = self.gpm.phenotypes
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

    def predict(self, X=None):
        # Build input to
        if X is None:
            X = self.X_helper(
                genotypes=self.gpm.binary.complete_genotypes,
                **self.get_params())
        # fit linear regression.
        return super(self.__class__, self).predict(X)

    def score(self, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.gpm.phenotypes
        return super(self.__class__, self).score(X, y)
