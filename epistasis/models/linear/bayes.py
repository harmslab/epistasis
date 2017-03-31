import numpy as _np
from sklearn.naive_bayes import GaussianNB as _GaussianNB

from epistasis.models.base import BaseModel as _BaseModel
from epistasis.models.base import X_fitter as X_fitter
from epistasis.models.base import X_predictor as X_predictor

class EpistasisGaussianNB(_GaussianNB, _BaseModel):
    """ Ordinary least-squares regression of epistatic interactions.
    """
    def __init__(self, order=1, model_type="global", **kwargs):
        # Set Linear Regression settings.
        self.set_params(model_type=model_type, order=order)

    @X_fitter
    def fit(self, X=None, y=None, sample_weight=None):
        # Build input linear regression.
        super(self.__class__, self).fit(X, y, sample_weight)

    @X_predictor
    def predict(self, X=None):
        return super(self.__class__, self).predict(X)

    @X_fitter
    def score(self, X=None, y=None):
        return super(self.__class__, self).score(X, y)

    @X_fitter
    def _sample_fit(self, X=None, y=None):
        """Sample the `fit` method from phenotype standard deviations."""
        # Fit a model
        y = self.gpm.stdeviations * _np.random.randn() + self.gpm.phenotypes
        model = self.__class__()
        model.fit(X=X, y=y)
        return model.coef_

    @X_fitter
    def _sample_predict(self, X=None):
        """Sample the `predict` method from phenotype standard deviations."""
        # Fit a model
        y = self.gpm.stdeviations * _np.random.randn() + self.gpm.phenotypes
        model = self.__class__()
        model.fit(X=X, y=y)
        # predict from that model
        predictions = model.predict(X=None)
        return predictions
