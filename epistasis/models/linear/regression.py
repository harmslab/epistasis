import numpy as _np
from sklearn.linear_model import LinearRegression as _LinearRegression

from epistasis.models.base import BaseModel as _BaseModel
from epistasis.models.base import X_fitter as X_fitter
from epistasis.models.base import X_predictor as X_predictor

# Suppress an annoying error from scikit-learn
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

class EpistasisLinearRegression(_LinearRegression, _BaseModel):
    """Ordinary least-squares regression of epistatic interactions.
    """
    def __init__(self, order=1, model_type="global", n_jobs=1, **kwargs):
        # Set Linear Regression settings.
        self.fit_intercept = False
        self.normalize = False
        self.copy_X = False
        self.n_jobs = n_jobs
        self.set_params(model_type=model_type, order=order)

    @X_fitter
    def fit(self, X=None, y=None, sample_weight=None):
        # Build input linear regression.
        super(self.__class__, self).fit(X, y, sample_weight)
        self._score = self.score(X,y)

    @X_predictor
    def predict(self, X=None):
        return super(self.__class__, self).predict(X)

    @X_fitter
    def score(self, X=None, y=None):
        return super(self.__class__, self).score(X, y)

    def hypothesis(self, thetas):
        """Given a set of parameters, compute a set of phenotypes. This is method
        can be used to test a set of parameters (Useful for bayesian sampling).
        """
        if hasattr(self, "X") is False:
            raise Exception("A model matrix X needs to be attached to the model. "
                "Try calling `X_constructor()`.")
        return _np.dot(self.X, thetas)
