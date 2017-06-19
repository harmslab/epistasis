import numpy as _np
from sklearn.linear_model import LinearRegression as _LinearRegression

from .base import BaseModel as _BaseModel
from .utils import X_fitter as X_fitter
from .utils import X_predictor as X_predictor

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

    @property
    def thetas(self):
        return self.coef_

    @X_fitter
    def fit(self, X=None, y=None, sample_weight=None):
        # If a threshold exists in the data, pre-classify genotypes
        return super(self.__class__, self).fit(X, y, sample_weight)

    @X_predictor
    def predict(self, X=None):
        return super(self.__class__, self).predict(X)

    @X_fitter
    def score(self, X=None, y=None):
        return super(self.__class__, self).score(X, y)

    @X_predictor
    def hypothesis(self, X=None, thetas=None):
        """Given a set of parameters, compute a set of phenotypes. This is method
        can be used to test a set of parameters (Useful for bayesian sampling).
        """
        return _np.dot(X, thetas)

    @X_predictor
    def lnlikelihood(self, X=None, thetas=None):
        """Calculate the log likelihood of the model."""
        if thetas is None:
            thetas = self.thetas
        ydata = self.gpm.phenotypes
        yerr = self.gpm.std.upper
        ymodel = self.hypothesis(X=X, thetas=thetas)
        inv_sigma2 = 1.0/(yerr**2)
        return -0.5*(np.sum((ydata-ymodel)**2*inv_sigma2 - np.log(inv_sigma2))), ymodel
