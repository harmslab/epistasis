import numpy as _np
from sklearn.linear_model import LinearRegression as _LinearRegression

from .base import BaseModel as _BaseModel
from .utils import X_fitter as X_fitter
from .utils import X_predictor as X_predictor

# Suppress an annoying error from scikit-learn
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

class EpistasisLinearRegression(_LinearRegression, _BaseModel):
    """Ordinary least-squares regression for estimating high-order, epistatic
    interactions in a genotype-phenotype map.

    Methods are described in the following publication:
        Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in Nonlinear
        Genotype-Phenotype Maps'. Genetics 205, 1079-1088 (2017).

    Parameters
    ----------
    order : int
        order of epistasis
    model_type : str (default="global")
        model matrix type. See publication above for more information
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
    def fit(self, X=None, y=None, sample_weight=None, **kwargs):
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

    def lnlikelihood(self, X=None, ydata=None, yerr=None, thetas=None):
        """Calculate the log likelihood of data, given a set of model coefficients.

        Parameters
        ----------
        X : 2d array
            model matrix
        ydata : array
            data to calculate the likelihood
        yerr: array
            uncertainty in data
        thetas : array
            array of model coefficients

        Returns
        -------
        lnlike : float
            log-likelihood of the data given the model.
        """
        if thetas is None:
            thetas = self.thetas
        if ydata is None:
            ydata = self.gpm.phenotypes
            yerr = self.gpm.std.upper
        if X is None:
            X = self.Xfit
        ymodel = self.hypothesis(X=X, thetas=thetas)
        inv_sigma2 = 1.0/(yerr**2)
        return -0.5*(_np.sum((ydata-ymodel)**2*inv_sigma2 - _np.log(inv_sigma2)))
