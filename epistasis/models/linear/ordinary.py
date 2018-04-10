import numpy as np
from sklearn.linear_model import LinearRegression

from ..base import BaseModel, use_sklearn
from ..utils import arghandler

# Suppress an annoying error from scikit-learn
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

@use_sklearn(LinearRegression)
class EpistasisLinearRegression(BaseModel):
    """Ordinary least-squares regression for estimating high-order, epistatic
    interactions in a genotype-phenotype map.

    Methods are described in the following publication:
        Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in
        Nonlinear Genotype-Phenotype Maps'. Genetics 205, 1079-1088 (2017).

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
        self.Xbuilt = {}

        # Store model specs.
        self.model_specs = dict(
            order=self.order,
            model_type=self.model_type,
            **kwargs)

    @property
    def num_of_params(self):
        n = 0
        n += self.epistasis.n
        return n

    #@epistasis_fitter
    @arghandler
    def fit(self, X=None, y=None, **kwargs):
        self = super(self.__class__, self).fit(X, y)

        # Link coefs to epistasis values.
        self.epistasis.values = np.reshape(self.coef_, (-1,))
        return self

    def fit_transform(self, X=None, y=None, **kwargs):
        return self.fit(X=X, y=y, **kwargs)

    @arghandler
    def predict(self, X=None):
        return super(self.__class__, self).predict(X)

    def predict_transform(self, X=None, y=None):
        return self.predict(X=X)

    @arghandler
    def score(self, X=None, y=None):
        return super(self.__class__, self).score(X, y)

    @property
    def thetas(self):
        return self.coef_

    @arghandler
    def hypothesis(self, X=None, thetas=None):
        return np.dot(X, thetas)

    def hypothesis_transform(self, X=None, y=None, thetas=None):
        return self.hypothesis(X=X, thetas=thetas)

    @arghandler
    def lnlike_of_data(
            self,
            X=None,
            y=None,
            yerr=None,
            thetas=None):
        # Calculate y from model.
        ymodel = self.hypothesis(X=X, thetas=thetas)
        return (- 0.5 * np.log(2 * np.pi * yerr**2) -
                (0.5 * ((y - ymodel)**2 / yerr**2)))

    @arghandler
    def lnlike_transform(
            self,
            X=None,
            y=None,
            yerr=None,
            lnprior=None,
            thetas=None):
        # Update likelihood.
        lnlike = self.lnlike_of_data(X=X, y=y, yerr=yerr, thetas=thetas)
        return lnlike + lnprior
