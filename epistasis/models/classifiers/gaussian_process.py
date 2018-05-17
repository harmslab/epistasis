import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import binarize

from epistasis.mapping import EpistasisMap
from epistasis.models.base import BaseModel, use_sklearn
from epistasis.models.utils import arghandler
from epistasis.models.linear import EpistasisLinearRegression

from .base import EpistasisClassifierMixin

# Use if inheriting from a scikit-learn class
@use_sklearn(GaussianProcessClassifier)
class EpistasisGaussianProcess(EpistasisClassifierMixin, BaseModel):
    """testing quadratic
    """
    def __init__(self, order=1, threshold=5, model_type='global', **kwargs):
        self.model_type = model_type
        self.order = 1
        self.Xbuilt = {}
        self.threshold=threshold

        super(self.__class__, self).__init__(**kwargs)

        # Store model specs.
        self.model_specs = dict(
            priors=None,
            threshold=threshold,
            model_type=self.model_type,
            **kwargs)

        # Set up additive linear model for pre-classifying
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)

    @property
    def num_of_params(self):
        n = 0
        n += self.epistasis.n
        return n

    @arghandler
    def fit(self, X=None, y=None, **kwargs):
        # Use Additive model to establish the phenotypic scale.
        # Prepare Additive model
        self._fit_additive(X=X, y=y)
        self._fit_classifier(X=X, y=y)
        return self

    def hypothesis(self, X=None, thetas=None):
        pass

    def hypothesis_transform(self, X=None, y=None, thetas=None):
        pass

    def lnlike_of_data(
        self,
        X=None,
        y=None,
        yerr=None,
        thetas=None):

        pass

    def lnlike_transform(
        self,
        X=None,
        y=None,
        yerr=None,
        lnprior=None,
        thetas=None):
        pass
