import numpy as np
import pandas as pd

# Scikit-learn classifiers
from sklearn.preprocessing import binarize

from epistasis.models.base import BaseModel, use_sklearn
from epistasis.models.utils import (XMatrixException, arghandler)

from .base import EpistasisClassifierMixin

#@use_sklearn(sklearn_class)
class EpistasisClassiferClass(EpistasisClassifierMixin, BaseModel):
    """Template for EpistasisClassifier
    """
    def __init__(self, threshold, model_type="global", **kwargs):
        pass
        super().__init__(**kwargs)

        # Store model specs.
        self.model_specs = dict(
            threshold=self.threshold,
            model_type=self.model_type,
            **kwargs)

    #@arghandler
    def fit(self, X=None, y=None, **kwargs):
        # Use Additive model to establish the phenotypic scale.
        # Prepare Additive model
        self._fit_additive(X=X, y=y)
        self._fit_classifier(X=X, y=y)
        return self

    @property
    def num_of_params(self):
        pass

    #@arghandler
    def score(self, X=None, y=None, **kwargs):
        pass

    #@arghandler
    def lnlike_of_data(self, X=None, y=None, yerr=None, thetas=None):
        pass

    #@arghandler
    def lnlike_transform(
        self,
        X=None,
        y=None,
        yerr=None,
        lnprior=None,
        thetas=None):
        pass

    #@arghandler
    def hypothesis(self, X=None, thetas=None):
        pass

    def hypothesis_transform(self, X=None, y=None, thetas=None):
        pass

    @property
    def thetas(self):
        pass
