import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import binarize

from epistasis.mapping import EpistasisMap
from epistasis.models.base import BaseModel, use_sklearn
from epistasis.models.utils import arghandler
from epistasis.models.linear import EpistasisLinearRegression

# Use if inheriting from a scikit-learn class
@use_sklearn(QuadraticDiscriminantAnalysis)
class EpistasisQuadraticDA(BaseModel):
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
        pass

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
