import numpy as np
import pandas as pd

from epistasis.models.base import BaseModel, use_sklearn
from epistasis.models.utils import arghandler


# Use if inheriting from a scikit-learn class
#@use_sklearn()
class EpistasisModel(BaseModel):
    """Epistasis model template.
    """
    def __init__(self, ):
        pass

    @property
    def num_of_params(self):
        pass

    #@arghandler
    def fit(self, X=None, y=None, **kwargs):
        pass

    def fit_transform(self, X=None, y=None, **kwargs):
        pass

    def predict(self, X=None):
        pass

    def predict_transform(self, X=None, y=None, **kwargs):
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
