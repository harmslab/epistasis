import numpy as np
import pandas as pd

# Scikit-learn classifiers
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import binarize

from epistasis.mapping import EpistasisMap
from epistasis.models.base import BaseModel, use_sklearn
from epistasis.models.utils import (XMatrixException, arghandler)

from epistasis.models.linear import EpistasisLinearRegression

from gpmap import GenotypePhenotypeMap

from .base import EpistasisClassifierMixin

@use_sklearn(GaussianMixture)
class EpistasisGaussianMixture(EpistasisClassifierMixin, BaseModel):
    """Logistic regression for estimating epistatic interactions that lead to
    nonviable phenotypes. Useful for predicting viable/nonviable phenotypes.

    Parameters
    ----------
    order : int
        order of epistasis model

    model_type : str (default="global")
        type of model matrix to use. "global" defines epistasis with respect to
        a background-averaged "genotype-phenotype". "local" defines epistasis
        with respect to the wildtype genotype.
    """
    def __init__(
        self,
        n_components=1,
        model_type="global",
        **kwargs):

        super(self.__class__, self).__init__(n_components=n_components, **kwargs)
        self.model_type = model_type
        self.order = 1
        self.Xbuilt = {}

        # Store model specs.
        self.model_specs = dict(
            model_type=self.model_type,
            **kwargs)

    @arghandler
    def lnlike_of_data(self, X=None, y=None, yerr=None, thetas=None):
        pass

    @arghandler
    def lnlike_transform(
        self,
        X=None,
        y=None,
        yerr=None,
        lnprior=None,
        thetas=None):
        pass

    @arghandler
    def hypothesis(self, X=None, thetas=None):
        pass

    def hypothesis_transform(self, X=None, y=None, thetas=None):
        pass

    @property
    def thetas(self):
        pass
