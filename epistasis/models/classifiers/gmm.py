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
#
# # Suppress Deprecation warning
# import warnings
# warnings.filterwarnings(action="ignore", module="sklearn",
#                         category=DeprecationWarning)


@use_sklearn(GaussianMixture)
class EpistasisGaussianMixture(BaseModel):
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

        # Set up additive linear model for pre-classifying
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)

    @arghandler
    def fit(self, X=None, y=None, **kwargs):
        # Use Additive model to establish the phenotypic scale.
        # Prepare Additive model
        self.Additive.add_gpm(self.gpm)

        # Prepare a high-order model
        self.Additive.epistasis = EpistasisMap(
            sites=self.Additive.Xcolumns,
            order=self.Additive.order,
            model_type=self.Additive.model_type
        )

        # Fit the additive model and infer additive phenotypes
        self.Additive.fit(X=X, y=y)

        super(self.__class__, self).fit(X=X, y=y)
        return self

    def fit_transform(self, X=None, y=None, **kwargs):
        self.fit(X=X, y=y, **kwargs)
        ypred = self.predict(X=X)


        # Transform map.
        gpm = GenotypePhenotypeMap.read_dataframe(
            dataframe=self.gpm.data[ypred==1],
            wildtype=self.gpm.wildtype,
            mutations=self.gpm.mutations
        )
        return gpm

    @property
    def num_of_params(self):
        n = 0
        n += self.epistasis.n
        return n

    @arghandler
    def predict(self, X=None):
        return super(self.__class__, self).predict(X=X)

    def predict_transform(self, X=None, y=None):
        pass

    @arghandler
    def predict_log_proba(self, X=None):
        return super(self.__class__, self).predict_log_proba(X)

    @arghandler
    def predict_proba(self, X=None):
        return super(self.__class__, self).predict_proba(X)

    @arghandler
    def score(self, X=None, y=None, **kwargs):
        return super(self.__class__, self).score(X=X, y=y)

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
