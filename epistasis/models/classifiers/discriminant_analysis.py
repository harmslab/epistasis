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
        n = 0
        n += self.epistasis.n
        return n

    #@epistasis_fitter
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
        Xclass = self.Additive.Xbuilt['fit'] * self.Additive.epistasis.values
        yclass = binarize(y.reshape(1, -1), self.threshold)[0]

        self = self._fit_(X=Xclass, y=yclass)
        return self

    def _fit_(self, X=None, y=None, **kwargs):
        # Fit the classifier
        super(self.__class__, self).fit(X=X, y=y)
        return self

    def fit_transform(self, X=None, y=None, **kwargs):
        return self.fit(X=X, y=y, **kwargs)

    @arghandler
    def predict(self, X=None):
        self.Additive.predict(X=X)
        Xclass = self.Additive.Xbuilt['predict'] * self.Additive.epistasis.values
        return super(self.__class__, self).predict(X=Xclass)

    def predict_transform(self, X=None, y=None):
        return self.predict(X=X)

    @arghandler
    def score(self, X=None, y=None):
        return super(self.__class__, self).score(X, y)

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
