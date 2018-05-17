import numpy as np
import pandas as pd

# Scikit-learn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import binarize

from epistasis.mapping import EpistasisMap
from epistasis.models.base import BaseModel, use_sklearn
from epistasis.models.utils import (XMatrixException, arghandler)

from epistasis.models.linear import EpistasisLinearRegression

from gpmap import GenotypePhenotypeMap

# # Suppress Deprecation warning
# import warnings
# warnings.filterwarnings(action="ignore", module="sklearn",
#                         category=DeprecationWarning)
#

class EpistasisClassifierMixin:
    """A Mixin class for epistasis classifiers
    """
    def _fit_additive(self, X=None, y=None):
        # Construct an additive model.
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)

        self.Additive.add_gpm(self.gpm)

        # Prepare a high-order model
        self.Additive.epistasis = EpistasisMap(
            sites=self.Additive.Xcolumns,
            order=self.Additive.order,
            model_type=self.Additive.model_type
        )

        # Fit the additive model and infer additive phenotypes
        self.Additive.fit(X=X, y=y)
        return self

    def _fit_classifier(self, X=X, y=y):
        # This method builds x and y from data.
        add_coefs = self.Additive.epistasis.values
        add_X = self.Additive._X(data=X)

        # Project X into padd space.
        X = add_X * add_coefs

        # Label X.
        y = binarize(y.reshape(1, -1), self.threshold)[0]

        # Fit classifier.
        super(self, self.__class__).fit(X=X, y=y)
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

    def fit(self, X=None, y=None, **kwargs):
        # Use Additive model to establish the phenotypic scale.
        # Prepare Additive model
        self._fit_additive(X=X, y=y)
        self._fit_classifier(X=X, y=y)
        return self

    def predict(self, X=None):
        Xadd = self.Additive._X(X=X)
        X = Xadd * self.Additive.epistasis
        return super(self.__class__, self).predict(X=X)

    def predict_transform(self, X=None, y=None):
        x = self.predict(X=X)
        y[x <= 0.5] = self.threshold
        return y

    @arghandler
    def predict_log_proba(self, X=None):
        return super(self.__class__, self).predict_log_proba(X)

    def predict_proba(self, X=None):
        self.Additive.predict(X=X)
        Xclass = self.Additive.Xbuilt['predict'] * self.Additive.epistasis.values
        return super(self.__class__, self).predict_proba(X=Xclass)

    @arghandler
    def score(self, X=None, y=None, **kwargs):
        yclass = binarize(y.reshape(1, -1), threshold=self.threshold)[0]
        return super(self.__class__, self).score(X=X, y=yclass)

    @arghandler
    def lnlike_of_data(self, X=None, y=None, yerr=None, thetas=None):
        # Calculate Y's
        ymodel = self.hypothesis(X=X, thetas=thetas)
        ymodel_ = 1 - ymodel
        ymodel[ymodel < 0.5] = ymodel_[ymodel < 0.5]

        return np.log(ymodel)

    @arghandler
    def lnlike_transform(
            self,
            X=None,
            y=None,
            yerr=None,
            lnprior=None,
            thetas=None):
        # Update likelihood.
        ymodel = self.hypothesis(X=X, thetas=thetas)
        yclass = np.ones(len(ymodel))
        yclass[ymodel > 0.5] = 0

        lnlike = self.lnlike_of_data(X=X, y=y, yerr=yerr, thetas=thetas)
        lnprior[yclass == 0] = 0
        return lnlike + lnprior
